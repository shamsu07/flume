from __future__ import annotations

import json
import math
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

from flume.router import WorkerLoad

Prompt = str | list[int]
RESERVED_EXTRA_FIELDS = {
    "cache_salt",
    "max_tokens",
    "model",
    "prompt",
    "stop",
    "stream",
    "temperature",
    "top_p",
}


class VLLMError(RuntimeError):
    """Base error safe for API translation without upstream response bodies."""


class VLLMConnectionError(VLLMError):
    pass


class VLLMUpstreamError(VLLMError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"vLLM returned HTTP {status_code}")


@dataclass(slots=True)
class CompletionResult:
    text: str
    latency_ms: float
    ttft_ms: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None


class VLLMStream:
    """An already validated upstream stream that forwards raw bytes unchanged."""

    def __init__(
        self,
        response: httpx.Response,
        *,
        started: float,
        on_first_token: Callable[[float], None] | None = None,
        on_done: Callable[[float], None] | None = None,
    ):
        self.response = response
        self.started = started
        self.on_first_token = on_first_token
        self.on_done = on_done
        self._first_token_seen = False
        self._buffer = b""
        self._event_data: list[str] = []
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.response.aiter_raw():
                self._inspect(chunk)
                yield chunk
        finally:
            await self.aclose()

    def _inspect(self, chunk: bytes) -> None:
        if self._first_token_seen:
            return
        self._buffer += chunk
        while b"\n" in self._buffer:
            raw_line, self._buffer = self._buffer.split(b"\n", 1)
            if raw_line.endswith(b"\r"):
                raw_line = raw_line[:-1]
            self._inspect_line(raw_line.decode("utf-8", errors="replace"))
            if self._first_token_seen:
                self._buffer = b""
                return

    def _inspect_line(self, line: str) -> None:
        if not line:
            data = "\n".join(self._event_data)
            self._event_data.clear()
            if VLLMClient.data_has_token(data):
                self._first_token_seen = True
                if self.on_first_token is not None:
                    self.on_first_token((time.perf_counter() - self.started) * 1000)
            return
        if line.startswith(":"):
            return
        field, separator, value = line.partition(":")
        if field != "data":
            return
        if separator and value.startswith(" "):
            value = value[1:]
        self._event_data.append(value)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.response.aclose()
        finally:
            if self.on_done is not None:
                self.on_done((time.perf_counter() - self.started) * 1000)


class VLLMClient:
    def __init__(
        self,
        timeout_seconds: float = 120.0,
        *,
        connect_timeout_seconds: float = 5.0,
        health_timeout_seconds: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.health_timeout_seconds = health_timeout_seconds
        self.transport = transport
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is not None:
            return
        timeout = httpx.Timeout(self.timeout_seconds, connect=self.connect_timeout_seconds)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            transport=self.transport,
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("vLLM client has not been started")
        return self._client

    async def health(self, worker_url: str) -> bool:
        try:
            response = await self.client.get(
                f"{worker_url.rstrip('/')}/health",
                timeout=self.health_timeout_seconds,
            )
            return response.is_success
        except httpx.HTTPError:
            return False

    async def load(self, worker_url: str) -> WorkerLoad | None:
        try:
            response = await self.client.get(
                f"{worker_url.rstrip('/')}/metrics",
                timeout=self.health_timeout_seconds,
            )
        except httpx.HTTPError:
            return None
        if not response.is_success:
            return None
        try:
            values = self._request_load_values(response.text)
        except ValueError:
            return None
        if values is None:
            return None
        running, waiting = values
        return WorkerLoad(running=running, waiting=waiting)

    @staticmethod
    def _request_load_values(metrics_text: str) -> tuple[float, float] | None:
        metric_names = {
            "vllm:num_requests_running": "running",
            "vllm_num_requests_running": "running",
            "vllm:num_requests_waiting": "waiting",
            "vllm_num_requests_waiting": "waiting",
        }
        totals = {"running": 0.0, "waiting": 0.0}
        found: set[str] = set()
        for family in text_string_to_metric_families(metrics_text):
            for sample in family.samples:
                kind = metric_names.get(sample.name)
                if kind is None:
                    continue
                value = float(sample.value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid vLLM request load")
                totals[kind] += value
                found.add(kind)
        if found != {"running", "waiting"}:
            return None
        return totals["running"], totals["waiting"]

    async def complete(
        self,
        *,
        worker_url: str,
        model: str,
        prompt: Prompt,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        stop: str | list[str] | None = None,
        cache_salt: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        payload = self._payload(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=False,
            cache_salt=cache_salt,
            extra_body=extra_body,
        )
        started = time.perf_counter()
        try:
            response = await self.client.post(
                f"{worker_url.rstrip('/')}/v1/completions",
                json=payload,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise VLLMConnectionError("could not connect to vLLM") from exc
        if response.is_error:
            raise VLLMUpstreamError(response.status_code)
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise VLLMError("vLLM returned invalid JSON") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        choice = (body.get("choices") or [{}])[0]
        usage = body.get("usage") or {}
        return CompletionResult(
            text=choice.get("text") or "",
            latency_ms=latency_ms,
            ttft_ms=None,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )

    async def open_stream_completion(
        self,
        *,
        worker_url: str,
        model: str,
        prompt: Prompt,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        stop: str | list[str] | None = None,
        cache_salt: str | None = None,
        extra_body: dict[str, Any] | None = None,
        on_first_token: Callable[[float], None] | None = None,
        on_done: Callable[[float], None] | None = None,
    ) -> VLLMStream:
        payload = self._payload(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
            cache_salt=cache_salt,
            extra_body=extra_body,
        )
        request = self.client.build_request(
            "POST",
            f"{worker_url.rstrip('/')}/v1/completions",
            json=payload,
            timeout=self.timeout_seconds,
        )
        started = time.perf_counter()
        try:
            response = await self.client.send(request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise VLLMConnectionError("could not connect to vLLM") from exc
        if response.is_error:
            status_code = response.status_code
            await response.aclose()
            raise VLLMUpstreamError(status_code)
        return VLLMStream(
            response,
            started=started,
            on_first_token=on_first_token,
            on_done=on_done,
        )

    async def stream_completion(self, **kwargs: Any) -> AsyncIterator[bytes]:
        stream = await self.open_stream_completion(**kwargs)
        async for chunk in stream:
            yield chunk

    @staticmethod
    def _payload(
        *,
        model: str,
        prompt: Prompt,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: str | list[str] | None,
        stream: bool,
        cache_salt: str | None,
        extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        extra_body = extra_body or {}
        reserved = sorted(RESERVED_EXTRA_FIELDS.intersection(extra_body))
        if reserved:
            raise ValueError(f"extra_body contains reserved fields: {', '.join(reserved)}")
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **extra_body,
        }
        if stop:
            payload["stop"] = stop
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt
        return payload

    @staticmethod
    def line_has_token(line: str) -> bool:
        if not line.startswith("data:"):
            return False
        data = line.removeprefix("data:").strip()
        return VLLMClient.data_has_token(data)

    @staticmethod
    def data_has_token(data: str) -> bool:
        if not data:
            return False
        if data == "[DONE]":
            return False
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return False
        return any(choice.get("text") for choice in payload.get("choices") or [])
