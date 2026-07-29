from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class CompletionResult:
    text: str
    latency_ms: float
    ttft_ms: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None


class VLLMClient:
    def __init__(
        self,
        timeout_seconds: float = 120.0,
        *,
        connect_timeout_seconds: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
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
        response = await self.client.get(
            f"{worker_url.rstrip('/')}/health",
            timeout=self.connect_timeout_seconds,
        )
        return response.status_code < 500

    async def complete(
        self,
        *,
        worker_url: str,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        if extra_body:
            payload.update(extra_body)

        started = time.perf_counter()
        response = await self.client.post(
            f"{worker_url.rstrip('/')}/v1/completions",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
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

    async def stream_completion(
        self,
        *,
        worker_url: str,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        extra_body: dict[str, Any] | None = None,
        on_first_token: Callable[[float], None] | None = None,
        on_done: Callable[[float], None] | None = None,
    ) -> AsyncIterator[bytes]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        if extra_body:
            payload.update(extra_body)

        started = time.perf_counter()
        first_token_seen = False
        async with self.client.stream(
            "POST",
            f"{worker_url.rstrip('/')}/v1/completions",
            json=payload,
            timeout=self.timeout_seconds,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    yield b"\n"
                    continue
                if not first_token_seen and self._line_has_token(line):
                    first_token_seen = True
                    ttft_ms = (time.perf_counter() - started) * 1000
                    if on_first_token:
                        on_first_token(ttft_ms)
                yield f"{line}\n\n".encode()
        latency_ms = (time.perf_counter() - started) * 1000
        if on_done:
            on_done(latency_ms)

    def _line_has_token(self, line: str) -> bool:
        if not line.startswith("data: "):
            return bool(line.strip())
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            return False
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return False
        for choice in payload.get("choices") or []:
            if choice.get("text"):
                return True
        return False
