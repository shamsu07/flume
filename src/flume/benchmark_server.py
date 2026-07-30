"""Private subprocess servers used by the local benchmark harness."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse


def create_mock_worker(worker_id: str, delay_ms: float = 0.0) -> FastAPI:
    app = FastAPI(title=f"Flume benchmark worker {worker_id}")
    state: dict[str, float | int] = {"completions": 0, "delay_ms": delay_ms}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            "flume_benchmark_synthetic_prefix_cache_queries "
            f"{state['completions']}\n"
            "flume_benchmark_synthetic_prefix_cache_hits "
            f"{max(0, state['completions'] - 1)}\n"
        )

    @app.post("/benchmark/control")
    async def control(payload: dict[str, Any]) -> dict[str, float]:
        requested_delay = float(payload.get("delay_ms", 0.0))
        if requested_delay < 0 or requested_delay > 10_000:
            raise HTTPException(status_code=422, detail="delay_ms must be between 0 and 10000")
        state["delay_ms"] = requested_delay
        return {"delay_ms": requested_delay}

    @app.post("/v1/completions")
    async def completions(payload: dict[str, Any]) -> Any:
        state["completions"] += 1
        prompt = payload.get("prompt", [])
        prompt_tokens = len(prompt) if isinstance(prompt, list) else len(str(prompt))
        completion_tokens = int(payload.get("max_tokens", 1))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        choice = {
            "text": "x" * completion_tokens,
            "index": 0,
            "logprobs": None,
            "finish_reason": "length",
        }
        if payload.get("stream"):

            async def events() -> AsyncIterator[bytes]:
                if state["delay_ms"]:
                    await asyncio.sleep(float(state["delay_ms"]) / 1000)
                yield f"data: {json.dumps({'choices': [choice]})}\n\n".encode()
                yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        if state["delay_ms"]:
            await asyncio.sleep(float(state["delay_ms"]) / 1000)
        return {
            "id": f"mock-{worker_id}-{state['completions']}",
            "object": "text_completion",
            "created": 0,
            "model": payload.get("model", "local-benchmark"),
            "choices": [choice],
            "usage": usage,
        }

    return app


def run_worker(args: argparse.Namespace) -> None:
    uvicorn.run(
        create_mock_worker(args.worker_id, args.delay_ms),
        host="127.0.0.1",
        port=args.port,
        log_level="error",
        access_log=False,
    )


def run_flume(args: argparse.Namespace) -> None:
    from flume.api import create_app
    from flume.compiler import (
        ContextPackCompiler,
        DeterministicByteTokenizer,
        Tokenizer,
    )
    from flume.config import Settings

    tokenizer = DeterministicByteTokenizer()
    settings = Settings(
        host="127.0.0.1",
        port=args.port,
        database_url=args.database_url,
        vllm_workers=args.worker_url,
        model_id="local-benchmark",
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_revision=tokenizer.revision,
        request_timeout_seconds=10.0,
        connect_timeout_seconds=1.0,
        health_timeout_seconds=1.0,
        health_refresh_seconds=0.1,
        cache_salt_secret="local-benchmark-secret-not-for-production",
        metrics_enabled=True,
    )
    compiler = ContextPackCompiler(
        cast(Tokenizer, tokenizer),
        model_id=settings.model_id,
    )
    uvicorn.run(
        create_app(settings, compiler=compiler),
        host="127.0.0.1",
        port=args.port,
        log_level="error",
        access_log=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="server", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--port", type=int, required=True)
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--delay-ms", type=float, default=0.0)
    worker.set_defaults(handler=run_worker)

    flume = subparsers.add_parser("flume")
    flume.add_argument("--port", type=int, required=True)
    flume.add_argument("--database-url", required=True)
    flume.add_argument("--worker-url", action="append", required=True)
    flume.set_defaults(handler=run_flume)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
