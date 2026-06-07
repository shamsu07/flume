from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from flume.compiler import ContextPackCompiler
from flume.config import Settings
from flume.models import BenchmarkRun, DocumentChunk, PackCreateRequest
from flume.router import PackRouter
from flume.store import FlumeStore
from flume.vllm import VLLMClient


@dataclass(slots=True)
class BenchmarkRunner:
    store: FlumeStore
    compiler: ContextPackCompiler
    router: PackRouter
    vllm: VLLMClient
    settings: Settings

    async def run(self, run: BenchmarkRun) -> BenchmarkRun:
        try:
            results = {}
            for context_length in run.request.context_lengths:
                pack = self.compiler.compile(
                    PackCreateRequest(
                        tenant_id="bench",
                        model_id=self.settings.model_id,
                        tokenizer_id=self.settings.tokenizer_id,
                        template_id=f"bench-{context_length}",
                        chunks=[self._synthetic_chunk(context_length)],
                    )
                )
                self.store.save_pack(pack)
                worker_url = await self.router.choose(pack.pack_id)

                if run.request.baseline == "flume_warm_affinity":
                    await self.vllm.complete(
                        worker_url=worker_url,
                        model=pack.model_id,
                        prompt=f"{pack.compiled_prefix}warm\n\nAnswer:\n",
                        max_tokens=1,
                        temperature=0.0,
                    )

                samples = []
                for iteration in range(run.request.iterations):
                    question = run.request.questions[iteration % len(run.request.questions)]
                    started = time.perf_counter()
                    result = await self.vllm.complete(
                        worker_url=worker_url,
                        model=pack.model_id,
                        prompt=f"{pack.compiled_prefix}{question}\n\nAnswer:\n",
                        max_tokens=64,
                        temperature=0.0,
                    )
                    wall_ms = (time.perf_counter() - started) * 1000
                    samples.append(
                        {
                            "iteration": iteration,
                            "latency_ms": result.latency_ms or wall_ms,
                            "wall_ms": wall_ms,
                            "output_tokens": result.output_tokens,
                        }
                    )

                latencies = [sample["latency_ms"] for sample in samples]
                results[str(context_length)] = {
                    "pack_id": pack.pack_id,
                    "worker_url": worker_url,
                    "token_count": pack.token_count,
                    "samples": samples,
                    "latency_ms": _summary(latencies),
                }

            run.results = results
            run.status = "completed"
            return run
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            return run

    def _synthetic_chunk(self, context_length: int) -> DocumentChunk:
        base = (
            "Flume benchmark context. This synthetic policy text is intentionally repetitive "
            "so that the prompt reaches the requested context size while remaining deterministic. "
        )
        approx_chars = max(context_length * 4, len(base))
        text = (base * ((approx_chars // len(base)) + 1))[:approx_chars]
        return DocumentChunk(
            doc_id=f"synthetic-{context_length}",
            chunk_id="0",
            version="1",
            text=text,
            metadata={"target_context_tokens": context_length},
        )


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
