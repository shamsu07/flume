from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Generic, TypeVar

from flume.metrics import ROUTER_AFFINITY, ROUTER_UNAVAILABLE

HealthChecker = Callable[[str], Awaitable[bool]]
T = TypeVar("T")


class NoHealthyWorkers(RuntimeError):
    """Raised immediately when the cached health snapshot has no usable worker."""


class PackRouter:
    """Health-aware rendezvous hashing with a background-only health hot path."""

    def __init__(
        self,
        worker_urls: list[str],
        health_checker: HealthChecker | None = None,
        *,
        refresh_seconds: float = 5.0,
    ):
        if not worker_urls:
            raise ValueError("at least one vLLM worker is required")
        self.worker_urls = tuple(dict.fromkeys(url.rstrip("/") for url in worker_urls))
        self.health_checker = health_checker
        self.refresh_seconds = refresh_seconds
        initial_health = health_checker is None
        self._health = {worker: initial_health for worker in self.worker_urls}
        self._monitor_task: asyncio.Task[None] | None = None
        self._assignments: dict[str, str] = {}
        self._affinity: Counter[str] = Counter()

    async def start(self) -> None:
        if self.health_checker is None or self._monitor_task is not None:
            return
        await self.refresh_health()
        self._monitor_task = asyncio.create_task(
            self._monitor(),
            name="flume-worker-health-monitor",
        )

    async def close(self) -> None:
        if self._monitor_task is None:
            return
        self._monitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._monitor_task
        self._monitor_task = None

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_seconds)
            await self.refresh_health()

    async def refresh_health(self) -> dict[str, bool]:
        health_checker = self.health_checker
        if health_checker is None:
            return dict(self._health)

        async def checked(worker: str) -> tuple[str, bool]:
            try:
                return worker, bool(await health_checker(worker))
            except Exception:
                return worker, False

        results = await asyncio.gather(*(checked(worker) for worker in self.worker_urls))
        self._health = dict(results)
        return dict(self._health)

    async def choose(self, pack_id: str, *, exclude: set[str] | None = None) -> str:
        excluded = exclude or set()
        candidates = [
            worker
            for worker in self.worker_urls
            if self._health.get(worker, False) and worker not in excluded
        ]
        if not candidates:
            ROUTER_UNAVAILABLE.inc()
            raise NoHealthyWorkers("no healthy vLLM workers")
        worker = max(candidates, key=lambda item: self._score(pack_id, item))
        previous = self._assignments.get(pack_id)
        if previous == worker:
            self._affinity["hit"] += 1
            ROUTER_AFFINITY.labels(result="hit").inc()
        else:
            self._affinity["miss"] += 1
            ROUTER_AFFINITY.labels(result="miss").inc()
            self._assignments[pack_id] = worker
        return worker

    def mark_unhealthy(self, worker_url: str) -> None:
        if worker_url in self._health:
            updated = dict(self._health)
            updated[worker_url] = False
            self._health = updated

    def health(self) -> dict[str, bool]:
        return dict(self._health)

    def stats(self) -> dict[str, dict[str, int]]:
        assigned = Counter(self._assignments.values())
        return {
            worker: {
                "assigned_packs": assigned[worker],
                "affinity_hits": self._affinity["hit"],
                "affinity_misses": self._affinity["miss"],
            }
            for worker in self.worker_urls
        }

    @staticmethod
    def _score(pack_id: str, worker_url: str) -> int:
        digest = hashlib.blake2b(
            f"{pack_id}\0{worker_url}".encode(),
            digest_size=16,
            person=b"flume-route-v1",
        ).digest()
        return int.from_bytes(digest, "big")


class WarmupSingleFlight(Generic[T]):
    """Coalesce concurrent warmups for the same tenant, pack, and worker."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[tuple[str, str, str], asyncio.Future[T]] = {}

    async def run(
        self,
        key: tuple[str, str, str],
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        async with self._lock:
            task = self._tasks.get(key)
            if task is None:
                task = asyncio.ensure_future(operation())
                self._tasks[key] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._lock:
                    if self._tasks.get(key) is task:
                        del self._tasks[key]
