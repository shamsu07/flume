from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Generic, TypeVar

from flume.metrics import ROUTER_AFFINITY, ROUTER_UNAVAILABLE

HealthChecker = Callable[[str], Awaitable[bool]]
T = TypeVar("T")


class NoHealthyWorkers(RuntimeError):
    """Raised immediately when the cached health snapshot has no usable worker."""


@dataclass(frozen=True, slots=True)
class RoutingState:
    worker_url: str
    last_seen: float
    spill_until: float


@dataclass(frozen=True, slots=True)
class WorkerLoad:
    running: float
    waiting: float


@dataclass(frozen=True, slots=True)
class WorkerLoadSnapshot:
    running: float
    waiting: float
    observed_at: float


LoadChecker = Callable[[str], Awaitable[WorkerLoad | None]]


class PackRouter:
    """Health-aware rendezvous hashing with a background-only health hot path."""

    def __init__(
        self,
        worker_urls: list[str],
        health_checker: HealthChecker | None = None,
        load_checker: LoadChecker | None = None,
        *,
        refresh_seconds: float = 5.0,
        load_refresh_seconds: float = 0.5,
        load_stale_seconds: float = 2.0,
        routing_policy: str = "hrw",
        load_slack: int = 2,
        spill_hold_seconds: float = 2.0,
        capacity_weights: dict[str, float] | None = None,
        state_max_entries: int = 10_000,
        state_ttl_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not worker_urls:
            raise ValueError("at least one vLLM worker is required")
        if routing_policy not in {"hrw", "bounded_hrw"}:
            raise ValueError("routing_policy must be 'hrw' or 'bounded_hrw'")
        if load_slack < 0:
            raise ValueError("load_slack cannot be negative")
        if spill_hold_seconds < 0:
            raise ValueError("spill_hold_seconds cannot be negative")
        if load_refresh_seconds <= 0:
            raise ValueError("load_refresh_seconds must be positive")
        if load_stale_seconds < load_refresh_seconds:
            raise ValueError("load_stale_seconds must be at least load_refresh_seconds")
        if state_max_entries < 1:
            raise ValueError("state_max_entries must be positive")
        if state_ttl_seconds <= 0:
            raise ValueError("state_ttl_seconds must be positive")
        self.worker_urls = tuple(dict.fromkeys(url.rstrip("/") for url in worker_urls))
        weights = capacity_weights or {}
        if set(weights).difference(self.worker_urls):
            raise ValueError("capacity_weights contains an unknown worker")
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError("capacity_weights must be greater than zero")
        self.health_checker = health_checker
        self.load_checker = load_checker
        self.refresh_seconds = refresh_seconds
        self.load_refresh_seconds = load_refresh_seconds
        self.load_stale_seconds = load_stale_seconds
        self.routing_policy = routing_policy
        self.load_slack = load_slack
        self.spill_hold_seconds = spill_hold_seconds
        self.capacity_weights = {
            worker: weights.get(worker, 1.0) for worker in self.worker_urls
        }
        self.state_max_entries = state_max_entries
        self.state_ttl_seconds = state_ttl_seconds
        self._clock = clock
        initial_health = health_checker is None
        self._health = {worker: initial_health for worker in self.worker_urls}
        self._health_monitor_task: asyncio.Task[None] | None = None
        self._load_monitor_task: asyncio.Task[None] | None = None
        self._worker_load: dict[str, WorkerLoadSnapshot | None] = {
            worker: None for worker in self.worker_urls
        }
        self._assignments: OrderedDict[str, RoutingState] = OrderedDict()
        self._affinity: Counter[str] = Counter()
        self._local_in_flight: Counter[str] = Counter()

    async def start(self) -> None:
        if self.health_checker is not None and self._health_monitor_task is None:
            await self.refresh_health()
            self._health_monitor_task = asyncio.create_task(
                self._monitor_health(),
                name="flume-worker-health-monitor",
            )
        if self.load_checker is not None and self._load_monitor_task is None:
            await self.refresh_load()
            self._load_monitor_task = asyncio.create_task(
                self._monitor_load(),
                name="flume-worker-load-monitor",
            )

    async def close(self) -> None:
        tasks = [
            task
            for task in (self._health_monitor_task, self._load_monitor_task)
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._health_monitor_task = None
        self._load_monitor_task = None

    async def _monitor_health(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_seconds)
            await self.refresh_health()

    async def _monitor_load(self) -> None:
        while True:
            await asyncio.sleep(self.load_refresh_seconds)
            await self.refresh_load()

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

    async def refresh_load(self) -> dict[str, WorkerLoadSnapshot | None]:
        load_checker = self.load_checker
        if load_checker is None:
            return dict(self._worker_load)

        async def checked(worker: str) -> tuple[str, WorkerLoad | None]:
            try:
                load = await load_checker(worker)
            except Exception:
                return worker, None
            if load is None:
                return worker, None
            if (
                not math.isfinite(load.running)
                or not math.isfinite(load.waiting)
                or load.running < 0
                or load.waiting < 0
            ):
                return worker, None
            return worker, load

        results = await asyncio.gather(*(checked(worker) for worker in self.worker_urls))
        observed_at = self._clock()
        updated = dict(self._worker_load)
        for worker, load in results:
            if load is not None:
                updated[worker] = WorkerLoadSnapshot(
                    running=load.running,
                    waiting=load.waiting,
                    observed_at=observed_at,
                )
        self._worker_load = updated
        return dict(self._worker_load)

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
        ranked = sorted(candidates, key=lambda item: self._score(pack_id, item), reverse=True)
        primary = ranked[0]
        now = self._clock()
        previous = self._get_state(pack_id, now)
        worker = primary
        if self.routing_policy == "bounded_hrw" and len(ranked) > 1:
            if (
                previous is not None
                and previous.worker_url != primary
                and previous.worker_url in candidates
                and previous.spill_until > now
            ):
                worker = previous.worker_url
            else:
                effective_loads = {
                    candidate: self._effective_load(candidate, now) for candidate in candidates
                }
                minimum_load = min(effective_loads.values())
                load_bound = minimum_load + self.load_slack
                if effective_loads[primary] > load_bound:
                    worker = next(
                        candidate
                        for candidate in ranked[1:]
                        if effective_loads[candidate] <= load_bound
                    )

        if previous is not None and previous.worker_url == worker:
            self._affinity["hit"] += 1
            ROUTER_AFFINITY.labels(result="hit").inc()
        else:
            self._affinity["miss"] += 1
            ROUTER_AFFINITY.labels(result="miss").inc()
        spill_until = now + self.spill_hold_seconds if worker != primary else 0.0
        self._record_state(
            pack_id,
            RoutingState(worker_url=worker, last_seen=now, spill_until=spill_until),
            now,
        )
        return worker

    def _get_state(self, pack_id: str, now: float) -> RoutingState | None:
        state = self._assignments.get(pack_id)
        if state is None:
            return None
        if now - state.last_seen >= self.state_ttl_seconds:
            del self._assignments[pack_id]
            return None
        self._assignments.move_to_end(pack_id)
        return state

    def _record_state(self, pack_id: str, state: RoutingState, now: float) -> None:
        self._assignments[pack_id] = state
        self._assignments.move_to_end(pack_id)
        while self._assignments:
            oldest_pack_id, oldest = next(iter(self._assignments.items()))
            if (
                len(self._assignments) <= self.state_max_entries
                and now - oldest.last_seen < self.state_ttl_seconds
            ):
                break
            del self._assignments[oldest_pack_id]

    def _effective_load(self, worker_url: str, now: float | None = None) -> float:
        observed_at = self._clock() if now is None else now
        upstream_load = 0.0
        snapshot = self._worker_load[worker_url]
        if (
            snapshot is not None
            and observed_at - snapshot.observed_at < self.load_stale_seconds
        ):
            upstream_load = snapshot.running + snapshot.waiting
        return (
            self._local_in_flight[worker_url] + upstream_load
        ) / self.capacity_weights[worker_url]

    def mark_unhealthy(self, worker_url: str) -> None:
        if worker_url in self._health:
            updated = dict(self._health)
            updated[worker_url] = False
            self._health = updated

    def acquire(self, worker_url: str) -> None:
        if worker_url not in self._health:
            raise ValueError("worker is not managed by this router")
        self._local_in_flight[worker_url] += 1

    def release(self, worker_url: str) -> None:
        current = self._local_in_flight[worker_url]
        if current <= 1:
            self._local_in_flight.pop(worker_url, None)
        else:
            self._local_in_flight[worker_url] = current - 1

    def local_in_flight(self, worker_url: str) -> int:
        return self._local_in_flight[worker_url]

    def health(self) -> dict[str, bool]:
        return dict(self._health)

    def stats(self) -> dict[str, dict[str, int]]:
        assigned = Counter(state.worker_url for state in self._assignments.values())
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
