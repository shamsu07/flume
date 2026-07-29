from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from flume.metrics import (
    ROUTER_AFFINITY,
    ROUTER_DECISION_LATENCY,
    ROUTER_DECISIONS,
    ROUTER_UNAVAILABLE,
    ROUTER_WORKER_LOAD,
    ROUTER_WORKER_LOAD_FRESH,
)

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


@dataclass(frozen=True, slots=True)
class WorkerRoutingStats:
    assigned_packs: int
    affinity_hits: int
    affinity_misses: int
    local_in_flight: int
    upstream_running: float | None
    upstream_waiting: float | None
    effective_load: float
    load_fresh: bool
    capacity_weight: float


LoadChecker = Callable[[str], Awaitable[WorkerLoad | None]]


@dataclass(slots=True)
class WorkerLease:
    """Idempotent ownership token for one local in-flight worker slot."""

    _router: PackRouter
    worker_url: str
    _released: bool = False

    def release(self) -> bool:
        if self._released:
            return False
        self._released = True
        self._router._release_worker(self.worker_url)
        return True


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
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
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
        self._worker_affinity = {
            worker: Counter[str]() for worker in self.worker_urls
        }
        self._local_in_flight: Counter[str] = Counter()

    async def start(self) -> None:
        try:
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
        except BaseException:
            await self._close_monitor_tasks()
            raise

    async def close(self) -> None:
        await self._close_monitor_tasks()

    async def _close_monitor_tasks(self) -> None:
        tasks = [
            task
            for task in (self._health_monitor_task, self._load_monitor_task)
            if task is not None
        ]
        self._health_monitor_task = None
        self._load_monitor_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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

        async def checked(worker: str) -> tuple[str, WorkerLoad | None, float | None]:
            try:
                load = await load_checker(worker)
            except Exception:
                return worker, None, None
            if load is None:
                return worker, None, None
            if (
                not math.isfinite(load.running)
                or not math.isfinite(load.waiting)
                or load.running < 0
                or load.waiting < 0
            ):
                return worker, None, None
            return worker, load, self._clock()

        results = await asyncio.gather(*(checked(worker) for worker in self.worker_urls))
        updated = dict(self._worker_load)
        for worker, load, observed_at in results:
            if load is not None and observed_at is not None:
                updated[worker] = WorkerLoadSnapshot(
                    running=load.running,
                    waiting=load.waiting,
                    observed_at=observed_at,
                )
        self._worker_load = updated
        self._publish_worker_snapshot_metrics(self._clock())
        return dict(self._worker_load)

    async def choose(self, pack_id: str, *, exclude: set[str] | None = None) -> str:
        started = time.perf_counter()
        try:
            excluded = exclude or set()
            candidates = [
                worker
                for worker in self.worker_urls
                if self._health.get(worker, False) and worker not in excluded
            ]
            if not candidates:
                ROUTER_UNAVAILABLE.inc()
                raise NoHealthyWorkers("no healthy vLLM workers")
            if self.routing_policy == "hrw":
                ranked: list[str] | None = None
                primary = max(candidates, key=lambda item: self._score(pack_id, item))
            else:
                ranked = sorted(
                    candidates,
                    key=lambda item: self._score(pack_id, item),
                    reverse=True,
                )
                primary = ranked[0]
            now = self._clock()
            previous = self._get_state(pack_id, now)
            worker = primary
            decision = "primary"
            if self.routing_policy == "bounded_hrw":
                assert ranked is not None
                snapshots_fresh = all(
                    self._load_is_fresh(candidate, now) for candidate in candidates
                )
                if not snapshots_fresh:
                    decision = "stale_fallback"
                elif len(ranked) > 1:
                    effective_loads = {
                        candidate: self._effective_load(candidate, now)
                        for candidate in candidates
                    }
                    minimum_load = min(effective_loads.values())
                    load_bound = minimum_load + self.load_slack
                    held_worker = (
                        previous.worker_url
                        if previous is not None
                        and previous.worker_url != primary
                        and previous.worker_url in candidates
                        and previous.spill_until > now
                        else None
                    )
                    if (
                        held_worker is not None
                        and effective_loads[held_worker] <= load_bound
                    ):
                        worker = held_worker
                        decision = "held_spill"
                    elif effective_loads[primary] > load_bound:
                        worker = next(
                            candidate
                            for candidate in ranked[1:]
                            if effective_loads[candidate] <= load_bound
                        )
                        decision = "spill"

            if previous is not None and previous.worker_url == worker:
                self._worker_affinity[worker]["hit"] += 1
                ROUTER_AFFINITY.labels(result="hit").inc()
            else:
                self._worker_affinity[worker]["miss"] += 1
                ROUTER_AFFINITY.labels(result="miss").inc()
            ROUTER_DECISIONS.labels(policy=self.routing_policy, decision=decision).inc()
            spill_until = now + self.spill_hold_seconds if worker != primary else 0.0
            self._record_state(
                pack_id,
                RoutingState(worker_url=worker, last_seen=now, spill_until=spill_until),
                now,
            )
            return worker
        finally:
            ROUTER_DECISION_LATENCY.labels(policy=self.routing_policy).observe(
                time.perf_counter() - started
            )

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
        self._prune_state(now)

    def _prune_state(self, now: float) -> None:
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
        snapshot = self._worker_load[worker_url]
        upstream_load = (
            snapshot.running + snapshot.waiting
            if self._load_is_fresh(worker_url, observed_at) and snapshot is not None
            else 0.0
        )
        raw_load = max(float(self._local_in_flight[worker_url]), upstream_load)
        return raw_load / self.capacity_weights[worker_url]

    def _load_is_fresh(self, worker_url: str, now: float) -> bool:
        snapshot = self._worker_load[worker_url]
        return (
            snapshot is not None
            and now - snapshot.observed_at < self.load_stale_seconds
        )

    def mark_unhealthy(self, worker_url: str) -> None:
        if worker_url in self._health:
            updated = dict(self._health)
            updated[worker_url] = False
            self._health = updated

    def acquire(self, worker_url: str) -> WorkerLease:
        if worker_url not in self._health:
            raise ValueError("worker is not managed by this router")
        self._local_in_flight[worker_url] += 1
        self._publish_worker_local_metrics(worker_url, self._clock())
        return WorkerLease(self, worker_url)

    def release(self, worker_url: str) -> None:
        self._release_worker(worker_url)

    def _release_worker(self, worker_url: str) -> None:
        current = self._local_in_flight[worker_url]
        if current <= 0:
            raise RuntimeError(f"worker lease underflow: {self._safe_worker_id(worker_url)}")
        if current == 1:
            self._local_in_flight.pop(worker_url, None)
        else:
            self._local_in_flight[worker_url] = current - 1
        if worker_url in self._health:
            self._publish_worker_local_metrics(worker_url, self._clock())

    def local_in_flight(self, worker_url: str) -> int:
        return self._local_in_flight[worker_url]

    def health(self) -> dict[str, bool]:
        return dict(self._health)

    def stats(self) -> dict[str, WorkerRoutingStats]:
        now = self._clock()
        self._prune_state(now)
        assigned = Counter(state.worker_url for state in self._assignments.values())
        return {
            worker: self._worker_stats(worker, assigned[worker], now)
            for worker in self.worker_urls
        }

    def _worker_stats(
        self,
        worker_url: str,
        assigned_packs: int,
        now: float,
    ) -> WorkerRoutingStats:
        snapshot = self._worker_load[worker_url]
        fresh = self._load_is_fresh(worker_url, now)
        affinity = self._worker_affinity[worker_url]
        return WorkerRoutingStats(
            assigned_packs=assigned_packs,
            affinity_hits=affinity["hit"],
            affinity_misses=affinity["miss"],
            local_in_flight=self._local_in_flight[worker_url],
            upstream_running=snapshot.running if fresh and snapshot is not None else None,
            upstream_waiting=snapshot.waiting if fresh and snapshot is not None else None,
            effective_load=self._effective_load(worker_url, now),
            load_fresh=fresh,
            capacity_weight=self.capacity_weights[worker_url],
        )

    def _publish_worker_snapshot_metrics(self, now: float) -> None:
        for worker in self.worker_urls:
            stats = self._worker_stats(worker, assigned_packs=0, now=now)
            safe_worker_id = self._safe_worker_id(worker)
            for source, value in (
                ("running", stats.upstream_running or 0.0),
                ("waiting", stats.upstream_waiting or 0.0),
                ("effective", stats.effective_load),
            ):
                ROUTER_WORKER_LOAD.labels(
                    worker_id=safe_worker_id,
                    source=source,
                ).set(value)
            ROUTER_WORKER_LOAD_FRESH.labels(worker_id=safe_worker_id).set(
                1 if stats.load_fresh else 0
            )

    def _publish_worker_local_metrics(self, worker_url: str, now: float) -> None:
        stats = self._worker_stats(worker_url, assigned_packs=0, now=now)
        safe_worker_id = self._safe_worker_id(worker_url)
        for source, value in (
            ("local", float(stats.local_in_flight)),
            ("effective", stats.effective_load),
        ):
            ROUTER_WORKER_LOAD.labels(
                worker_id=safe_worker_id,
                source=source,
            ).set(value)

    @staticmethod
    def _safe_worker_id(worker_url: str) -> str:
        return hashlib.sha256(worker_url.encode()).hexdigest()[:12]

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
