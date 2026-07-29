import asyncio

import pytest
from prometheus_client import REGISTRY

from flume.router import NoHealthyWorkers, PackRouter, WarmupSingleFlight, WorkerLoad


@pytest.mark.asyncio
async def test_router_is_consistent_without_hot_path_health_calls() -> None:
    calls = 0

    async def healthy(_: str) -> bool:
        nonlocal calls
        calls += 1
        return True

    router = PackRouter(["http://a", "http://b"], healthy, refresh_seconds=60)
    await router.start()
    calls_after_refresh = calls

    first = await router.choose("pack-a")
    second = await router.choose("pack-a")

    assert first == second
    assert calls == calls_after_refresh
    await router.close()


@pytest.mark.asyncio
async def test_router_start_cancellation_closes_started_health_monitor_once() -> None:
    load_started = asyncio.Event()

    async def healthy(_: str) -> bool:
        return True

    async def blocked_load(_: str) -> WorkerLoad:
        load_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    router = PackRouter(
        ["http://a"],
        health_checker=healthy,
        load_checker=blocked_load,
        refresh_seconds=60,
    )
    start_task = asyncio.create_task(router.start())
    await load_started.wait()
    health_monitor = router._health_monitor_task
    assert health_monitor is not None

    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert health_monitor.cancelled()
    assert health_monitor.cancelling() == 1
    assert router._health_monitor_task is None
    assert router._load_monitor_task is None
    await router.close()
    assert health_monitor.cancelling() == 1


@pytest.mark.asyncio
async def test_router_fails_over_from_cached_unhealthy_worker() -> None:
    async def healthy(worker_url: str) -> bool:
        return worker_url == "http://b"

    router = PackRouter(["http://a", "http://b"], healthy, refresh_seconds=60)
    await router.start()

    assert await router.choose("pack-a") == "http://b"
    await router.close()


@pytest.mark.asyncio
async def test_router_fails_fast_when_pool_is_unavailable() -> None:
    async def unhealthy(_: str) -> bool:
        return False

    router = PackRouter(["http://a", "http://b"], unhealthy, refresh_seconds=60)
    await router.start()

    with pytest.raises(NoHealthyWorkers):
        await router.choose("pack-a")
    await router.close()


@pytest.mark.asyncio
async def test_router_observes_decision_latency_once_for_success_and_unavailable() -> None:
    labels = {"policy": "hrw"}
    before = (
        REGISTRY.get_sample_value(
            "flume_router_decision_duration_seconds_count",
            labels,
        )
        or 0
    )
    await PackRouter(["http://a"]).choose("pack")
    unavailable = PackRouter(["http://a"], health_checker=lambda _: asyncio.sleep(0, False))
    with pytest.raises(NoHealthyWorkers):
        await unavailable.choose("pack")
    after = REGISTRY.get_sample_value(
        "flume_router_decision_duration_seconds_count",
        labels,
    )

    assert after == before + 2


@pytest.mark.asyncio
async def test_rendezvous_hashing_has_bounded_remapping() -> None:
    before = PackRouter(["http://a", "http://b", "http://c"])
    after = PackRouter(["http://a", "http://b", "http://c", "http://d"])
    pack_ids = [f"pack-{index}" for index in range(10_000)]

    changed = 0
    for pack_id in pack_ids:
        changed += await before.choose(pack_id) != await after.choose(pack_id)

    assert changed / len(pack_ids) <= (1 / 4) + 0.02


@pytest.mark.asyncio
async def test_hrw_policy_ignores_load_and_preserves_primary() -> None:
    router = PackRouter(["http://a", "http://b"])
    primary = await router.choose("pack")
    for _ in range(10):
        router.acquire(primary)

    assert await router.choose("pack") == primary


async def zero_load(_: str) -> WorkerLoad:
    return WorkerLoad(running=0, waiting=0)


@pytest.mark.asyncio
async def test_bounded_hrw_spills_to_one_secondary_and_holds() -> None:
    now = 0.0
    workers = ["http://a", "http://b", "http://c"]
    primary_router = PackRouter(workers)
    primary = await primary_router.choose("pack")
    ranked = sorted(
        workers,
        key=lambda worker: PackRouter._score("pack", worker),
        reverse=True,
    )
    secondary = ranked[1]
    router = PackRouter(
        workers,
        load_checker=zero_load,
        routing_policy="bounded_hrw",
        load_slack=2,
        spill_hold_seconds=2.0,
        clock=lambda: now,
    )
    await router.refresh_load()
    for _ in range(3):
        router.acquire(primary)

    assert await router.choose("pack") == secondary
    for _ in range(3):
        router.release(primary)
    assert await router.choose("pack") == secondary

    now = 2.1
    assert await router.choose("pack") == primary


@pytest.mark.asyncio
async def test_held_spill_reselects_when_held_worker_exceeds_load_bound() -> None:
    now = 0.0
    workers = ["http://a", "http://b", "http://c"]
    ranked = sorted(
        workers,
        key=lambda worker: PackRouter._score("pack", worker),
        reverse=True,
    )
    loads = {
        ranked[0]: WorkerLoad(running=3, waiting=0),
        ranked[1]: WorkerLoad(running=0, waiting=0),
        ranked[2]: WorkerLoad(running=0, waiting=0),
    }

    async def load(worker_url: str) -> WorkerLoad:
        return loads[worker_url]

    router = PackRouter(
        workers,
        load_checker=load,
        routing_policy="bounded_hrw",
        load_slack=2,
        spill_hold_seconds=2,
        clock=lambda: now,
    )
    await router.refresh_load()
    assert await router.choose("pack") == ranked[1]

    now = 0.5
    loads[ranked[0]] = WorkerLoad(running=5, waiting=0)
    loads[ranked[1]] = WorkerLoad(running=5, waiting=0)
    await router.refresh_load()

    assert await router.choose("pack") == ranked[2]


@pytest.mark.asyncio
async def test_bounded_hrw_skips_overloaded_second_rank_for_global_load_bound() -> None:
    workers = ["http://a", "http://b", "http://c"]
    ranked = sorted(
        workers,
        key=lambda worker: PackRouter._score("pack", worker),
        reverse=True,
    )
    router = PackRouter(
        workers,
        load_checker=zero_load,
        routing_policy="bounded_hrw",
        load_slack=2,
    )
    await router.refresh_load()
    for _ in range(3):
        router.acquire(ranked[0])
        router.acquire(ranked[1])

    assert await router.choose("pack") == ranked[2]


@pytest.mark.asyncio
async def test_bounded_hrw_uses_capacity_normalized_load() -> None:
    workers = ["http://a", "http://b"]
    primary = await PackRouter(workers).choose("pack")
    secondary = next(worker for worker in workers if worker != primary)
    router = PackRouter(
        workers,
        load_checker=zero_load,
        routing_policy="bounded_hrw",
        load_slack=0,
        capacity_weights={primary: 4.0, secondary: 1.0},
    )
    await router.refresh_load()
    router.acquire(primary)
    router.acquire(primary)

    assert await router.choose("pack") == secondary

    router.acquire(secondary)
    assert await router.choose("another-pack") in workers


@pytest.mark.asyncio
async def test_bounded_hrw_uses_fresh_upstream_load_then_falls_back_when_stale() -> None:
    now = 0.0
    workers = ["http://a", "http://b"]
    primary = await PackRouter(workers).choose("pack")
    loads = {
        worker: WorkerLoad(running=3 if worker == primary else 0, waiting=0)
        for worker in workers
    }

    async def load(worker_url: str) -> WorkerLoad:
        return loads[worker_url]

    router = PackRouter(
        workers,
        load_checker=load,
        routing_policy="bounded_hrw",
        load_slack=2,
        spill_hold_seconds=0,
        load_refresh_seconds=0.5,
        load_stale_seconds=2.0,
        clock=lambda: now,
    )
    await router.refresh_load()
    assert await router.choose("pack") != primary

    now = 2.0
    assert await router.choose("pack") == primary


@pytest.mark.asyncio
async def test_bounded_hrw_uses_pure_hrw_when_any_worker_load_is_missing() -> None:
    workers = ["http://a", "http://b"]
    primary = await PackRouter(workers).choose("pack")

    async def partial_load(worker_url: str) -> WorkerLoad | None:
        if worker_url == primary:
            return WorkerLoad(running=0, waiting=0)
        return None

    router = PackRouter(
        workers,
        load_checker=partial_load,
        routing_policy="bounded_hrw",
    )
    await router.refresh_load()
    for _ in range(10):
        router.acquire(primary)

    assert await router.choose("pack") == primary


@pytest.mark.asyncio
async def test_effective_load_uses_max_of_local_and_upstream() -> None:
    async def upstream(_: str) -> WorkerLoad:
        return WorkerLoad(running=2, waiting=1)

    router = PackRouter(
        ["http://a"],
        load_checker=upstream,
        capacity_weights={"http://a": 2},
    )
    await router.refresh_load()
    router.acquire("http://a")
    router.acquire("http://a")
    assert router.stats()["http://a"].effective_load == 1.5

    router.acquire("http://a")
    router.acquire("http://a")
    assert router.stats()["http://a"].effective_load == 2


@pytest.mark.asyncio
async def test_failed_load_refresh_keeps_snapshot_only_until_stale() -> None:
    now = 0.0
    workers = ["http://a", "http://b"]
    primary = await PackRouter(workers).choose("pack")
    available = True

    async def load(worker_url: str) -> WorkerLoad | None:
        if not available:
            return None
        return WorkerLoad(running=4 if worker_url == primary else 0, waiting=0)

    router = PackRouter(
        workers,
        load_checker=load,
        routing_policy="bounded_hrw",
        spill_hold_seconds=0,
        load_refresh_seconds=0.5,
        load_stale_seconds=2.0,
        clock=lambda: now,
    )
    await router.refresh_load()
    available = False
    now = 1.0
    await router.refresh_load()
    assert await router.choose("pack") != primary

    now = 2.0
    assert await router.choose("pack") == primary


@pytest.mark.asyncio
async def test_load_results_are_timestamped_when_each_worker_returns() -> None:
    now = 0.0

    async def load(worker_url: str) -> WorkerLoad:
        nonlocal now
        if worker_url == "http://slow":
            await asyncio.sleep(0)
            now = 3.0
        return WorkerLoad(running=1, waiting=0)

    router = PackRouter(
        ["http://fast", "http://slow"],
        load_checker=load,
        load_refresh_seconds=0.5,
        load_stale_seconds=2,
        clock=lambda: now,
    )
    snapshots = await router.refresh_load()

    assert snapshots["http://fast"] is not None
    assert snapshots["http://fast"].observed_at == 0
    assert snapshots["http://slow"] is not None
    assert snapshots["http://slow"].observed_at == 3
    assert not router.stats()["http://fast"].load_fresh
    assert router.stats()["http://slow"].load_fresh


@pytest.mark.asyncio
async def test_routing_state_is_size_and_ttl_bounded() -> None:
    now = 0.0
    router = PackRouter(
        ["http://a"],
        state_max_entries=2,
        state_ttl_seconds=10,
        clock=lambda: now,
    )
    await router.choose("pack-a")
    await router.choose("pack-b")
    await router.choose("pack-c")

    assert list(router._assignments) == ["pack-b", "pack-c"]

    now = 10.0
    await router.choose("pack-d")
    assert list(router._assignments) == ["pack-d"]


def test_router_tracks_local_in_flight_without_underflow() -> None:
    router = PackRouter(["http://a"])

    first = router.acquire("http://a")
    second = router.acquire("http://a")
    assert first.release()
    assert not first.release()
    assert router.local_in_flight("http://a") == 1

    assert second.release()
    assert router.local_in_flight("http://a") == 0

    with pytest.raises(RuntimeError, match="underflow"):
        router.release("http://a")
    with pytest.raises(ValueError, match="not managed"):
        router.acquire("http://unknown")


@pytest.mark.asyncio
async def test_worker_stats_are_per_worker_and_prune_expired_assignments() -> None:
    now = 0.0
    router = PackRouter(
        ["http://a", "http://b"],
        state_ttl_seconds=10,
        clock=lambda: now,
    )
    selected = await router.choose("pack")
    await router.choose("pack")
    other = next(worker for worker in router.worker_urls if worker != selected)

    stats = router.stats()
    assert stats[selected].assigned_packs == 1
    assert stats[selected].affinity_hits == 1
    assert stats[selected].affinity_misses == 1
    assert stats[other].assigned_packs == 0
    assert stats[other].affinity_hits == 0
    assert stats[other].affinity_misses == 0

    now = 10.0
    assert router.stats()[selected].assigned_packs == 0


@pytest.mark.asyncio
async def test_worker_stats_report_fresh_and_stale_load_without_urls() -> None:
    now = 0.0

    async def load(worker_url: str) -> WorkerLoad:
        assert worker_url == "http://a"
        return WorkerLoad(running=2, waiting=1)

    router = PackRouter(
        ["http://a"],
        load_checker=load,
        load_refresh_seconds=0.5,
        load_stale_seconds=2,
        capacity_weights={"http://a": 2},
        clock=lambda: now,
    )
    await router.refresh_load()

    fresh = router.stats()["http://a"]
    assert fresh.upstream_running == 2
    assert fresh.upstream_waiting == 1
    assert fresh.effective_load == 1.5
    assert fresh.load_fresh

    now = 2.0
    stale = router.stats()["http://a"]
    assert stale.upstream_running is None
    assert stale.upstream_waiting is None
    assert stale.effective_load == 0
    assert not stale.load_fresh


@pytest.mark.asyncio
async def test_warmup_singleflight_coalesces_operations() -> None:
    singleflight = WarmupSingleFlight()
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return "done"

    results = await asyncio.gather(
        *(singleflight.run(("tenant", "pack", "worker"), operation) for _ in range(20))
    )

    assert results == ["done"] * 20
    assert calls == 1
