import asyncio

import pytest

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
        routing_policy="bounded_hrw",
        load_slack=2,
        spill_hold_seconds=2.0,
        clock=lambda: now,
    )
    for _ in range(3):
        router.acquire(primary)

    assert await router.choose("pack") == secondary
    for _ in range(3):
        router.release(primary)
    assert await router.choose("pack") == secondary

    now = 2.1
    assert await router.choose("pack") == primary


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
        routing_policy="bounded_hrw",
        load_slack=2,
    )
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
        routing_policy="bounded_hrw",
        load_slack=0,
        capacity_weights={primary: 4.0, secondary: 1.0},
    )
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

    router.acquire("http://a")
    router.acquire("http://a")
    router.release("http://a")
    assert router.local_in_flight("http://a") == 1

    router.release("http://a")
    router.release("http://a")
    assert router.local_in_flight("http://a") == 0

    with pytest.raises(ValueError, match="not managed"):
        router.acquire("http://unknown")


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
