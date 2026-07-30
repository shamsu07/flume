import asyncio

import pytest

from flume.router import NoHealthyWorkers, PackRouter, WarmupSingleFlight


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
