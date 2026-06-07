import pytest

from flume.router import PackRouter
from flume.store import FlumeStore


@pytest.mark.asyncio
async def test_router_is_consistent_for_pack(tmp_path) -> None:
    store = FlumeStore(f"sqlite:///{tmp_path / 'flume.db'}")
    store.init_schema()
    router = PackRouter(["http://a", "http://b"], store)

    first = await router.choose("pack-a")
    second = await router.choose("pack-a")

    assert first == second


@pytest.mark.asyncio
async def test_router_falls_back_from_unhealthy_worker(tmp_path) -> None:
    store = FlumeStore(f"sqlite:///{tmp_path / 'flume.db'}")
    store.init_schema()

    async def healthy(worker_url: str) -> bool:
        return worker_url == "http://b"

    router = PackRouter(["http://a", "http://b"], store, health_checker=healthy)
    selected = await router.choose("pack-a")

    assert selected == "http://b"
