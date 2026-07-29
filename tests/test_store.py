from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from flume.compiler import ContextPackCompiler
from flume.models import DocumentChunk, PackCreateRequest
from flume.store import ByteBoundedPackCache, FlumeStore


def make_pack(*, tenant_id: str = "demo", doc_id: str = "doc"):
    return ContextPackCompiler().compile(
        PackCreateRequest(
            tenant_id=tenant_id,
            model_id="model",
            tokenizer_id="tokenizer",
            chunks=[DocumentChunk(doc_id=doc_id, text="hello")],
        )
    )


@pytest.mark.asyncio
async def test_store_round_trips_pack_idempotently(tmp_path) -> None:
    store = FlumeStore(f"sqlite:///{tmp_path / 'flume.db'}")
    await store.init_schema()
    pack = make_pack()

    first = await store.save_pack(pack)
    second = await store.save_pack(pack)
    loaded = await store.get_pack("demo", pack.pack_id)

    assert first.pack_id == second.pack_id
    assert loaded is not None
    assert loaded.compiled_prefix == pack.compiled_prefix
    assert len((await store.list_packs("demo")).items) == 1
    await store.close()


@pytest.mark.asyncio
async def test_store_scopes_packs_and_paginates(tmp_path) -> None:
    store = FlumeStore(f"sqlite:///{tmp_path / 'flume.db'}")
    await store.init_schema()
    for index in range(3):
        pack = make_pack(doc_id=f"doc-{index}")
        pack.created_at = datetime.now(UTC) + timedelta(seconds=index)
        await store.save_pack(pack)
    other = make_pack(tenant_id="other")
    await store.save_pack(other)

    first = await store.list_packs("demo", limit=2)
    second = await store.list_packs("demo", limit=2, cursor=first.next_cursor)

    assert len(first.items) == 2
    assert first.next_cursor is not None
    assert len(second.items) == 1
    assert await store.get_pack("demo", other.pack_id) is None
    await store.close()


@pytest.mark.asyncio
async def test_store_enables_sqlite_safety_pragmas(tmp_path) -> None:
    store = FlumeStore(f"sqlite:///{tmp_path / 'flume.db'}", busy_timeout_ms=1234)
    await store.init_schema()

    async with store.engine.connect() as connection:
        journal_mode = (await connection.execute(text("PRAGMA journal_mode"))).scalar_one()
        busy_timeout = (await connection.execute(text("PRAGMA busy_timeout"))).scalar_one()
        foreign_keys = (await connection.execute(text("PRAGMA foreign_keys"))).scalar_one()

    assert journal_mode == "wal"
    assert busy_timeout == 1234
    assert foreign_keys == 1
    await store.close()


def test_pack_cache_is_byte_bounded_and_tenant_scoped() -> None:
    first = make_pack(doc_id="one")
    second = make_pack(doc_id="two")
    entry_size = len(first.model_dump_json().encode())
    cache = ByteBoundedPackCache(max_bytes=entry_size + 16)

    cache.put(first)
    cache.put(second)

    assert cache.entries == 1
    assert cache.get("demo", first.pack_id) is None
    assert cache.get("other", second.pack_id) is None
    assert cache.get("demo", second.pack_id) is second
