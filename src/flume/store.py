from __future__ import annotations

import base64
import json
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    and_,
    event,
    func,
    inspect,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from flume.models import ContextPack


class Base(DeclarativeBase):
    pass


class PackRecord(Base):
    __tablename__ = "context_packs"

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    pack_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(512), index=True)
    tokenizer_id: Mapped[str] = mapped_column(String(512))
    template_id: Mapped[str] = mapped_column(String(128), index=True)
    document_hash: Mapped[str] = mapped_column(String(64), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), index=True)
    token_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[str] = mapped_column(Text)


class PackAnnotationRecord(Base):
    __tablename__ = "pack_annotations"

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    pack_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    data: Mapped[str] = mapped_column(Text)


class PackConflictError(RuntimeError):
    """Raised when a pack id is reused for different immutable content."""


@dataclass(frozen=True, slots=True)
class StoredPackPage:
    items: list[ContextPack]
    next_cursor: str | None


class ByteBoundedPackCache:
    """Small synchronous LRU used from the event loop's request path."""

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple[str, str], tuple[ContextPack, int]] = OrderedDict()
        self._bytes = 0

    @property
    def size_bytes(self) -> int:
        return self._bytes

    @property
    def entries(self) -> int:
        return len(self._entries)

    def get(self, tenant_id: str, pack_id: str) -> ContextPack | None:
        key = (tenant_id, pack_id)
        item = self._entries.get(key)
        if item is None:
            return None
        pack, _ = item
        if pack.expired:
            self.delete(tenant_id, pack_id)
            return None
        self._entries.move_to_end(key)
        return pack

    def put(self, pack: ContextPack) -> None:
        key = (pack.tenant_id, pack.pack_id)
        serialized_size = len(pack.model_dump_json().encode("utf-8"))
        self.delete(*key)
        if serialized_size > self.max_bytes:
            return
        self._entries[key] = (pack, serialized_size)
        self._bytes += serialized_size
        while self._bytes > self.max_bytes:
            _, (_, removed_size) = self._entries.popitem(last=False)
            self._bytes -= removed_size

    def delete(self, tenant_id: str, pack_id: str) -> None:
        item = self._entries.pop((tenant_id, pack_id), None)
        if item is not None:
            self._bytes -= item[1]


def _as_async_url(database_url: str) -> str:
    if database_url.startswith("sqlite+aiosqlite:"):
        return database_url
    if database_url.startswith("sqlite:"):
        return database_url.replace("sqlite:", "sqlite+aiosqlite:", 1)
    raise ValueError("Flume v1 only supports SQLite database URLs")


class FlumeStore:
    def __init__(self, database_url: str, *, busy_timeout_ms: int = 5_000):
        self.database_url = _as_async_url(database_url)
        self.busy_timeout_ms = busy_timeout_ms
        self.engine: AsyncEngine = create_async_engine(self.database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self._install_sqlite_pragmas()

    def _install_sqlite_pragmas(self) -> None:
        busy_timeout_ms = self.busy_timeout_ms

        @event.listens_for(self.engine.sync_engine, "connect")
        def configure_sqlite(dbapi_connection: Any, _: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

    async def init_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(self._migrate_legacy_schema)
            await connection.run_sync(Base.metadata.create_all)

    @staticmethod
    def _migrate_legacy_schema(connection: Any) -> None:
        inspector = inspect(connection)
        if "context_packs" not in inspector.get_table_names():
            return
        columns = {column["name"] for column in inspector.get_columns("context_packs")}
        primary_key = inspector.get_pk_constraint("context_packs").get("constrained_columns") or []
        if "expires_at" in columns and set(primary_key) == {"tenant_id", "pack_id"}:
            return

        tables = set(inspector.get_table_names())
        legacy_name = "legacy_context_packs"
        suffix = 1
        while legacy_name in tables:
            suffix += 1
            legacy_name = f"legacy_context_packs_{suffix}"
        connection.execute(text(f'ALTER TABLE "context_packs" RENAME TO "{legacy_name}"'))
        for index in inspect(connection).get_indexes(legacy_name):
            connection.execute(text(f'DROP INDEX IF EXISTS "{index["name"]}"'))

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def ping(self) -> bool:
        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def save_pack(self, pack: ContextPack) -> ContextPack:
        expires_at = None
        if pack.ttl_seconds is not None:
            expires_at = pack.created_at + timedelta(seconds=pack.ttl_seconds)
        values = {
            "tenant_id": pack.tenant_id,
            "pack_id": pack.pack_id,
            "model_id": pack.model_id,
            "tokenizer_id": pack.tokenizer_id,
            "template_id": pack.template_id,
            "document_hash": pack.document_hash,
            "token_hash": pack.token_hash,
            "token_count": pack.token_count,
            "created_at": pack.created_at,
            "expires_at": expires_at,
            "data": pack.model_dump_json(),
        }
        async with self.session() as session:
            statement = (
                sqlite_insert(PackRecord)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["tenant_id", "pack_id"])
            )
            result = await session.execute(statement)
            if getattr(result, "rowcount", None) == 1:
                return pack
            existing = await session.get(PackRecord, (pack.tenant_id, pack.pack_id))
            if existing is None:
                raise RuntimeError("pack registration lost during concurrent insert")
            if self._immutable_identity(existing) != self._pack_identity(pack):
                raise PackConflictError("pack id already exists with different immutable content")
            return ContextPack.model_validate_json(existing.data)

    async def get_pack(self, tenant_id: str, pack_id: str) -> ContextPack | None:
        now = datetime.now(UTC)
        async with self.session() as session:
            statement = select(PackRecord).where(
                PackRecord.tenant_id == tenant_id,
                PackRecord.pack_id == pack_id,
                or_(PackRecord.expires_at.is_(None), PackRecord.expires_at > now),
            )
            record = await session.scalar(statement)
            if record is None:
                return None
            return ContextPack.model_validate_json(record.data)

    async def list_packs(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
        cursor: str | None = None,
    ) -> StoredPackPage:
        limit = min(max(limit, 1), 200)
        now = datetime.now(UTC)
        statement = (
            select(PackRecord)
            .where(
                PackRecord.tenant_id == tenant_id,
                or_(PackRecord.expires_at.is_(None), PackRecord.expires_at > now),
            )
            .order_by(PackRecord.created_at.desc(), PackRecord.pack_id.desc())
            .limit(limit + 1)
        )
        if cursor:
            created_at, pack_id = self._decode_cursor(cursor)
            statement = statement.where(
                or_(
                    PackRecord.created_at < created_at,
                    and_(PackRecord.created_at == created_at, PackRecord.pack_id < pack_id),
                )
            )
        async with self.session() as session:
            records = list((await session.scalars(statement)).all())
        has_more = len(records) > limit
        records = records[:limit]
        next_cursor = None
        if has_more and records:
            next_cursor = self._encode_cursor(records[-1].created_at, records[-1].pack_id)
        return StoredPackPage(
            items=[ContextPack.model_validate_json(record.data) for record in records],
            next_cursor=next_cursor,
        )

    async def update_annotations(
        self,
        tenant_id: str,
        pack_id: str,
        annotations: dict[str, Any],
    ) -> dict[str, Any] | None:
        if await self.get_pack(tenant_id, pack_id) is None:
            return None
        values = {
            "tenant_id": tenant_id,
            "pack_id": pack_id,
            "updated_at": datetime.now(UTC),
            "data": json.dumps(annotations, sort_keys=True, separators=(",", ":")),
        }
        async with self.session() as session:
            statement = sqlite_insert(PackAnnotationRecord).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=["tenant_id", "pack_id"],
                set_={"updated_at": values["updated_at"], "data": values["data"]},
            )
            await session.execute(statement)
        return annotations

    async def counts(self, tenant_id: str) -> tuple[int, int]:
        now = datetime.now(UTC)
        async with self.session() as session:
            packs = await session.scalar(
                select(func.count())
                .select_from(PackRecord)
                .where(
                    PackRecord.tenant_id == tenant_id,
                    or_(PackRecord.expires_at.is_(None), PackRecord.expires_at > now),
                )
            )
        return int(packs or 0), 0

    @staticmethod
    def _immutable_identity(record: PackRecord) -> tuple[Any, ...]:
        return (
            record.model_id,
            record.tokenizer_id,
            record.template_id,
            record.document_hash,
            record.token_hash,
            record.token_count,
        )

    @staticmethod
    def _pack_identity(pack: ContextPack) -> tuple[Any, ...]:
        return (
            pack.model_id,
            pack.tokenizer_id,
            pack.template_id,
            pack.document_hash,
            pack.token_hash,
            pack.token_count,
        )

    @staticmethod
    def _encode_cursor(created_at: datetime, pack_id: str) -> str:
        raw = json.dumps([created_at.isoformat(), pack_id], separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str) -> tuple[datetime, str]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            created_at_text, pack_id = json.loads(
                base64.urlsafe_b64decode(padded.encode()).decode()
            )
            created_at = datetime.fromisoformat(created_at_text)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid pagination cursor") from exc
        return created_at, str(pack_id)
