from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from flume.models import BenchmarkRun, ContextPack


class Base(DeclarativeBase):
    pass


class PackRecord(Base):
    __tablename__ = "context_packs"

    pack_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    model_id: Mapped[str] = mapped_column(String(512), index=True)
    tokenizer_id: Mapped[str] = mapped_column(String(512))
    template_id: Mapped[str] = mapped_column(String(128), index=True)
    document_hash: Mapped[str] = mapped_column(String(64), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), index=True)
    token_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[str] = mapped_column(Text)


class RouteRecord(Base):
    __tablename__ = "routes"

    pack_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    worker_url: Mapped[str] = mapped_column(String(1024), index=True)
    last_used_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    affinity_hits: Mapped[int] = mapped_column(Integer, default=0)
    affinity_misses: Mapped[int] = mapped_column(Integer, default=0)


class BenchmarkRecord(Base):
    __tablename__ = "benchmark_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[str] = mapped_column(Text)


class FlumeStore:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def init_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_pack(self, pack: ContextPack) -> ContextPack:
        with self.session() as session:
            record = PackRecord(
                pack_id=pack.pack_id,
                tenant_id=pack.tenant_id,
                model_id=pack.model_id,
                tokenizer_id=pack.tokenizer_id,
                template_id=pack.template_id,
                document_hash=pack.document_hash,
                token_hash=pack.token_hash,
                token_count=pack.token_count,
                created_at=pack.created_at,
                data=pack.model_dump_json(),
            )
            session.merge(record)
        return pack

    def get_pack(self, pack_id: str) -> ContextPack | None:
        with self.session() as session:
            record = session.get(PackRecord, pack_id)
            if record is None:
                return None
            pack = ContextPack.model_validate_json(record.data)
            if pack.expired:
                return None
            return pack

    def list_packs(self, tenant_id: str | None = None) -> list[ContextPack]:
        with self.session() as session:
            statement = select(PackRecord).order_by(PackRecord.created_at.desc())
            if tenant_id:
                statement = statement.where(PackRecord.tenant_id == tenant_id)
            return [
                ContextPack.model_validate_json(row.data)
                for row in session.scalars(statement)
            ]

    def save_route(
        self,
        pack_id: str,
        worker_url: str,
        *,
        hit: bool = False,
        miss: bool = False,
    ) -> None:
        with self.session() as session:
            record = session.get(RouteRecord, pack_id)
            now = datetime.now(UTC)
            if record is None:
                record = RouteRecord(
                    pack_id=pack_id,
                    worker_url=worker_url,
                    last_used_at=now,
                    affinity_hits=1 if hit else 0,
                    affinity_misses=1 if miss else 0,
                )
            else:
                record.worker_url = worker_url
                record.last_used_at = now
                if hit:
                    record.affinity_hits += 1
                if miss:
                    record.affinity_misses += 1
            session.merge(record)

    def get_route(self, pack_id: str) -> RouteRecord | None:
        with self.session() as session:
            record = session.get(RouteRecord, pack_id)
            if record is None:
                return None
            return RouteRecord(
                pack_id=record.pack_id,
                worker_url=record.worker_url,
                last_used_at=record.last_used_at,
                affinity_hits=record.affinity_hits,
                affinity_misses=record.affinity_misses,
            )

    def route_counts_by_worker(self) -> dict[str, dict[str, int]]:
        with self.session() as session:
            counts: dict[str, dict[str, int]] = {}
            for record in session.scalars(select(RouteRecord)):
                item = counts.setdefault(
                    record.worker_url,
                    {"assigned_packs": 0, "affinity_hits": 0, "affinity_misses": 0},
                )
                item["assigned_packs"] += 1
                item["affinity_hits"] += record.affinity_hits
                item["affinity_misses"] += record.affinity_misses
            return counts

    def counts(self) -> tuple[int, int]:
        with self.session() as session:
            packs = len(list(session.scalars(select(PackRecord.pack_id))))
            routes = len(list(session.scalars(select(RouteRecord.pack_id))))
            return packs, routes

    def save_benchmark(self, run: BenchmarkRun) -> BenchmarkRun:
        with self.session() as session:
            session.merge(
                BenchmarkRecord(
                    run_id=run.run_id,
                    status=run.status,
                    created_at=run.created_at,
                    data=run.model_dump_json(),
                )
            )
        return run

    def get_benchmark(self, run_id: str) -> BenchmarkRun | None:
        with self.session() as session:
            record = session.get(BenchmarkRecord, run_id)
            if record is None:
                return None
            return BenchmarkRun.model_validate_json(record.data)

    def list_benchmarks(self) -> list[BenchmarkRun]:
        with self.session() as session:
            statement = select(BenchmarkRecord).order_by(BenchmarkRecord.created_at.desc())
            return [
                BenchmarkRun.model_validate_json(row.data)
                for row in session.scalars(statement)
            ]
