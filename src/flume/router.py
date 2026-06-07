from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable

from flume.store import FlumeStore

HealthChecker = Callable[[str], Awaitable[bool]]


class PackRouter:
    def __init__(
        self,
        worker_urls: list[str],
        store: FlumeStore,
        health_checker: HealthChecker | None = None,
    ):
        if not worker_urls:
            raise ValueError("at least one vLLM worker is required")
        self.worker_urls = [url.rstrip("/") for url in worker_urls]
        self.store = store
        self.health_checker = health_checker

    async def choose(self, pack_id: str) -> str:
        route = self.store.get_route(pack_id)
        if route is not None and await self._healthy(route.worker_url):
            self.store.save_route(pack_id, route.worker_url, hit=True)
            return route.worker_url

        worker_url = await self._first_healthy(self._deterministic_worker(pack_id))
        self.store.save_route(pack_id, worker_url, miss=route is not None)
        return worker_url

    async def health(self) -> dict[str, bool | None]:
        return {worker: await self._healthy(worker) for worker in self.worker_urls}

    def _deterministic_worker(self, pack_id: str) -> str:
        digest = hashlib.sha256(pack_id.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], byteorder="big") % len(self.worker_urls)
        return self.worker_urls[index]

    async def _first_healthy(self, preferred: str) -> str:
        candidates = [preferred, *[worker for worker in self.worker_urls if worker != preferred]]
        for worker in candidates:
            if await self._healthy(worker):
                return worker
        return preferred

    async def _healthy(self, worker_url: str) -> bool:
        if self.health_checker is None:
            return True
        try:
            return await self.health_checker(worker_url)
        except Exception:
            return False
