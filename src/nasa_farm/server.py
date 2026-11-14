"""nasa-farm HTTP API — thin FastAPI adapter over `service.plan`.

- `GET /healthz` liveness, `GET /v1/meta` vocab, `POST /v1/plan` compute
- Compute is serialized through a semaphore (each plan already saturates
  every core it gets) and identical requests are served from a small TTL
  cache — deterministic outputs make caching exact, not approximate.
- `tunnel()` wires `cloudflared` quick tunnels for zero-cost public URLs.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware

from nasa_farm import CropLibrary, __version__
from nasa_farm.heads import METRICS, OBJECTIVE_DIRECTION
from nasa_farm.service import PlanRequest, PlanResponse, plan

TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _env(key: str, default: str) -> str:
    return os.environ.get(f"NASA_FARM_{key}", default)


class _TimedCache:
    """Bounded LRU + TTL; key = canonical request JSON (output is a pure
    function of the request, so a hit is exact)."""

    def __init__(self, size: int, ttl: float) -> None:
        self.size, self.ttl = size, ttl
        self._d: OrderedDict[str, tuple[float, PlanResponse]] = OrderedDict()

    def get(self, key: str) -> PlanResponse | None:
        hit = self._d.get(key)
        if hit is None:
            return None
        ts, result = hit
        if time.monotonic() - ts > self.ttl:
            self._d.pop(key, None)
            return None
        self._d.move_to_end(key)
        return result

    def put(self, key: str, result: PlanResponse) -> None:
        self._d[key] = (time.monotonic(), result)
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)


def create_app() -> FastAPI:
    app = FastAPI(title="nasa-farm", version=__version__)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    jobs = asyncio.Semaphore(int(_env("JOBS", "1")))
    cache = _TimedCache(int(_env("CACHE", "128")),
                      float(_env("CACHE_TTL", "86400")))

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/meta")
    async def metadata() -> dict:
        return {"version": __version__,
                "crops": list(CropLibrary.default().names),
                "metrics": list(METRICS),
                "objectives": OBJECTIVE_DIRECTION}

    @app.post("/v1/plan")
    async def v1_plan(body: PlanRequest, request: Request) -> PlanResponse:
        key = body.model_dump_json()
        hit = cache.get(key)
        if hit is not None:
            request.state.cached = True
            return hit
        try:
            async with jobs:
                result = await asyncio.to_thread(plan, body)
        except ValueError as exception:
            raise HTTPException(status_code=400, detail=str(exception))
        cache.put(key, result)
        return result

    @app.middleware("http")
    async def cache_header(request: Request, call_next):
        result = await call_next(request)
        if getattr(request.state, "cached", False):
            result.headers["x-cache"] = "hit"
        return result

    return app


app = create_app()


def tunnel(port: int, binary: str = "cloudflared") -> subprocess.Popen:
    """Spawn `cloudflared tunnel --url http://localhost:port`; stream its log
    to stderr and announce the public URL once assigned."""
    try:
        process = subprocess.Popen(
            [binary, "tunnel", "--url", f"http://localhost:{port}",
             "--no-autoupdate"],
            stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        raise SystemExit(
            "cloudflared not installed — "
            "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")

    def watch() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            sys.stderr.write(line)
            if m := TUNNEL_URL.search(line):
                sys.stderr.write(f"\npublic api: {m.group(0)}\n\n")

    threading.Thread(target=watch, daemon=True).start()
    return process
