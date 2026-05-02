from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import Settings
from app.obs.logging import get_logger

log = get_logger(__name__)

_pool: asyncpg.Pool | None = None
_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register pgvector codec on every new connection in the pool."""
    await register_vector(conn)


async def init_pool(settings: Settings, *, min_size: int = 2, max_size: int = 10) -> asyncpg.Pool:
    """
    Create the global asyncpg pool. Retries on startup so the app can come
    up before Postgres has finished booting (docker-compose handles ordering
    via healthcheck, but extra safety doesn't hurt).
    """
    global _pool

    last_err: Exception | None = None
    for attempt in range(20):
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.postgres_dsn,
                min_size=min_size,
                max_size=max_size,
                init=_init_connection,
                command_timeout=30,
            )
            await ensure_schema(_pool)
            log.info("db.pool.ready", attempt=attempt + 1)
            return _pool
        except (OSError, asyncpg.PostgresError) as exc:
            last_err = exc
            log.warning("db.pool.retry", attempt=attempt + 1, error=str(exc))
            await asyncio.sleep(min(2.0, 0.25 * (attempt + 1)))

    raise RuntimeError(f"could not connect to Postgres: {last_err!r}")


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Apply idempotent DDL so existing named volumes get new tables too.

    Migrations are plain SQL files in `migrations/`, applied in lexical
    order. Each file must be idempotent (IF NOT EXISTS, DO blocks, etc.)
    so reruns on an already-migrated volume are no-ops.
    """
    files = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"no migrations found in {_MIGRATIONS_DIR}")
    async with pool.acquire() as conn:
        for path in files:
            log.info("db.migration.apply", file=path.name)
            await conn.execute(path.read_text(encoding="utf-8"))


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("db pool not initialised — call init_pool() first")
    return _pool
