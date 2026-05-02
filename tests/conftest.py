"""Shared pytest fixtures."""

from __future__ import annotations

import asyncio
import os

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure tests don't accidentally hit a real API.
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("MEMORY_AUTH_TOKEN", "")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client():
    """
    HTTPX client wired to the ASGI app. We do NOT bring up the lifespan
    here — Epic 0 contract tests use a stub pool. Once Epic 1 lands,
    this fixture will switch to a real Postgres test container.
    """
    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
