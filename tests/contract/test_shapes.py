"""Contract-shape tests for the simple RAG endpoints.

These run against a live `docker compose up` stack and call the OpenAI
embeddings API, so we skip them by default unless RAG_E2E=1 is set.
Run them with `make test-e2e`.
"""

from __future__ import annotations

import os

import httpx
import pytest

E2E = os.environ.get("RAG_E2E") == "1"
BASE = os.environ.get("RAG_BASE_URL", "http://localhost:8080")

pytestmark = pytest.mark.skipif(not E2E, reason="set RAG_E2E=1 to run against live stack")


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=30)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ingest_then_query_returns_most_similar_first():
    docs = [
        {"id": "d1", "text": "The capital of France is Paris."},
        {"id": "d2", "text": "Bananas are a yellow fruit."},
        {"id": "d3", "text": "Mount Everest is the tallest mountain on Earth."},
    ]
    r = httpx.post(f"{BASE}/ingest", json={"documents": docs}, timeout=60)
    assert r.status_code == 201
    assert r.json() == {"ingested": 3}

    r = httpx.post(
        f"{BASE}/query",
        json={"query": "Where is Paris located?", "top_k": 3},
        timeout=30,
    )
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits, "expected at least one hit"
    assert hits[0]["id"] == "d1", f"expected d1 first, got {[h['id'] for h in hits]}"
    # Cosine similarity on normalized vectors lives in roughly [0, 1]:
    assert 0.0 <= hits[0]["score"] <= 1.0


def test_query_validation():
    r = httpx.post(f"{BASE}/query", json={"top_k": 5}, timeout=10)
    assert r.status_code == 422  # missing query
