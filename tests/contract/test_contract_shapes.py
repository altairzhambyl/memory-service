"""API contract-shape tests — MS-15 DoD.

Asserts every endpoint exists, returns the right status, and the
JSON body has the spec shape. Recall quality and structured memory
content are NOT exercised here — that's MS-16 / MS-17.

Skipped unless RAG_E2E=1 (run against a live `docker compose up`).
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

E2E = os.environ.get("RAG_E2E") == "1"
BASE = os.environ.get("RAG_BASE_URL", "http://localhost:8080")

pytestmark = pytest.mark.skipif(not E2E, reason="set RAG_E2E=1 to run against live stack")


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _post_turn(session_id: str, user_id: str | None) -> str:
    """Helper: write a minimal valid turn, return its id."""
    r = httpx.post(
        f"{BASE}/turns",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "timestamp": "2026-05-02T10:00:00Z",
            "metadata": {},
        },
        timeout=30,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert isinstance(body.get("id"), str) and body["id"], body
    return body["id"]


def test_post_turns_returns_201_with_id():
    sid = _unique("sess")
    uid = _unique("user")
    turn_id = _post_turn(sid, uid)
    assert turn_id


def test_post_turns_accepts_null_user_and_tool_messages():
    r = httpx.post(
        f"{BASE}/turns",
        json={
            "session_id": _unique("sess"),
            "user_id": None,
            "messages": [
                {"role": "user", "content": "use the calculator"},
                {"role": "tool", "name": "calc", "content": "2+2=4"},
                {"role": "assistant", "content": "4"},
            ],
            "timestamp": "2026-05-02T10:00:00Z",
            "metadata": {"trace": "t-1"},
        },
        timeout=30,
    )
    assert r.status_code == 201, r.text
    assert isinstance(r.json()["id"], str)


def test_post_recall_cold_returns_empty_shape():
    r = httpx.post(
        f"{BASE}/recall",
        json={
            "query": "anything",
            "session_id": _unique("cold"),
            "user_id": _unique("nouser"),
            "max_tokens": 256,
        },
        timeout=30,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"context": "", "citations": []}


def test_post_search_cold_returns_empty_results():
    r = httpx.post(
        f"{BASE}/search",
        json={
            "query": "anything",
            "session_id": None,
            "user_id": _unique("nouser"),
            "limit": 5,
        },
        timeout=30,
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"results": []}


def test_get_user_memories_returns_empty_list_for_unknown_user():
    r = httpx.get(f"{BASE}/users/{_unique('ghost')}/memories", timeout=10)
    assert r.status_code == 200, r.text
    assert r.json() == {"memories": []}


def test_delete_session_returns_204_and_is_idempotent():
    sid = _unique("sess")
    _post_turn(sid, _unique("user"))
    r = httpx.delete(f"{BASE}/sessions/{sid}", timeout=10)
    assert r.status_code == 204
    assert r.text == ""
    # idempotent: deleting again is still 204
    r = httpx.delete(f"{BASE}/sessions/{sid}", timeout=10)
    assert r.status_code == 204


def test_delete_user_returns_204_and_is_idempotent():
    uid = _unique("user")
    _post_turn(_unique("sess"), uid)
    r = httpx.delete(f"{BASE}/users/{uid}", timeout=10)
    assert r.status_code == 204
    r = httpx.delete(f"{BASE}/users/{uid}", timeout=10)
    assert r.status_code == 204


def test_post_turns_rejects_malformed_payload():
    r = httpx.post(f"{BASE}/turns", json={"session_id": "x"}, timeout=10)
    assert r.status_code == 422


def test_post_recall_rejects_missing_query():
    r = httpx.post(
        f"{BASE}/recall",
        json={"session_id": "x", "max_tokens": 100},
        timeout=10,
    )
    assert r.status_code == 422
