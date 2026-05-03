"""required-test set — MS-20.

Restart persistence, concurrent-session isolation, malformed-input
robustness, synchronous availability after /turns. Skipped unless
RAG_E2E=1 (run against a live `docker compose up`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid

import httpx
import pytest

E2E = os.environ.get("RAG_E2E") == "1"
BASE = os.environ.get("RAG_BASE_URL", "http://localhost:8080")

pytestmark = pytest.mark.skipif(not E2E, reason="set RAG_E2E=1 to run against live stack")


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _wait_healthy(timeout_s: float = 30.0) -> None:
    """Poll /health until 200 or timeout. Used after a container restart."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"/health did not return 200 within {timeout_s}s")


def _post_turn(session_id: str, user_id: str | None, content: str) -> str:
    """Write a single user-message turn and return the turn id."""
    r = httpx.post(
        f"{BASE}/turns",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "noted"},
            ],
            "timestamp": "2026-05-03T10:00:00Z",
            "metadata": {},
        },
        timeout=60,
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_synchronous_availability():
    """: after POST /turns returns, the extracted memory MUST be
    queryable via /users/{id}/memories AND /recall in the next call.
    No eventual consistency.
    """
    sid = _unique("sess-sync")
    uid = _unique("user-sync")
    try:
        _post_turn(sid, uid, "I live in Paris and work as a data scientist.")

        # Memories endpoint should have at least one structured row.
        mem = httpx.get(f"{BASE}/users/{uid}/memories", timeout=10).json()
        assert mem["memories"], f"no memories extracted: {mem}"
        assert any(m["active"] for m in mem["memories"]), mem

        # /recall should surface a `## Known facts` block immediately.
        rec = httpx.post(
            f"{BASE}/recall",
            json={
                "query": "where does this user live",
                "session_id": _unique("recall-sync"),
                "user_id": uid,
                "max_tokens": 1024,
            },
            timeout=30,
        ).json()
        assert "## Known facts about this user" in rec["context"], rec
    finally:
        httpx.delete(f"{BASE}/users/{uid}", timeout=10)


def test_concurrent_sessions_isolated():
    """: two users posting interleaved must not bleed into each
    other's /recall. Cross-session sharing within the same user_id is
    intentional, so we verify isolation at the user_id boundary.
    """
    uid_a = _unique("user-a")
    uid_b = _unique("user-b")
    try:
        # Post a distinct, easy-to-spot fact for each user, interleaved.
        _post_turn(_unique("sess-a"), uid_a, "I have a cat named Mochi.")
        _post_turn(_unique("sess-b"), uid_b, "I drive a 1998 Volvo.")
        _post_turn(_unique("sess-a2"), uid_a, "Mochi loves tuna.")
        _post_turn(_unique("sess-b2"), uid_b, "The Volvo is dark green.")

        # /recall for A must mention Mochi/cat, never Volvo.
        ctx_a = httpx.post(
            f"{BASE}/recall",
            json={
                "query": "what pet does this user have",
                "session_id": _unique("recall-a"),
                "user_id": uid_a,
                "max_tokens": 1024,
            },
            timeout=30,
        ).json()["context"]
        assert "Volvo" not in ctx_a, ctx_a
        assert "Mochi" in ctx_a or "cat" in ctx_a.lower(), ctx_a

        # /recall for B must mention Volvo, never Mochi.
        ctx_b = httpx.post(
            f"{BASE}/recall",
            json={
                "query": "what does this user drive",
                "session_id": _unique("recall-b"),
                "user_id": uid_b,
                "max_tokens": 1024,
            },
            timeout=30,
        ).json()["context"]
        assert "Mochi" not in ctx_b, ctx_b
        assert "Volvo" in ctx_b, ctx_b
    finally:
        httpx.delete(f"{BASE}/users/{uid_a}", timeout=10)
        httpx.delete(f"{BASE}/users/{uid_b}", timeout=10)


def test_malformed_input_no_crash():
    """: bad JSON, missing fields, oversized payloads, unicode
    oddities — all must yield 4xx and the service stays up.
    """
    cases = [
        # bad JSON body
        ("post-bad-json", lambda: httpx.post(
            f"{BASE}/turns",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )),
        # missing required field (messages)
        ("post-missing-fields", lambda: httpx.post(
            f"{BASE}/turns",
            json={"session_id": "x", "user_id": "u", "timestamp": "2026-05-03T10:00:00Z"},
            timeout=10,
        )),
        # wrong type — messages should be list, not string
        ("post-wrong-type", lambda: httpx.post(
            f"{BASE}/turns",
            json={
                "session_id": "x",
                "user_id": "u",
                "messages": "not-a-list",
                "timestamp": "2026-05-03T10:00:00Z",
            },
            timeout=10,
        )),
        # unicode + control characters in content (must NOT crash)
        ("post-unicode-noise", lambda: httpx.post(
            f"{BASE}/turns",
            json={
                "session_id": _unique("sess-u"),
                "user_id": _unique("user-u"),
                "messages": [
                    {"role": "user", "content": "hello   \U0001F600 ‮ RTL"},
                    {"role": "assistant", "content": "ok"},
                ],
                "timestamp": "2026-05-03T10:00:00Z",
                "metadata": {},
            },
            timeout=60,
        )),
        # /recall with missing query
        ("recall-missing-query", lambda: httpx.post(
            f"{BASE}/recall",
            json={"session_id": "x", "max_tokens": 100},
            timeout=10,
        )),
    ]
    for label, call in cases:
        r = call()
        # Either rejected with 4xx OR accepted (unicode noise is valid).
        assert r.status_code < 500, f"{label} crashed: {r.status_code} {r.text}"

    # Service still healthy after all the abuse.
    r = httpx.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200


@pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker CLI not available on host",
)
def test_restart_persistence():
    """: docker compose restart MUST NOT lose data."""
    # docker-compose.yml lives at the repo root; resolve it from this
    # test file rather than depending on pytest's cwd.
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sid = _unique("sess-rst")
    uid = _unique("user-rst")
    try:
        _post_turn(sid, uid, "I have a hamster named Pickle.")
        before = httpx.get(f"{BASE}/users/{uid}/memories", timeout=10).json()
        assert before["memories"], before

        # conftest.py sets OPENAI_API_KEY="" via setdefault; docker compose
        # `${OPENAI_API_KEY:?...}` rejects empty strings, so strip empties
        # from the subprocess env to let `.env` resolve the variable.
        clean_env = {k: v for k, v in os.environ.items() if v}
        result = subprocess.run(
            ["docker", "compose", "restart", "app"],
            capture_output=True, text=True, timeout=60,
            cwd=project_root, env=clean_env,
        )
        if result.returncode != 0:
            pytest.skip(f"docker compose restart failed: {result.stderr.strip()}")
        _wait_healthy(timeout_s=30)

        after = httpx.get(f"{BASE}/users/{uid}/memories", timeout=10).json()
        assert after == before, f"data drift across restart: before={before} after={after}"
    finally:
        httpx.delete(f"{BASE}/users/{uid}", timeout=10)
