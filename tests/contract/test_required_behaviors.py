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


def _inject_embed_failure(count: int, error: str = "connection") -> None:
    """Test-only: load N forced embed failures into the running app."""
    r = httpx.post(
        f"{BASE}/admin/inject_embed_failure",
        params={"count": count, "error": error},
        timeout=5,
    )
    assert r.status_code == 200, r.text


def _clear_injected_failures() -> None:
    """Reset the injection counter so other tests aren't polluted."""
    _inject_embed_failure(0)


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


# =====================================================================
# MS-24 — graceful upstream-failure handling. Tests use the
# /admin/inject_embed_failure knob to simulate transient OpenAI blips
# without monkey-patching (DB is not exposed to the host so in-process
# tests aren't an option).
# =====================================================================


def test_turns_survives_embed_blip():
    """: a transient embed failure must NOT 500 /turns when extraction
    can still produce memories. Turn metadata records the failure.
    """
    sid = _unique("sess-eb")
    uid = _unique("user-eb")
    try:
        _inject_embed_failure(1)
        r = httpx.post(
            f"{BASE}/turns",
            json={
                "session_id": sid,
                "user_id": uid,
                "messages": [
                    {"role": "user", "content": "I work at Notion as a PM in Berlin."},
                    {"role": "assistant", "content": "Got it."},
                ],
                "timestamp": "2026-05-03T10:00:00Z",
                "metadata": {},
            },
            timeout=60,
        )
        assert r.status_code == 201, r.text

        # Memories from the LLM extractor (real call, not faked) survive.
        mem = httpx.get(f"{BASE}/users/{uid}/memories", timeout=10).json()
        assert mem["memories"], f"extractor produced no memories: {mem}"

        # /admin/turn_index reflects the embed failure.
        idx = httpx.get(
            f"{BASE}/admin/turn_index",
            params={"user_id": uid, "session_id": sid, "limit": 5},
            timeout=10,
        ).json()
        assert idx["embed_failures"] >= 1, idx
        # No turn_message documents were indexed for this turn.
        assert idx["documents_count"] == 0, idx
    finally:
        _clear_injected_failures()
        httpx.delete(f"{BASE}/users/{uid}", timeout=10)


def test_recall_returns_facts_when_embed_fails():
    """MS-24: `## Known facts` block must still render when
    embed_query fails — facts come straight from Postgres, no embedding
    needed. Only the conversation block goes cold.
    """
    sid = _unique("sess-rf")
    uid = _unique("user-rf")
    try:
        # Seed real memories first (no failure injected).
        _post_turn(sid, uid, "I live in Paris and work as a data scientist.")

        # Now force the next embed call to fail.
        _inject_embed_failure(1)
        rec = httpx.post(
            f"{BASE}/recall",
            json={
                "query": "where does this user live",
                "session_id": _unique("recall-rf"),
                "user_id": uid,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        assert rec.status_code == 200, rec.text
        body = rec.json()
        assert "## Known facts about this user" in body["context"], body
        assert body["citations"] == [], body
    finally:
        _clear_injected_failures()
        httpx.delete(f"{BASE}/users/{uid}", timeout=10)


def test_recall_fully_cold_when_no_memories_and_embed_fails():
    """cold semantics + MS-24: unknown user with no memories AND
    embed_query failing → 200 with fully empty body, never 500.
    """
    uid = _unique("ghost")
    try:
        _inject_embed_failure(1)
        rec = httpx.post(
            f"{BASE}/recall",
            json={
                "query": "anything",
                "session_id": _unique("recall-cold"),
                "user_id": uid,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        assert rec.status_code == 200, rec.text
        assert rec.json() == {"context": "", "citations": []}
    finally:
        _clear_injected_failures()


def test_fixture_runner_aggregates_failures():
    """MS-24: a transient embed blip on one turn must NOT abort
    the whole fixture run. With graceful /turns the blipped turn
    returns 201 with `embed_error` in metadata (no 503), so the
    aggregate's `ingest_failures` correctly stays at 0; the failure is
    observable via `embed_failures` on /admin/turn_index for the
    fixture users.
    """
    try:
        _inject_embed_failure(1)
        r = httpx.post(
            f"{BASE}/admin/run_fixture",
            json={"name": "v1"},
            timeout=180,
        )
        assert r.status_code == 200, r.text
        agg = r.json()["aggregate"]
        # Fixture completed end-to-end despite the injection.
        assert agg["positive_total"] >= 1, agg
        assert agg["ingest_failures"] == 0, agg

        # Failure is observable on /admin/turn_index for one of the fixture users.
        # We don't know which user got hit (depends on fixture iteration order),
        # so sum the embed_failures across all fix-* users in v1.
        total_embed_failures = 0
        for uid in ("fix-berlin", "fix-pet", "fix-diet"):
            idx = httpx.get(
                f"{BASE}/admin/turn_index",
                params={"user_id": uid, "limit": 1},
                timeout=10,
            ).json()
            total_embed_failures += idx["embed_failures"]
        assert total_embed_failures >= 1, f"no embed_error markers found, expected ≥1"
    finally:
        _clear_injected_failures()


def test_both_embed_and_extract_fail_returns_503():
    """MS-24: when neither embed nor extract has anything to
    persist, /turns returns 503 retryable. Anonymous turn skips
    extraction entirely, so injecting an embed failure leaves us with
    zero useful state — the canonical "both fail" condition.
    """
    sid = _unique("sess-503")
    try:
        _inject_embed_failure(1)
        r = httpx.post(
            f"{BASE}/turns",
            json={
                "session_id": sid,
                "user_id": None,  # extraction skipped
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                "timestamp": "2026-05-03T10:00:00Z",
                "metadata": {},
            },
            timeout=30,
        )
        assert r.status_code == 503, r.text
        body = r.json()
        # FastAPI wraps the dict detail under a "detail" key.
        detail = body.get("detail", body)
        assert detail.get("retryable") is True, body
        assert detail.get("error") == "upstream_unavailable", body
    finally:
        _clear_injected_failures()


# =====================================================================
# MS-23 — query-relevant memories layer (middle priority).
# Without this test, the headline upgrade is only verified through the
# admin fixture runner; a tight-budget regression in the Python sort
# would slip past the eval's pytest invocation.
# =====================================================================


def test_recall_query_aware_memory_ranking():
    """: under tight max_tokens, query-relevant facts must survive
    the budget while irrelevant ones are dropped. Asserts MS-23 sort
    by query similarity, not the legacy type-priority sort.
    """
    sid = sid_recall = _unique("sess-qa")
    uid = _unique("user-qa")
    try:
        # Two turns covering disjoint topics so the user has facts in
        # both employment and diet namespaces. The LLM extracts both
        # batches synchronously inside the /turns transaction.
        _post_turn(sid, uid, "I work at Notion as a product manager.")
        _post_turn(sid, uid,
                   "I'm vegetarian, allergic to shellfish, and don't drink coffee.")

        # Verify both fact families landed before stressing the ranker.
        mem = httpx.get(f"{BASE}/users/{uid}/memories", timeout=10).json()
        keys = {m["key"] for m in mem["memories"] if m["active"]}
        assert any(k.startswith("employment.") for k in keys), keys
        assert any(k in ("diet.style", "allergy", "beverage.coffee") for k in keys), keys

        # The API contract: query-relevant facts come FIRST. Under
        # the legacy type-priority sort, employment.* would lead the
        # block; under MS-23, diet/allergy/beverage lead and
        # employment is pushed to the tail (or out, under tight
        # budget). We assert the order — that's the falsifiable
        # query-aware signal — independent of how many tail facts the
        # budget happens to accommodate.
        rec = httpx.post(
            f"{BASE}/recall",
            json={
                "query": "What does this user eat?",
                "session_id": sid_recall,
                "user_id": uid,
                "max_tokens": 60,
            },
            timeout=30,
        ).json()
        ctx = rec["context"]
        assert "## Known facts about this user" in ctx, ctx

        # Pull the fact lines in render order.
        in_block = False
        ordered_keys: list[str] = []
        for line in ctx.splitlines():
            if line.startswith("## Known facts"):
                in_block = True
                continue
            if line.startswith("## "):
                in_block = False
                continue
            if in_block and line.startswith("- ") and ": " in line:
                key = line[2:].split(": ", 1)[0]
                ordered_keys.append(key)

        diet_keys = {"diet.style", "allergy", "beverage.coffee"}
        employment_keys = {"employment.company", "employment.role"}
        diet_positions = [i for i, k in enumerate(ordered_keys) if k in diet_keys]
        employment_positions = [
            i for i, k in enumerate(ordered_keys) if k in employment_keys
        ]
        assert diet_positions, f"no diet fact in block: {ordered_keys}"
        # If any employment fact appears, every one must come AFTER
        # every diet fact — the query-aware signal.
        if employment_positions:
            assert min(employment_positions) > max(diet_positions), (
                f"employment fact ranked above diet (legacy type-priority "
                f"behaviour): {ordered_keys}"
            )
    finally:
        httpx.delete(f"{BASE}/users/{uid}", timeout=10)
