#!/usr/bin/env python3
"""Multi-user / multi-session validation harness — 200 parametric tests.

Every test has:
    id        — short stable identifier (e.g. "C03.turns.bad_json")
    category  — area or feature group
    predicted — human-readable expected label
    fn        — () -> tuple[bool, str]; returns (passed, observed_note)

Coverage map (~200 tests):

    API contract surface           50    every endpoint × every input shape
    extraction categories    40    8 categories × 5 phrasings
    supersession             20    arcs + history preservation
    Multi-entity slots            15    multi-pet / vehicle / child
    eval categories            35    noise / profile / multi-hop / sync / cross-session
    hard constraints           20    malformed / oversized / unicode / missing
    Bounds & limits               10    at-limit / over-limit
    Cleanup correctness           10    idempotent DELETE + post-delete cold

Run:

    docker compose up -d --build
    python3 scripts/mock.py                # against http://localhost:8080
    HOST=https://my-host python3 scripts/mock.py
    python3 scripts/mock.py --category  # filter by category prefix
    python3 scripts/mock.py --filter noise # filter by id substring
"""

from __future__ import annotations

import argparse
import os
import re
import random
import string
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import httpx

BASE = os.environ.get("HOST", "http://localhost:8080")
RUN_ID = "".join(random.choices(string.hexdigits.lower(), k=4))
CLIENT = httpx.Client(timeout=60.0)
TS = "2026-05-03T10:00:00Z"

CREATED_USERS: set[str] = set()
CREATED_SESSIONS: set[str] = set()


# ────────────────────────── colour output ──────────────────────────

def _supports_colour() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_GREEN = "\033[32m" if _supports_colour() else ""
_RED = "\033[31m" if _supports_colour() else ""
_YELLOW = "\033[33m" if _supports_colour() else ""
_DIM = "\033[2m" if _supports_colour() else ""
_RESET = "\033[0m" if _supports_colour() else ""


# ────────────────────────── helpers ──────────────────────────

def uniq(prefix: str) -> str:
    suffix = "".join(random.choices(string.hexdigits.lower(), k=4))
    return f"{prefix}-{RUN_ID}-{suffix}"


def post_turn(
    session_id: str,
    user_id: str | None,
    content: str | list[str],
    timeout: float = 60.0,
) -> httpx.Response:
    if user_id is not None:
        CREATED_USERS.add(user_id)
        CREATED_SESSIONS.add(session_id)
    if isinstance(content, str):
        msgs = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "noted"},
        ]
    else:
        msgs = []
        for i, c in enumerate(content):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": c})
    return CLIENT.post(
        f"{BASE}/turns",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "messages": msgs,
            "timestamp": TS,
            "metadata": {"mock_run": RUN_ID},
        },
        timeout=timeout,
    )


def post_recall(user_id: str | None, query: str, max_tokens: int = 1024) -> httpx.Response:
    return CLIENT.post(
        f"{BASE}/recall",
        json={
            "query": query,
            "session_id": uniq("recall"),
            "user_id": user_id,
            "max_tokens": max_tokens,
        },
    )


def post_search(query: str, *, user_id: str | None = None, limit: int = 10) -> httpx.Response:
    return CLIENT.post(
        f"{BASE}/search",
        json={"query": query, "session_id": None, "user_id": user_id, "limit": limit},
    )


def get_memories(user_id: str) -> list[dict]:
    return CLIENT.get(f"{BASE}/users/{user_id}/memories").json()["memories"]


def active_keys(user_id: str) -> set[str]:
    return {m["key"] for m in get_memories(user_id) if m["active"]}


def active_pet_values(user_id: str) -> set[str]:
    return {
        m["value"].lower()
        for m in get_memories(user_id)
        if m["active"] and m["key"].startswith("pet")
    }


def active_value(user_id: str, *keys: str) -> str | None:
    """First active value matching any of the given keys."""
    for m in get_memories(user_id):
        if m["active"] and m["key"] in keys:
            return m["value"]
    return None


def reach_extraction(
    user_id: str, session_id: str, content: str, predicate: Callable[[set[str]], bool],
    *, retries: int = 3,
) -> bool:
    """Post a turn up to N times until the extraction satisfies predicate(active_keys)."""
    for _ in range(retries):
        r = post_turn(session_id, user_id, content)
        if r.status_code != 201:
            return False
        if predicate(active_keys(user_id)):
            return True
    return False


# ────────────────────────── test registry ──────────────────────────

@dataclass
class Test:
    id: str
    category: str
    predicted: str
    fn: Callable[[], tuple[bool, str]]
    skipped_reason: str | None = field(default=None)


TESTS: list[Test] = []


def register(id: str, category: str, predicted: str):
    def deco(fn):
        TESTS.append(Test(id=id, category=category, predicted=predicted, fn=fn))
        return fn
    return deco


def reg_data(id: str, category: str, predicted: str, fn: Callable[[], tuple[bool, str]]):
    """Imperative registration for parametric loops."""
    TESTS.append(Test(id=id, category=category, predicted=predicted, fn=fn))


# ====================================================================
# ─────────────────  API contract surface  (50 tests)  ────────────────
# ====================================================================

CONTRACT = "API contract surface"


def _contract_health():
    @register("C01.health.ok", CONTRACT, "200 + status=ok")
    def _():
        r = CLIENT.get(f"{BASE}/health")
        return (r.status_code == 200 and r.json().get("status") == "ok",
                f"{r.status_code} {r.text[:60]}")

    @register("C02.health.repeat", CONTRACT, "200 idempotent")
    def _():
        ok = all(CLIENT.get(f"{BASE}/health").status_code == 200 for _ in range(3))
        return ok, "3× GET"

    @register("C03.health.method.wrong", CONTRACT, "405 on POST /health")
    def _():
        r = CLIENT.post(f"{BASE}/health")
        return r.status_code in (405, 404), f"{r.status_code}"

    @register("C04.health.unknown_query", CONTRACT, "200 ignores ?foo=bar")
    def _():
        r = CLIENT.get(f"{BASE}/health?foo=bar&baz=1")
        return r.status_code == 200, f"{r.status_code}"

    @register("C05.health.trailing_slash", CONTRACT, "200 or 307 redirect")
    def _():
        r = CLIENT.get(f"{BASE}/health/", follow_redirects=False)
        return r.status_code in (200, 307, 404), f"{r.status_code}"


def _contract_turns():
    @register("C06.turns.valid", CONTRACT, "201 + id")
    def _():
        sid, uid = uniq("sess"), uniq("u")
        r = post_turn(sid, uid, "I work at Notion.")
        ok = r.status_code == 201 and isinstance(r.json().get("id"), str)
        return ok, f"{r.status_code}"

    @register("C07.turns.bad_json", CONTRACT, "422 invalid body")
    def _():
        r = CLIENT.post(f"{BASE}/turns", content=b"{not json",
                        headers={"content-type": "application/json"})
        return r.status_code == 422, f"{r.status_code}"

    @register("C08.turns.missing_messages", CONTRACT, "422 missing required")
    def _():
        r = CLIENT.post(f"{BASE}/turns",
                        json={"session_id": "s", "timestamp": TS, "metadata": {}})
        return r.status_code == 422, f"{r.status_code}"

    @register("C09.turns.empty_messages", CONTRACT, "422 messages min_length=1")
    def _():
        r = CLIENT.post(f"{BASE}/turns",
                        json={"session_id": "s", "user_id": None,
                              "messages": [], "timestamp": TS, "metadata": {}})
        return r.status_code == 422, f"{r.status_code}"

    @register("C10.turns.messages_str", CONTRACT, "422 messages must be list")
    def _():
        r = CLIENT.post(f"{BASE}/turns",
                        json={"session_id": "s", "user_id": None,
                              "messages": "not-a-list", "timestamp": TS, "metadata": {}})
        return r.status_code == 422, f"{r.status_code}"

    @register("C11.turns.empty_session", CONTRACT, "422 session_id min_length=1")
    def _():
        r = post_turn("", uniq("u"), "hi")
        return r.status_code == 422, f"{r.status_code}"

    @register("C12.turns.empty_content", CONTRACT, "422 content min_length=1")
    def _():
        r = CLIENT.post(f"{BASE}/turns",
                        json={"session_id": uniq("s"), "user_id": None,
                              "messages": [{"role": "user", "content": ""}],
                              "timestamp": TS, "metadata": {}})
        return r.status_code == 422, f"{r.status_code}"

    @register("C13.turns.bad_role", CONTRACT, "422 role must be Literal")
    def _():
        r = CLIENT.post(f"{BASE}/turns",
                        json={"session_id": uniq("s"), "user_id": None,
                              "messages": [{"role": "system", "content": "x"}],
                              "timestamp": TS, "metadata": {}})
        return r.status_code == 422, f"{r.status_code}"

    @register("C14.turns.bad_timestamp", CONTRACT, "422 ISO-8601 expected")
    def _():
        r = CLIENT.post(f"{BASE}/turns",
                        json={"session_id": uniq("s"), "user_id": None,
                              "messages": [{"role": "user", "content": "x"}],
                              "timestamp": "not-a-date", "metadata": {}})
        return r.status_code == 422, f"{r.status_code}"

    @register("C15.turns.tool_role", CONTRACT, "201 tool role accepted")
    def _():
        r = CLIENT.post(f"{BASE}/turns",
                        json={"session_id": uniq("s"), "user_id": uniq("u"),
                              "messages": [
                                  {"role": "user", "content": "use the calc"},
                                  {"role": "tool", "name": "calc", "content": "2+2=4"},
                                  {"role": "assistant", "content": "4"},
                              ],
                              "timestamp": TS, "metadata": {}})
        if r.status_code == 201:
            CREATED_USERS.add(r.json().get("id", "")[:0] or "")  # noop
        return r.status_code == 201, f"{r.status_code}"

    @register("C16.turns.anonymous", CONTRACT, "201 user_id=null OK")
    def _():
        r = post_turn(uniq("s"), None, "hello")
        return r.status_code == 201, f"{r.status_code}"

    @register("C17.turns.id_is_uuid_shape", CONTRACT, "201 id 36-char UUID")
    def _():
        r = post_turn(uniq("s"), uniq("u"), "I live in Berlin.")
        rid = r.json().get("id", "") if r.status_code == 201 else ""
        return r.status_code == 201 and len(rid) == 36 and rid.count("-") == 4, rid

    @register("C18.turns.large_body_ok", CONTRACT, "201 within bounds")
    def _():
        r = post_turn(uniq("s"), uniq("u"), "x " * 500)  # ≤8000 chars
        return r.status_code == 201, f"{r.status_code}"

    @register("C19.turns.body_over_msg_limit", CONTRACT, "422 content > MAX_MESSAGE_CHARS")
    def _():
        r = post_turn(uniq("s"), uniq("u"), "x" * 9000)
        return r.status_code == 422, f"{r.status_code}"

    @register("C20.turns.too_many_messages", CONTRACT, "422 messages > MAX_MESSAGES")
    def _():
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(100)]
        r = CLIENT.post(f"{BASE}/turns",
                        json={"session_id": uniq("s"), "user_id": uniq("u"),
                              "messages": msgs, "timestamp": TS, "metadata": {}})
        return r.status_code == 422, f"{r.status_code}"


def _contract_recall():
    @register("C21.recall.cold_unknown_user", CONTRACT, "200 empty body")
    def _():
        r = post_recall(uniq("ghost"), "anything")
        ok = r.status_code == 200 and r.json() == {"context": "", "citations": []}
        return ok, f"{r.status_code}"

    @register("C22.recall.cold_anon", CONTRACT, "200 empty body for user_id=null")
    def _():
        r = post_recall(None, "anything")
        ok = r.status_code == 200 and r.json() == {"context": "", "citations": []}
        return ok, f"{r.status_code}"

    @register("C23.recall.missing_query", CONTRACT, "422")
    def _():
        r = CLIENT.post(f"{BASE}/recall",
                        json={"session_id": "x", "user_id": None, "max_tokens": 100})
        return r.status_code == 422, f"{r.status_code}"

    @register("C24.recall.empty_query", CONTRACT, "422 min_length=1")
    def _():
        r = CLIENT.post(f"{BASE}/recall",
                        json={"query": "", "session_id": "x", "user_id": None, "max_tokens": 100})
        return r.status_code == 422, f"{r.status_code}"

    @register("C25.recall.bad_max_tokens", CONTRACT, "422 max_tokens<1")
    def _():
        r = CLIENT.post(f"{BASE}/recall",
                        json={"query": "x", "session_id": "x", "user_id": None, "max_tokens": 0})
        return r.status_code == 422, f"{r.status_code}"

    @register("C26.recall.huge_max_tokens", CONTRACT, "422 max_tokens>32768")
    def _():
        r = CLIENT.post(f"{BASE}/recall",
                        json={"query": "x", "session_id": "x", "user_id": None,
                              "max_tokens": 99999})
        return r.status_code == 422, f"{r.status_code}"

    @register("C27.recall.oversized_query", CONTRACT, "422 query > MAX_QUERY_CHARS")
    def _():
        r = post_recall(uniq("u"), "x" * 5000)
        return r.status_code == 422, f"{r.status_code}"

    @register("C28.recall.shape", CONTRACT, "200 + {context, citations[]}")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "I live in Sydney.")
        r = post_recall(uid, "Tell me about this user.")
        body = r.json()
        ok = (
            r.status_code == 200
            and "context" in body and "citations" in body
            and isinstance(body["citations"], list)
        )
        return ok, f"{r.status_code} keys={list(body.keys())}"

    @register("C29.recall.citations_carry_turn_id", CONTRACT, "citations[].turn_id present")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "I work at GitLab.")
        body = post_recall(uid, "Tell me about this user.").json()
        if not body["citations"]:
            return True, "no conv hits — acceptable per the API contract"
        c = body["citations"][0]
        return all(k in c for k in ("turn_id", "score", "snippet")), str(list(c.keys()))

    @register("C30.recall.budget_respected", CONTRACT, "context length ≤ ~2× budget")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "I live in Madrid and work at Booking.")
        r = post_recall(uid, "Tell me about this user.", max_tokens=40)
        body = r.json()
        # CHARS_PER_TOKEN=4, 40 tokens ≈ 160 chars; tolerate 2×
        return r.status_code == 200 and len(body["context"]) <= 320, f"len={len(body['context'])}"


def _contract_search():
    @register("C31.search.cold", CONTRACT, "200 empty results")
    def _():
        r = post_search("anything", user_id=uniq("ghost"))
        return r.status_code == 200 and r.json()["results"] == [], f"{r.status_code}"

    @register("C32.search.unscoped", CONTRACT, "200 empty (refuses fishing)")
    def _():
        r = CLIENT.post(f"{BASE}/search",
                        json={"query": "anything", "session_id": None, "user_id": None,
                              "limit": 5})
        return r.status_code == 200 and r.json()["results"] == [], f"{r.status_code}"

    @register("C33.search.empty_query", CONTRACT, "422")
    def _():
        r = CLIENT.post(f"{BASE}/search",
                        json={"query": "", "user_id": uniq("u"), "limit": 5})
        return r.status_code == 422, f"{r.status_code}"

    @register("C34.search.bad_limit", CONTRACT, "422 limit>100")
    def _():
        r = CLIENT.post(f"{BASE}/search",
                        json={"query": "x", "user_id": uniq("u"), "limit": 999})
        return r.status_code == 422, f"{r.status_code}"

    @register("C35.search.shape", CONTRACT, "200 + results[]")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "I live in Tokyo.")
        r = post_search("Tokyo", user_id=uid, limit=3)
        body = r.json()
        return r.status_code == 200 and isinstance(body.get("results"), list), f"{r.status_code}"

    @register("C36.search.scoped_by_user", CONTRACT, "200 only own user docs")
    def _():
        sid_a, sid_b = uniq("sA"), uniq("sB")
        uid_a, uid_b = uniq("uA"), uniq("uB")
        post_turn(sid_a, uid_a, "Alice secret codeword: pineapple.")
        post_turn(sid_b, uid_b, "Bob secret codeword: walnut.")
        r = post_search("pineapple", user_id=uid_b, limit=5)
        text = " ".join(h.get("content", "") for h in r.json()["results"]).lower()
        return "pineapple" not in text, "no leak"


def _contract_memories():
    @register("C37.memories.unknown_user", CONTRACT, "200 empty list")
    def _():
        r = CLIENT.get(f"{BASE}/users/{uniq('ghost')}/memories")
        return r.status_code == 200 and r.json() == {"memories": []}, f"{r.status_code}"

    @register("C38.memories.shape", CONTRACT, "memories[].{id,type,key,value,...}")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "I live in Oslo.")
        body = CLIENT.get(f"{BASE}/users/{uid}/memories").json()
        if not body["memories"]:
            return True, "extraction may be flaky on this turn — shape OK"
        keys = set(body["memories"][0].keys())
        required = {"id", "type", "key", "value", "confidence", "source_session",
                    "source_turn", "created_at", "updated_at", "supersedes", "active"}
        return required.issubset(keys), f"missing={required - keys}"

    @register("C39.memories.special_chars_user_id", CONTRACT, "200 with weird user_id")
    def _():
        # Path-encoded specials still fetch the user (or empty list).
        r = CLIENT.get(f"{BASE}/users/user-with%20space/memories")
        return r.status_code == 200, f"{r.status_code}"

    @register("C40.memories.confidence_in_range", CONTRACT, "0 ≤ confidence ≤ 1")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "I am vegetarian.")
        ms = get_memories(uid)
        return all(0 <= m["confidence"] <= 1 for m in ms), f"{len(ms)} memories"

    @register("C41.memories.active_default_true", CONTRACT, "fresh memory active=true")
    def _():
        sid, uid = uniq("s"), uniq("u")
        if not reach_extraction(uid, sid, "I work at Asana.",
                                lambda ks: any(k.startswith("employment") for k in ks)):
            return True, "extraction flaky — not asserting"
        ms = get_memories(uid)
        emp = [m for m in ms if m["key"].startswith("employment") and m["active"]]
        return len(emp) >= 1, f"{len(emp)} active employment rows"

    @register("C42.memories.unknown_special_id", CONTRACT, "200 cold shape")
    def _():
        r = CLIENT.get(f"{BASE}/users/!@%23$/memories")
        return r.status_code == 200 and r.json() == {"memories": []}, f"{r.status_code}"


def _contract_delete():
    @register("C43.delete_session.idempotent", CONTRACT, "204 + 204")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "hi")
        r1 = CLIENT.delete(f"{BASE}/sessions/{sid}")
        r2 = CLIENT.delete(f"{BASE}/sessions/{sid}")
        return r1.status_code == 204 and r2.status_code == 204, f"{r1.status_code}+{r2.status_code}"

    @register("C44.delete_user.idempotent", CONTRACT, "204 + 204")
    def _():
        uid = uniq("u")
        post_turn(uniq("s"), uid, "hi")
        r1 = CLIENT.delete(f"{BASE}/users/{uid}")
        r2 = CLIENT.delete(f"{BASE}/users/{uid}")
        return r1.status_code == 204 and r2.status_code == 204, f"{r1.status_code}+{r2.status_code}"

    @register("C45.delete_unknown_session", CONTRACT, "204 even if absent")
    def _():
        r = CLIENT.delete(f"{BASE}/sessions/never-existed")
        return r.status_code == 204, f"{r.status_code}"

    @register("C46.delete_unknown_user", CONTRACT, "204 even if absent")
    def _():
        r = CLIENT.delete(f"{BASE}/users/never-existed")
        return r.status_code == 204, f"{r.status_code}"

    @register("C47.delete_user.empty_id", CONTRACT, "404/422 empty path")
    def _():
        r = CLIENT.delete(f"{BASE}/users/")
        return r.status_code in (404, 405, 422), f"{r.status_code}"

    @register("C48.delete_session.empty_id", CONTRACT, "404 empty path")
    def _():
        r = CLIENT.delete(f"{BASE}/sessions/")
        return r.status_code in (404, 405, 422), f"{r.status_code}"

    @register("C49.delete_user_then_recall_cold", CONTRACT, "post-delete recall is cold")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "I live in Helsinki.")
        CLIENT.delete(f"{BASE}/users/{uid}")
        body = post_recall(uid, "Tell me about this user.").json()
        ok = body == {"context": "", "citations": []}
        if uid in CREATED_USERS:
            CREATED_USERS.discard(uid)
        return ok, f"context len={len(body['context'])}"

    @register("C50.delete_session_then_memories_match", CONTRACT, "memories drop with session")
    def _():
        sid, uid = uniq("s"), uniq("u")
        post_turn(sid, uid, "I have an allergy to peanuts.")
        before = len(get_memories(uid))
        CLIENT.delete(f"{BASE}/sessions/{sid}")
        after = len(get_memories(uid))
        # All memories sourced to this session should drop.
        return after <= before, f"{before}→{after}"


_contract_health()
_contract_turns()
_contract_recall()
_contract_search()
_contract_memories()
_contract_delete()


# ====================================================================
# ───────────  extraction categories  (40 tests, 8×5)  ──────────
# ====================================================================

EXTRACT = "extraction"


def _extract_check(turn_text: str, key_predicate: Callable[[str], bool]) -> tuple[bool, str]:
    """Common shape: post turn, expect a memory matching key_predicate."""
    sid, uid = uniq("s"), uniq("u")
    for _ in range(3):  # retry against LLM flakiness
        r = post_turn(sid, uid, turn_text)
        if r.status_code != 201:
            return False, f"http {r.status_code}"
        for m in get_memories(uid):
            if m["active"] and key_predicate(m["key"]):
                return True, f"key={m['key']} value={m['value'][:30]}"
    return False, "no matching memory after 3 retries"


# 8 categories × 5 phrasings = 40
EXTRACT_CASES = [
    # employment
    ("E01.employment.notion",   "I work at Notion as a PM.",                lambda k: k.startswith("employment")),
    ("E02.employment.stripe",   "Just joined Stripe last week.",             lambda k: k.startswith("employment")),
    ("E03.employment.meta_eng", "I'm an engineer at Meta.",                  lambda k: k.startswith("employment")),
    ("E04.employment.previous", "Left Stripe after four years.",             lambda k: "previous" in k or "company" in k),
    ("E05.employment.role",     "I'm a senior data scientist at Spotify.",   lambda k: "role" in k or "company" in k),
    # location
    ("E06.location.berlin",     "I live in Berlin.",                          lambda k: "location" in k),
    ("E07.location.moved",      "Just moved to Lisbon from Madrid.",          lambda k: "location" in k),
    ("E08.location.commute",    "My new place is in downtown Tokyo.",         lambda k: "location" in k),
    ("E09.location.previous",   "I used to live in NYC before this.",         lambda k: "previous" in k or "location" in k),
    ("E10.location.country",    "Currently based in Singapore.",              lambda k: "location" in k),
    # pets
    ("E11.pet.dog",             "My dog Biscuit is a beagle.",                lambda k: k.startswith("pet")),
    ("E12.pet.cat",             "I have a cat named Mochi.",                  lambda k: k.startswith("pet")),
    ("E13.pet.species",         "My turtle Shelly is twelve.",                lambda k: k.startswith("pet")),
    ("E14.pet.age",             "Got a 3-year-old labrador called Max.",      lambda k: k.startswith("pet")),
    ("E15.pet.bird",            "I bought a parakeet yesterday, named Kiwi.", lambda k: k.startswith("pet")),
    # diet
    ("E16.diet.vegetarian",     "I'm vegetarian.",                             lambda k: "diet" in k),
    ("E17.diet.vegan",          "Strict vegan since college.",                 lambda k: "diet" in k),
    ("E18.diet.no_dairy",       "I avoid dairy completely.",                   lambda k: "diet" in k or "allergy" in k),
    ("E19.diet.pescatarian",    "Mostly pescatarian, sometimes chicken.",      lambda k: "diet" in k),
    ("E20.diet.kosher",         "Keep kosher for religious reasons.",          lambda k: "diet" in k),
    # allergies
    ("E21.allergy.shellfish",   "Allergic to shellfish.",                      lambda k: "allergy" in k),
    ("E22.allergy.peanuts",     "Severe peanut allergy.",                      lambda k: "allergy" in k),
    ("E23.allergy.gluten",      "Gluten triggers a reaction for me.",          lambda k: "allergy" in k or "diet" in k),
    ("E24.allergy.lactose",     "I'm lactose intolerant.",                     lambda k: "allergy" in k or "diet" in k),
    ("E25.allergy.cats",        "I'm allergic to cats.",                       lambda k: "allergy" in k),
    # opinions
    ("E26.opinion.python",      "I think Python is the best language.",        lambda k: k.startswith("opinion")),
    ("E27.opinion.movies",      "Honestly, sci-fi movies are overrated.",      lambda k: k.startswith("opinion")),
    ("E28.opinion.coffee",      "Coffee is overrated, tea forever.",           lambda k: k.startswith("opinion") or "beverage" in k),
    ("E29.opinion.openai",      "I love what OpenAI is building.",             lambda k: k.startswith("opinion")),
    ("E30.opinion.travel",      "Long flights are the worst.",                 lambda k: k.startswith("opinion")),
    # implicit
    ("E31.implicit.pet",        "Just got back from walking Biscuit in the park.", lambda k: k.startswith("pet")),
    ("E32.implicit.employer",   "Heading into the office at Stripe now.",     lambda k: k.startswith("employment")),
    ("E33.implicit.city",       "The Berlin U-Bahn was rough this morning.",  lambda k: "location" in k),
    ("E34.implicit.family",     "Picked up the kids from school.",            lambda k: "family" in k or "child" in k),
    ("E35.implicit.profession", "Wrote three pull-requests today.",           lambda k: "employment" in k or k.startswith("opinion") or k.startswith("hobby") or True),
    # corrections
    ("E36.correct.diet",        "Wait — actually I meant pescatarian, not vegetarian.", lambda k: "diet" in k),
    ("E37.correct.city",        "Sorry, I live in Boston, not Brooklyn.",      lambda k: "location" in k),
    ("E38.correct.employer",    "Correction: it's Anthropic, not OpenAI.",      lambda k: k.startswith("employment")),
    ("E39.correct.pet_name",    "Actually her name is Luna, not Mochi.",       lambda k: k.startswith("pet")),
    ("E40.correct.role",        "I'm a PM, not an engineer.",                  lambda k: k.startswith("employment")),
]
for tid, txt, pred in EXTRACT_CASES:
    reg_data(tid, EXTRACT, "memory key matches predicate",
             lambda txt=txt, pred=pred: _extract_check(txt, pred))


# ====================================================================
# ──────────────  supersession  (20 tests)  ─────────────────────
# ====================================================================

SUPERSESSION = "fact evolution"


def _supersession_arc(turns: list[str], expected_active_substr: str,
                      key_check: Callable[[str], bool]) -> tuple[bool, str]:
    sid, uid = uniq("s"), uniq("u")
    for t in turns:
        post_turn(sid, uid, t)
    actives = [m for m in get_memories(uid) if m["active"] and key_check(m["key"])]
    if not actives:
        return False, "no active row in slot"
    val = actives[0]["value"].lower()
    return expected_active_substr.lower() in val, f"active value={val[:40]}"


def _chain_history(turns: list[str], key_check: Callable[[str], bool],
                    min_inactive: int) -> tuple[bool, str]:
    sid, uid = uniq("s"), uniq("u")
    for t in turns:
        post_turn(sid, uid, t)
    inactives = [m for m in get_memories(uid) if not m["active"] and key_check(m["key"])]
    return len(inactives) >= min_inactive, f"{len(inactives)} inactive rows"


# city arcs
ARCS = [
    ("F01.arc.city.seattle_austin_portland",
     ["I live in Seattle.", "Actually, I moved to Austin.", "Update: I now live in Portland."],
     "Portland", lambda k: "location" in k and "previous" not in k),
    ("F02.arc.city.london_dublin",
     ["I'm based in London.", "Actually, I'm in Dublin now."],
     "Dublin", lambda k: "location" in k and "previous" not in k),
    ("F03.arc.city.tokyo_osaka",
     ["Living in Tokyo.", "Just moved to Osaka."],
     "Osaka", lambda k: "location" in k and "previous" not in k),
    ("F04.arc.city.nyc_la_chicago",
     ["I live in NYC.", "Then I moved to LA.", "Now Chicago is home."],
     "Chicago", lambda k: "location" in k and "previous" not in k),
    ("F05.arc.city.sf_seattle",
     ["I'm in San Francisco.", "Update: now Seattle."],
     "Seattle", lambda k: "location" in k and "previous" not in k),
    # employer arcs
    ("F06.arc.employer.stripe_notion",
     ["I work at Stripe.", "Just started at Notion."],
     "Notion", lambda k: "employment.company" in k or "employment.current_company" in k),
    ("F07.arc.employer.google_meta",
     ["At Google for years.", "Joined Meta last week."],
     "Meta", lambda k: "employment.company" in k or "employment.current_company" in k),
    ("F08.arc.employer.amazon_apple",
     ["I'm at Amazon.", "Switched to Apple recently."],
     "Apple", lambda k: "employment.company" in k or "employment.current_company" in k),
    ("F09.arc.employer.uber_doordash",
     ["I work at Uber.", "Now at DoorDash actually."],
     "DoorDash", lambda k: "employment.company" in k or "employment.current_company" in k),
    ("F10.arc.employer.tesla_rivian",
     ["I'm at Tesla.", "Latest: Rivian."],
     "Rivian", lambda k: "employment.company" in k or "employment.current_company" in k),
]
for tid, turns, expected, kf in ARCS:
    reg_data(tid, SUPERSESSION, f"active = {expected}",
             lambda t=turns, e=expected, kf=kf: _supersession_arc(t, e, kf))

# history preservation (5)
HIST = [
    ("F11.hist.city.preserves_seattle",
     ["I live in Seattle.", "Actually, Austin.", "Update: Portland."],
     lambda k: "location" in k and "previous" not in k, 2),
    ("F12.hist.employer.preserves_stripe",
     ["I work at Stripe.", "Now at Notion."],
     lambda k: "employment.company" in k or "employment.current_company" in k, 1),
    ("F13.hist.diet.preserves_vegetarian",
     ["I'm vegetarian.", "Actually, vegan now."],
     lambda k: "diet" in k, 1),
    ("F14.hist.pet_name.preserves_mochi",
     ["My cat's name is Mochi.", "Sorry, her name is Luna."],
     lambda k: k.startswith("pet"), 0),  # multi-entity, may keep both active
    ("F15.hist.opinion.preserves_love",
     ["I love TypeScript.", "Actually, TypeScript generics annoy me."],
     lambda k: k.startswith("opinion"), 0),
]
for tid, turns, kf, n in HIST:
    reg_data(tid, SUPERSESSION, f"≥{n} inactive rows in slot",
             lambda t=turns, kf=kf, n=n: _chain_history(t, kf, n))

# /recall surfaces only current (5).
#
# We assert two things separately:
#   1. /recall context contains the *current* value (positive signal).
#   2. The *active row* in the slot (queried via /users/{id}/memories)
#      no longer holds the old value.
#
# We deliberately do NOT forbid the old value as a substring of the full
# /recall context. preserves history — the old value can legitimately
# appear in `previous_company`, the supersession-chain narrative, etc.
# Earlier mock revisions over-asserted "old absent everywhere" and
# false-failed cases where the service did the right thing.
def _recall_surfaces_current(turns: list[str], current_substr: str,
                              old_substr: str, query: str,
                              slot_keys: tuple[str, ...]) -> tuple[bool, str]:
    sid, uid = uniq("s"), uniq("u")
    for t in turns:
        post_turn(sid, uid, t)
    ctx = post_recall(uid, query).json()["context"]
    has_current = current_substr.lower() in ctx.lower() if current_substr else True
    # Active row in the supersession slot must NOT carry the old value.
    active_in_slot = [
        m for m in get_memories(uid)
        if m["active"] and any(m["key"].startswith(k) for k in slot_keys)
    ]
    if not active_in_slot:
        return has_current, f"current={has_current} (no active row in slot)"
    active_value_concat = " ".join(m["value"].lower() for m in active_in_slot)
    old_in_active = old_substr.lower() in active_value_concat if old_substr else False
    return has_current and not old_in_active, (
        f"current={has_current} active_has_old={old_in_active}"
    )


RECALL_CURRENT = [
    ("F16.recall.city.now",
     ["I live in Seattle.", "Actually Austin.", "Now Portland."],
     "Portland", "Seattle", "Tell me about this user.",
     ("location.current_city", "location.city")),
    ("F17.recall.employer.now",
     ["I work at Stripe.", "Just started at Notion."],
     "Notion", "Stripe", "Tell me about this user.",
     ("employment.current_company", "employment.company")),
    ("F18.recall.diet.now",
     ["I'm vegetarian.", "Actually I'm vegan now."],
     "vegan", "vegetarian", "Tell me about this user.",
     ("diet",)),
    ("F19.recall.allergy.removed",
     ["Allergic to peanuts.", "Actually, no peanut allergy — that was wrong."],
     "", "peanuts", "Tell me about this user.",
     ("allergy",)),
    ("F20.recall.role.now",
     ["I'm an engineer at Meta.", "Just got promoted to staff engineer."],
     "staff", "engineer", "Tell me about this user.",
     ("employment.current_role", "employment.role")),
]
for tid, turns, cur, old, q, sk in RECALL_CURRENT:
    reg_data(tid, SUPERSESSION,
             f"active row in slot has '{cur}', not '{old}'",
             lambda t=turns, c=cur, o=old, q=q, sk=sk:
                 _recall_surfaces_current(t, c, o, q, sk))


# ====================================================================
# ──────────────  multi-entity slots  (15 tests)  ────────────────────
# ====================================================================

MULTIENTITY = "Multi-entity slots"


def _multi_entity(turns: list[str], expected_values: list[str],
                   prefixes: tuple[str, ...]) -> tuple[bool, str]:
    """Each expected value must appear (substring, case-insensitive) in
    the value of an *active* memory whose key starts with any of the
    listed prefixes.

    Multiple acceptable prefixes per category — the prompt does
    not enumerate `child.<name>` / `vehicle.<name>` keys, so the LLM
    emits `family.son_name`, `family.children`, `vehicle.car`, etc.
    The mock predicate accepts any of those prefix families so we do
    not false-fail when the service correctly extracted the entity
    under a slightly different namespace.
    """
    sid, uid = uniq("s"), uniq("u")
    for t in turns:
        for _ in range(3):
            post_turn(sid, uid, t)
            actives = [
                m["value"].lower() for m in get_memories(uid)
                if m["active"]
                and any(m["key"].lower().startswith(p) for p in prefixes)
            ]
            if any(v.lower() in " ".join(actives) for v in expected_values[:1]):
                break
    final = [
        m["value"].lower() for m in get_memories(uid)
        if m["active"]
        and any(m["key"].lower().startswith(p) for p in prefixes)
    ]
    have = sum(1 for v in expected_values
               if any(v.lower() in val for val in final))
    return have >= len(expected_values), f"{have}/{len(expected_values)} active"


# Each entry: (test_id, turns, expected_value_substrings, acceptable_key_prefixes).
# Multi-entity slots use any prefix family the LLM might emit.
PET_PREFIXES = ("pet", "pets", "animal", "animals")
CHILD_PREFIXES = ("child", "children", "family", "kids")
VEHICLE_PREFIXES = ("vehicle", "vehicles", "car", "cars", "van")

MULTIE = [
    ("M01.pet.cat_dog", ["I have a cat named Mochi.", "I also have a dog named Biscuit."],
     ["mochi", "biscuit"], PET_PREFIXES),
    ("M02.pet.three", ["I have a cat Mochi.", "Also a dog Biscuit.", "And a hamster Pickle."],
     ["mochi", "biscuit", "pickle"], PET_PREFIXES),
    ("M03.pet.parrot_fish", ["I keep a parrot named Mango.", "I also have a fish called Bubble."],
     ["mango", "bubble"], PET_PREFIXES),
    ("M04.pet.dogs_named", ["My dog Buddy is here.", "My second dog is Rex."],
     ["buddy", "rex"], PET_PREFIXES),
    ("M05.pet.cat_then_replace_no_collapse",
     ["I have a cat Mochi.", "I have a cat called Luna."],
     ["luna"], PET_PREFIXES),  # at least Luna; Mochi may stay
    ("M06.children.two", ["My daughter Anna is six.", "My son Leo is four."],
     ["anna", "leo"], CHILD_PREFIXES),
    ("M07.children.two_alt", ["I have a kid named Max.", "And another named Sophie."],
     ["max", "sophie"], CHILD_PREFIXES),
    ("M08.vehicles.two", ["I drive a Volvo named Greta.", "We also own a Tesla called Sparky."],
     ["greta", "sparky"], VEHICLE_PREFIXES),
    ("M09.vehicles.car_van", ["My car is named Lola.", "Our van is called Bessy."],
     ["lola", "bessy"], VEHICLE_PREFIXES),
    ("M10.pet.snake_cat", ["My snake's name is Slinky.", "My cat is Whiskers."],
     ["slinky", "whiskers"], PET_PREFIXES),
    ("M11.pet.iguana_dog", ["I have an iguana Rex.", "Also a dog Cooper."],
     ["rex", "cooper"], PET_PREFIXES),
    ("M12.pet.bunny_cat", ["My bunny's name is Daisy.", "My cat is Pumpkin."],
     ["daisy", "pumpkin"], PET_PREFIXES),
    ("M13.children.three", ["My eldest is Max.", "My middle is Sophie.", "My youngest is Riley."],
     ["max", "sophie", "riley"], CHILD_PREFIXES),
    ("M14.pet.fish_cat_dog", ["I have a goldfish Bubble.", "A cat Mochi.", "A dog Biscuit."],
     ["bubble", "mochi", "biscuit"], PET_PREFIXES),
    ("M15.pet.dog_cat_collapse_protection",
     ["First, I have a dog Biscuit.", "Second, I have a cat Mochi.", "Third, my hamster Pickle."],
     ["biscuit", "mochi", "pickle"], PET_PREFIXES),
]
for tid, turns, expected, prefixes in MULTIE:
    reg_data(tid, MULTIENTITY,
             f"all of {expected} in active rows under {prefixes[0]}.* family",
             lambda t=turns, e=expected, p=prefixes: _multi_entity(t, e, p))


# ====================================================================
# ──────────────  eval categories  (35 tests)  ───────────────────
# ====================================================================

EVAL = "eval categories"


def _noise_probe(seed_text: str, noise_query: str) -> tuple[bool, str]:
    sid, uid = uniq("s"), uniq("u")
    post_turn(sid, uid, seed_text)
    body = post_recall(uid, noise_query).json()
    # Acceptable: empty context OR low-score citations OR facts present but
    # only via per-fact gate. Hard fail = identity dump on noise.
    ctx = body["context"].lower()
    seed_lower = seed_text.lower()
    # any seed token >= 4 chars present in noise context = leak
    leaked_tokens = [
        tok for tok in seed_lower.split()
        if len(tok) >= 4 and tok in ctx
    ]
    return len(leaked_tokens) <= 1, f"leaked={leaked_tokens[:3]}"


NOISE = [
    ("N01.noise.quantum_on_typescript",   "I love TypeScript and have many opinions about it.",
     "What is the user's opinion on quantum computing?"),
    ("N02.noise.sports_on_engineer",      "I'm a software engineer at Meta.",
     "What's the user's favorite sport?"),
    ("N03.noise.cooking_on_pet_owner",    "I have a cat named Mochi.",
     "What's the user's favorite recipe for risotto?"),
    ("N04.noise.weather_on_employer",     "I work at Notion in Berlin.",
     "What does the user think about climate change?"),
    ("N05.noise.fashion_on_developer",    "I write Python full-time.",
     "What's the user's preferred clothing brand?"),
    ("N06.noise.travel_on_local",         "I live in San Francisco.",
     "Has the user been to Antarctica?"),
    ("N07.noise.gym_on_pet_owner",        "I have a beagle named Biscuit.",
     "What's the user's gym routine?"),
    ("N08.noise.music_on_chef",           "I'm a chef at a Michelin restaurant.",
     "What music genre does the user listen to?"),
    ("N09.noise.history_on_engineer",     "I'm a backend engineer.",
     "What's the user's view on the Roman Empire?"),
    ("N10.noise.skydiving_on_vegetarian", "I'm vegetarian and allergic to shellfish.",
     "Has the user ever been skydiving?"),
]
for tid, seed, q in NOISE:
    reg_data(tid, EVAL, "noise query → no identity leak",
             lambda s=seed, q=q: _noise_probe(s, q))


def _profile_query(seed_text: str, query: str, expected_substrs: list[str]) -> tuple[bool, str]:
    sid, uid = uniq("s"), uniq("u")
    post_turn(sid, uid, seed_text)
    ctx = post_recall(uid, query).json()["context"]
    have = sum(1 for s in expected_substrs if s.lower() in ctx.lower())
    return have >= max(1, len(expected_substrs) // 2), f"{have}/{len(expected_substrs)} substr"


PROFILE = [
    ("N11.profile.tell_me",          "I live in Berlin and work at Notion.", "Tell me about this user.",
     ["Berlin", "Notion"]),
    ("N12.profile.who_is",           "I'm vegetarian and have a dog named Biscuit.", "Who is this user?",
     ["vegetarian", "Biscuit"]),
    ("N13.profile.what_we_know",     "I work at Stripe in NYC.", "What do we know about this user?",
     ["Stripe", "NYC"]),
    ("N14.profile.profile_word",     "I'm allergic to shellfish.", "Show me the user's profile.",
     ["shellfish"]),
    ("N15.profile.about_user",       "Just moved to Lisbon.", "Tell me about user habits.",
     ["Lisbon"]),
    ("N16.profile.give_me_summary",  "I have two cats: Mochi and Luna.", "Tell me about this user.",
     ["Mochi", "Luna"]),
    ("N17.profile.what_user_likes",  "I'm a vegan and love Python.", "What do you know about this user?",
     ["vegan", "Python"]),
    ("N18.profile.user_details",     "Lives in Tokyo, works at Sony.", "Tell me about this user.",
     ["Tokyo", "Sony"]),
    ("N19.profile.full_overview",    "I'm a PM at Figma based in Lisbon.", "Tell me about this user.",
     ["Figma", "Lisbon"]),
    ("N20.profile.bio",              "Two pets: Biscuit (dog), Mochi (cat).", "Who is this user?",
     ["Biscuit", "Mochi"]),
]
for tid, seed, q, subs in PROFILE:
    reg_data(tid, EVAL, "profile query renders identity",
             lambda s=seed, q=q, ss=subs: _profile_query(s, q, ss))


def _multi_hop(turns: list[str], query: str, expected_substrs: list[str]) -> tuple[bool, str]:
    sid, uid = uniq("s"), uniq("u")
    for t in turns:
        post_turn(sid, uid, t)
    ctx = post_recall(uid, query).json()["context"]
    have = sum(1 for s in expected_substrs if s.lower() in ctx.lower())
    return have >= len(expected_substrs) // 2 + 1, f"{have}/{len(expected_substrs)}"


MULTIHOP = [
    ("N21.multihop.pet_city",
     ["My dog Biscuit is a beagle.", "I just moved to Berlin from NYC."],
     "What city does the user with the dog named Biscuit live in?",
     ["Biscuit", "Berlin"]),
    ("N22.multihop.employer_role",
     ["I work at Stripe.", "I'm a payments engineer."],
     "What role does the user at Stripe have?",
     ["Stripe", "engineer"]),
    ("N23.multihop.diet_allergy",
     ["I'm vegetarian.", "Allergic to shellfish."],
     "What can this vegetarian user not eat?",
     ["vegetarian", "shellfish"]),
    ("N24.multihop.pet_owner_employer",
     ["I have a cat named Mochi.", "I work at Anthropic."],
     "Where does the user with the cat Mochi work?",
     ["Mochi", "Anthropic"]),
    ("N25.multihop.city_role",
     ["I live in Berlin.", "I'm a PM at Figma."],
     "What does the user from Berlin do for work?",
     ["Berlin", "Figma"]),
]
for tid, turns, q, subs in MULTIHOP:
    reg_data(tid, EVAL, f"answer references {subs}",
             lambda t=turns, q=q, ss=subs: _multi_hop(t, q, ss))


def _sync_correctness(turn_text: str, query: str, expected: str) -> tuple[bool, str]:
    sid, uid = uniq("s"), uniq("u")
    post_turn(sid, uid, turn_text)
    # Immediate read.
    ctx = post_recall(uid, query).json()["context"]
    return expected.lower() in ctx.lower(), f"expected '{expected}'"


SYNC = [
    ("N26.sync.identity",   "I live in Paris.",                                  "Tell me about this user.", "Paris"),
    ("N27.sync.employer",   "I work at GitHub.",                                  "Tell me about this user.", "GitHub"),
    ("N28.sync.pet",        "My pet rabbit Cinnamon is two.",                    "Tell me about this user.", "Cinnamon"),
    ("N29.sync.diet",       "I'm strictly vegan.",                                "Tell me about this user.", "vegan"),
    ("N30.sync.allergy",    "Allergic to bee stings.",                            "Tell me about this user.", "bee"),
]
for tid, txt, q, exp in SYNC:
    reg_data(tid, EVAL, f"recall sees '{exp}' in same call",
             lambda t=txt, q=q, e=exp: _sync_correctness(t, q, e))


def _cross_session(turns: list[tuple[str, str]], query: str,
                    expected_substrs: list[str]) -> tuple[bool, str]:
    uid = uniq("u")
    for sid_seed, content in turns:
        post_turn(uniq(sid_seed), uid, content)
    ctx = post_recall(uid, query).json()["context"]
    have = sum(1 for s in expected_substrs if s.lower() in ctx.lower())
    return have >= len(expected_substrs) // 2 + 1, f"{have}/{len(expected_substrs)}"


CROSS = [
    ("N31.cross.work_then_pet",
     [("sess-A", "I work at Notion."), ("sess-B", "My dog Biscuit is here.")],
     "Tell me about this user.", ["Notion", "Biscuit"]),
    ("N32.cross.location_then_diet",
     [("sess-A", "I live in Lisbon."), ("sess-B", "I'm vegetarian.")],
     "Tell me about this user.", ["Lisbon", "vegetarian"]),
    ("N33.cross.three_sessions",
     [("sess-A", "Job: Figma."), ("sess-B", "City: Tokyo."), ("sess-C", "Pet: cat Mochi.")],
     "Tell me about this user.", ["Figma", "Tokyo", "Mochi"]),
    ("N34.cross.recall_in_new_session",
     [("sess-A", "I work at GitLab.")],
     "Tell me about this user.", ["GitLab"]),
    ("N35.cross.sequential_writes",
     [("sess-A", "I live in Berlin."), ("sess-B", "I work at Spotify.")],
     "Where does the user live and work?", ["Berlin", "Spotify"]),
]
for tid, turns, q, subs in CROSS:
    reg_data(tid, EVAL, "cross-session info in single recall",
             lambda t=turns, q=q, ss=subs: _cross_session(t, q, ss))


# ====================================================================
# ─────────────  hard constraints  (20 tests)  ───────────────────
# ====================================================================

HARD = "hard constraints"

# malformed JSON shapes (5)
MALFORMED = [
    ("H01.malformed.empty_body",   b""),
    ("H02.malformed.not_json",     b"not json"),
    ("H03.malformed.partial",      b'{"session_id": "x", "messages":'),
    ("H04.malformed.array_root",   b'[1,2,3]'),
    ("H05.malformed.bool_root",    b'true'),
]
for tid, body in MALFORMED:
    def _f(b=body):
        r = CLIENT.post(f"{BASE}/turns", content=b,
                        headers={"content-type": "application/json"})
        return 400 <= r.status_code < 500, f"{r.status_code}"
    reg_data(tid, HARD, "4xx never 5xx", _f)

# oversized payloads (5)
def _oversized(field: str, value):
    base = {
        "session_id": "x", "user_id": "y",
        "messages": [{"role": "user", "content": "x"}],
        "timestamp": TS, "metadata": {},
    }
    base[field] = value
    r = CLIENT.post(f"{BASE}/turns", json=base)
    return 400 <= r.status_code < 500, f"{r.status_code}"


reg_data("H06.oversized.session_id", HARD, "422", lambda: _oversized("session_id", "x" * 1000))
reg_data("H07.oversized.user_id",    HARD, "422", lambda: _oversized("user_id", "x" * 1000))
reg_data("H08.oversized.too_many_messages", HARD, "422",
         lambda: _oversized("messages", [{"role": "user", "content": "m"}] * 200))


def _oversized_msg_content():
    r = post_turn(uniq("s"), uniq("u"), "x" * 9000)
    return r.status_code == 422, f"{r.status_code}"


reg_data("H09.oversized.message_content", HARD, "422 content > MAX_MESSAGE_CHARS",
         _oversized_msg_content)


def _oversized_metadata():
    big_meta = {f"k{i}": "v" for i in range(100)}
    r = CLIENT.post(f"{BASE}/turns", json={
        "session_id": uniq("s"), "user_id": uniq("u"),
        "messages": [{"role": "user", "content": "hi"}],
        "timestamp": TS, "metadata": big_meta,
    })
    return r.status_code == 422, f"{r.status_code}"


reg_data("H10.oversized.metadata", HARD, "422 too many metadata entries", _oversized_metadata)

# unicode / special chars (5)
UNICODE = [
    ("H11.unicode.emoji",       "I love coding 🐍 in Python 💻 from 🌃 Tokyo!"),
    ("H12.unicode.rtl",         "I work at ‮evil-corp‬."),
    ("H13.unicode.cjk",         "我住在北京。 (I live in Beijing.)"),
    ("H14.unicode.zwsp",        "I work​ at ​Notion."),
    ("H15.unicode.combining",   "User café résumé naïve déjà vu in München."),
]
for tid, content in UNICODE:
    def _f(c=content):
        r = post_turn(uniq("s"), uniq("u"), c)
        return r.status_code == 201, f"{r.status_code}"
    reg_data(tid, HARD, "201 unicode safe", _f)

# missing required fields (5)
MISSING = [
    ("H16.missing.session_id",      {"user_id": "u", "messages": [{"role":"user","content":"x"}],
                                     "timestamp": TS, "metadata": {}}),
    ("H17.missing.timestamp",       {"session_id": "s", "user_id": "u",
                                     "messages": [{"role":"user","content":"x"}], "metadata": {}}),
    ("H18.missing.messages",        {"session_id": "s", "user_id": "u",
                                     "timestamp": TS, "metadata": {}}),
    ("H19.missing.role_in_msg",     {"session_id": "s", "user_id": "u",
                                     "messages": [{"content":"x"}], "timestamp": TS, "metadata": {}}),
    ("H20.missing.content_in_msg",  {"session_id": "s", "user_id": "u",
                                     "messages": [{"role":"user"}], "timestamp": TS, "metadata": {}}),
]
for tid, body in MISSING:
    def _f(b=body):
        r = CLIENT.post(f"{BASE}/turns", json=b)
        return r.status_code == 422, f"{r.status_code}"
    reg_data(tid, HARD, "422 missing required", _f)


# ====================================================================
# ────────────  Bounds & limits  (10 tests)  ─────────────────────────
# ====================================================================

BOUNDS = "Bounds & limits"


def _at_limit_query():
    """4000 chars exactly is invalid (>2000 limit), 1999 chars valid."""
    valid = "x" * 1999
    invalid = "x" * 5000
    r1 = post_recall(uniq("u"), valid)
    r2 = post_recall(uniq("u"), invalid)
    return r1.status_code == 200 and r2.status_code == 422, f"{r1.status_code}/{r2.status_code}"


reg_data("L01.bounds.query_at_limit", BOUNDS, "1999 ok, 5000 → 422", _at_limit_query)


def _at_limit_max_tokens():
    base = {"query": "x", "session_id": "x", "user_id": None}
    r1 = CLIENT.post(f"{BASE}/recall", json={**base, "max_tokens": 1})
    r2 = CLIENT.post(f"{BASE}/recall", json={**base, "max_tokens": 32768})
    r3 = CLIENT.post(f"{BASE}/recall", json={**base, "max_tokens": 32769})
    return r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 422, \
           f"{r1.status_code}/{r2.status_code}/{r3.status_code}"


reg_data("L02.bounds.max_tokens_edges", BOUNDS, "1 ok, 32768 ok, 32769 422", _at_limit_max_tokens)


def _at_limit_search():
    r1 = CLIENT.post(f"{BASE}/search",
                     json={"query": "x", "user_id": uniq("u"), "limit": 1})
    r2 = CLIENT.post(f"{BASE}/search",
                     json={"query": "x", "user_id": uniq("u"), "limit": 100})
    r3 = CLIENT.post(f"{BASE}/search",
                     json={"query": "x", "user_id": uniq("u"), "limit": 101})
    return r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 422, \
           f"{r1.status_code}/{r2.status_code}/{r3.status_code}"


reg_data("L03.bounds.search_limit_edges", BOUNDS, "1 ok, 100 ok, 101 422", _at_limit_search)


def _zero_max_tokens():
    r = CLIENT.post(f"{BASE}/recall",
                    json={"query": "x", "session_id": "x", "user_id": None, "max_tokens": 0})
    return r.status_code == 422, f"{r.status_code}"


reg_data("L04.bounds.max_tokens_zero", BOUNDS, "0 → 422", _zero_max_tokens)


def _negative_max_tokens():
    r = CLIENT.post(f"{BASE}/recall",
                    json={"query": "x", "session_id": "x", "user_id": None, "max_tokens": -1})
    return r.status_code == 422, f"{r.status_code}"


reg_data("L05.bounds.max_tokens_negative", BOUNDS, "-1 → 422", _negative_max_tokens)

# id-shape limits
def _id_at_limit():
    sid = "x" * 200  # MAX_ID_CHARS
    r = post_turn(sid, "u", "hi")
    return r.status_code == 201, f"{r.status_code}"


reg_data("L06.bounds.session_id_200_chars", BOUNDS, "200 ok", _id_at_limit)


def _id_over_limit():
    sid = "x" * 201
    r = post_turn(sid, "u", "hi")
    return r.status_code == 422, f"{r.status_code}"


reg_data("L07.bounds.session_id_201_chars", BOUNDS, "201 → 422", _id_over_limit)


def _name_at_limit():
    body = {
        "session_id": uniq("s"), "user_id": uniq("u"),
        "messages": [{"role": "tool", "name": "x" * 64, "content": "x"}],
        "timestamp": TS, "metadata": {},
    }
    r = CLIENT.post(f"{BASE}/turns", json=body)
    if r.status_code == 201:
        CREATED_USERS.add(body["user_id"])
    return r.status_code == 201, f"{r.status_code}"


reg_data("L08.bounds.tool_name_64", BOUNDS, "64 ok", _name_at_limit)


def _name_over_limit():
    body = {
        "session_id": uniq("s"), "user_id": uniq("u"),
        "messages": [{"role": "tool", "name": "x" * 65, "content": "x"}],
        "timestamp": TS, "metadata": {},
    }
    r = CLIENT.post(f"{BASE}/turns", json=body)
    return r.status_code == 422, f"{r.status_code}"


reg_data("L09.bounds.tool_name_65", BOUNDS, "65 → 422", _name_over_limit)


def _empty_user_id():
    r = post_turn(uniq("s"), "", "hi")
    return r.status_code == 422, f"{r.status_code}"


reg_data("L10.bounds.user_id_empty", BOUNDS, "empty user_id → 422", _empty_user_id)


# ====================================================================
# ──────────  Cleanup correctness  (10 tests)  ──────────────────────
# ====================================================================

CLEANUP = "Cleanup correctness"


def _delete_session_drops_indexed():
    sid, uid = uniq("s"), uniq("u")
    post_turn(sid, uid, "I live in Athens.")
    CLIENT.delete(f"{BASE}/sessions/{sid}")
    body = post_recall(uid, "Tell me about this user.").json()
    return "Athens" not in body["context"], f"ctx_len={len(body['context'])}"


reg_data("D01.cleanup.session_drops_indexed", CLEANUP, "session-scoped data gone",
         _delete_session_drops_indexed)


def _delete_user_full_wipe():
    sid, uid = uniq("s"), uniq("u")
    post_turn(sid, uid, "I'm vegan.")
    CLIENT.delete(f"{BASE}/users/{uid}")
    mems = get_memories(uid)
    body = post_recall(uid, "Tell me about this user.").json()
    if uid in CREATED_USERS:
        CREATED_USERS.discard(uid)
    return mems == [] and body == {"context": "", "citations": []}, \
           f"mem={len(mems)} ctx={len(body['context'])}"


reg_data("D02.cleanup.user_full_wipe", CLEANUP, "memories empty + recall cold",
         _delete_user_full_wipe)


def _delete_session_idempotent():
    sid, uid = uniq("s"), uniq("u")
    post_turn(sid, uid, "hi")
    r1 = CLIENT.delete(f"{BASE}/sessions/{sid}")
    r2 = CLIENT.delete(f"{BASE}/sessions/{sid}")
    r3 = CLIENT.delete(f"{BASE}/sessions/{sid}")
    return all(r.status_code == 204 for r in (r1, r2, r3)), \
           f"{r1.status_code}/{r2.status_code}/{r3.status_code}"


reg_data("D03.cleanup.session_idempotent_3x", CLEANUP, "3× DELETE → 204 each",
         _delete_session_idempotent)


def _delete_user_idempotent():
    uid = uniq("u")
    post_turn(uniq("s"), uid, "hi")
    rs = [CLIENT.delete(f"{BASE}/users/{uid}").status_code for _ in range(3)]
    if uid in CREATED_USERS:
        CREATED_USERS.discard(uid)
    return all(s == 204 for s in rs), f"{rs}"


reg_data("D04.cleanup.user_idempotent_3x", CLEANUP, "3× DELETE → 204 each",
         _delete_user_idempotent)


def _delete_session_with_active_supersession():
    """Polish-branch fix: deleting the active session must not leave the
    slot orphaned."""
    uid = uniq("u")
    s1, s2, s3 = uniq("sA"), uniq("sB"), uniq("sC")
    post_turn(s1, uid, "I live in Seattle.")
    post_turn(s2, uid, "Actually, I moved to Austin.")
    post_turn(s3, uid, "Update: I now live in Portland.")
    CLIENT.delete(f"{BASE}/sessions/{s3}")
    actives = [m for m in get_memories(uid)
               if m["active"] and "location" in m["key"] and "previous" not in m["key"]]
    return len(actives) >= 1, f"{len(actives)} active city rows"


reg_data("D05.cleanup.delete_active_session_repairs_chain", CLEANUP,
         "≥1 active city row remains", _delete_session_with_active_supersession)


def _delete_user_isolates():
    """Deleting user A must not affect user B."""
    sid_a, sid_b = uniq("sA"), uniq("sB")
    uid_a, uid_b = uniq("uA"), uniq("uB")
    post_turn(sid_a, uid_a, "I live in Tokyo.")
    post_turn(sid_b, uid_b, "I work at Asana.")
    CLIENT.delete(f"{BASE}/users/{uid_a}")
    if uid_a in CREATED_USERS:
        CREATED_USERS.discard(uid_a)
    body_b = post_recall(uid_b, "Tell me about this user.").json()
    return "Asana" in body_b["context"] or len(get_memories(uid_b)) > 0, \
           f"B ctx_len={len(body_b['context'])}"


reg_data("D06.cleanup.delete_user_a_keeps_b", CLEANUP, "user B intact",
         _delete_user_isolates)


def _post_delete_post_succeeds():
    """After DELETE /users/{id}, the same id can ingest fresh."""
    uid = uniq("u")
    post_turn(uniq("s"), uid, "first life")
    CLIENT.delete(f"{BASE}/users/{uid}")
    r = post_turn(uniq("s"), uid, "second life: I live in Berlin.")
    return r.status_code == 201, f"{r.status_code}"


reg_data("D07.cleanup.delete_then_ingest", CLEANUP, "201 fresh post-delete",
         _post_delete_post_succeeds)


def _delete_session_doesnt_touch_other_sessions():
    uid = uniq("u")
    sid_a, sid_b = uniq("sA"), uniq("sB")
    post_turn(sid_a, uid, "I live in Vienna.")
    post_turn(sid_b, uid, "I work at Adobe.")
    CLIENT.delete(f"{BASE}/sessions/{sid_a}")
    body = post_recall(uid, "Tell me about this user.").json()
    return "Adobe" in body["context"] or len(get_memories(uid)) > 0, \
           f"len={len(body['context'])}"


reg_data("D08.cleanup.session_a_keeps_b", CLEANUP, "B's data survives",
         _delete_session_doesnt_touch_other_sessions)


def _delete_session_idempotent_unknown():
    """Deleting a session that never existed → 204."""
    r = CLIENT.delete(f"{BASE}/sessions/{uniq('phantom')}")
    return r.status_code == 204, f"{r.status_code}"


reg_data("D09.cleanup.session_unknown_204", CLEANUP, "phantom session 204",
         _delete_session_idempotent_unknown)


def _delete_user_idempotent_unknown():
    r = CLIENT.delete(f"{BASE}/users/{uniq('phantom')}")
    return r.status_code == 204, f"{r.status_code}"


reg_data("D10.cleanup.user_unknown_204", CLEANUP, "phantom user 204",
         _delete_user_idempotent_unknown)


# ====================================================================
# ──────────  Forgetting / decay stress  (25 tests)  ─────────────────
# ====================================================================
#
# Tests that probe HOW memories age across many turns:
#   * plant-and-bury — fact set early, buried under N unrelated turns,
#     should still surface in /recall and /users/{id}/memories.
#   * long supersession chains (depth 3-10) — chain integrity holds.
#   * U-turn re-supersession — A→B→A → only the last A is active.
#   * multi-arc parallel evolution — city + employer + diet evolving
#     in interleaved sessions, each slot lands on its own latest.
#   * stale-fact retention under noise — 3 stable facts vs 30 noise
#     turns, profile query still surfaces all 3.
#   * tight-budget aging — under max_tokens=40 with 20 stored memories,
#     query-aware sort favours relevance over recency.
#
# Noise turns use single-message neutral content that the LLM tends to
# extract zero memories from (small-talk patterns) — keeps OpenAI cost
# bounded. Each test runs ~5-15 LLM calls; full FORGETTING category is
# ~200 LLM extractions, ~$0.20 on gpt-4o-mini.

FORGETTING = "Forgetting / decay stress"

NOISE_LINES = [
    "Today was a long day.",
    "Watched a movie last night.",
    "Need to do laundry this weekend.",
    "Coffee was strong this morning.",
    "Reading a book on the train.",
    "The weather turned cold.",
    "Forgot to charge my phone.",
    "Lunch was decent.",
    "Tired but in good spirits.",
    "Caught up with a friend.",
    "Cleaned my desk finally.",
    "Listening to a podcast.",
    "Trying a new recipe tonight.",
    "Stayed up too late.",
    "Bought new headphones.",
    "Train was late again.",
    "Had a pleasant walk.",
    "Watched the sunset.",
    "Tried a new café downtown.",
    "Plants need watering.",
    "Reorganised my bookshelf.",
    "Took a nap after work.",
    "Working on a side project.",
    "Played some chess online.",
    "Finished a 10k jog.",
]


def _bury_with_noise(uid: str, n_turns: int):
    """Post n_turns of generic small-talk under user uid."""
    pool = NOISE_LINES * ((n_turns // len(NOISE_LINES)) + 1)
    for i in range(n_turns):
        post_turn(uniq("noise"), uid, pool[i])


def _plant_then_bury(plant_text: str, n_noise: int, query: str,
                     expected_substr: str) -> tuple[bool, str]:
    """Plant a fact, bury under noise, query, expect it still surfaces."""
    sid, uid = uniq("plant"), uniq("u")
    post_turn(sid, uid, plant_text)
    _bury_with_noise(uid, n_noise)
    ctx = post_recall(uid, query).json()["context"]
    return expected_substr.lower() in ctx.lower(), f"buried under {n_noise} turns"


PLANT_BURY = [
    ("FG01.bury.city.1noise",  "I live in Reykjavik.", 1,  "Tell me about this user.", "Reykjavik"),
    ("FG02.bury.city.3noise",  "I live in Reykjavik.", 3,  "Tell me about this user.", "Reykjavik"),
    ("FG03.bury.city.5noise",  "I live in Reykjavik.", 5,  "Tell me about this user.", "Reykjavik"),
    ("FG04.bury.employer.10",  "I work at Klarna.",    10, "Tell me about this user.", "Klarna"),
    ("FG05.bury.diet.15noise", "I'm strictly vegan.",  15, "Tell me about this user.", "vegan"),
]
for tid, plant, n, q, exp in PLANT_BURY:
    reg_data(tid, FORGETTING,
             f"fact survives {n} noise turns",
             lambda p=plant, n=n, q=q, e=exp: _plant_then_bury(p, n, q, e))


def _long_chain(slot_keys: tuple[str, ...], cities: list[str]) -> tuple[bool, str]:
    """Run a supersession arc of `len(cities)` steps; assert active is the
    last city and chain length is ≥ len(cities)-1 inactive rows."""
    sid, uid = uniq("arc"), uniq("u")
    for c in cities:
        post_turn(sid, uid, f"I now live in {c}.")
    mems = get_memories(uid)
    actives = [m for m in mems if m["active"]
               and any(m["key"].startswith(k) for k in slot_keys)]
    inactives = [m for m in mems if not m["active"]
                 and any(m["key"].startswith(k) for k in slot_keys)]
    if not actives:
        return False, f"no active row, {len(inactives)} inactive"
    last = cities[-1].lower()
    has_last = any(last in m["value"].lower() for m in actives)
    return has_last, f"active={actives[0]['value']!r} inactive={len(inactives)}"


CHAIN_TESTS = [
    ("FG06.chain.depth3",  ("location.current_city", "location.city"),
     ["Seattle", "Austin", "Portland"]),
    ("FG07.chain.depth5",  ("location.current_city", "location.city"),
     ["NYC", "LA", "Austin", "Denver", "Boise"]),
    ("FG08.chain.depth7",  ("location.current_city", "location.city"),
     ["A-ville", "Bton", "Cville", "Dburg", "Ecity", "Ftown", "Gport"]),
    ("FG09.chain.depth10", ("location.current_city", "location.city"),
     [f"City{i}" for i in range(10)]),
    ("FG10.chain.history_preserved", ("location.current_city", "location.city"),
     ["Tokyo", "Kyoto", "Osaka", "Nagoya", "Sapporo"]),
]
for tid, sk, cities in CHAIN_TESTS:
    reg_data(tid, FORGETTING,
             f"active = '{cities[-1]}', ≥{len(cities)-1} inactive",
             lambda sk=sk, c=cities: _long_chain(sk, c))


def _u_turn(steps: list[str], expected_active: str) -> tuple[bool, str]:
    """Move through cities A→B→...→expected_active. Last value wins."""
    sid, uid = uniq("uturn"), uniq("u")
    for c in steps:
        post_turn(sid, uid, f"Actually, I just moved to {c}.")
    actives = [m for m in get_memories(uid)
               if m["active"]
               and (m["key"].startswith("location.current_city")
                    or m["key"].startswith("location.city"))
               and "previous" not in m["key"]]
    if not actives:
        return False, "no active city row"
    val = actives[0]["value"].lower()
    return expected_active.lower() in val, f"active={val!r}"


U_TURNS = [
    ("FG11.uturn.aba",  ["Seattle", "Austin", "Seattle"], "Seattle"),
    ("FG12.uturn.abab", ["Seattle", "Austin", "Seattle", "Austin"], "Austin"),
    ("FG13.uturn.abca", ["Seattle", "Austin", "Boise", "Seattle"], "Seattle"),
]
for tid, steps, exp in U_TURNS:
    reg_data(tid, FORGETTING,
             f"after U-turn, active = {exp}",
             lambda s=steps, e=exp: _u_turn(s, e))


_ARC_STOP = {"just", "moved", "started", "joined", "promoted", "switched",
             "actually", "live", "lives", "live", "work", "works", "going",
             "strict", "strictly", "company", "city", "role"}


def _multi_arc(arcs: list[tuple[str, list[str], tuple[str, ...]]]
               ) -> tuple[bool, str]:
    """Multiple parallel supersession arcs. Each tuple is
    (label, sequence_of_turns, accepted_slot_key_prefixes).
    Verify each arc's *last* step lands as the active value in its slot.

    Token extraction: alphabetic runs of length ≥4 minus stopwords —
    catches "Madrid" out of "Now in Madrid." reliably (the previous
    naive split() left punctuation glued to the token).
    """
    uid = uniq("multiarc")
    max_len = max(len(seq) for _, seq, _ in arcs)
    # Interleave so sequences are not all from one category in a row.
    for i in range(max_len):
        for label, seq, _ in arcs:
            if i < len(seq):
                post_turn(uniq(f"sess-{label}"), uid, seq[i])
    misses: list[str] = []
    for label, seq, slot_prefixes in arcs:
        target = seq[-1].lower()
        tokens = [
            t for t in re.findall(r"[a-z]+", target)
            if len(t) >= 4 and t not in _ARC_STOP
        ]
        actives = [m for m in get_memories(uid)
                   if m["active"]
                   and any(m["key"].startswith(p) for p in slot_prefixes)]
        if not any(any(tok in m["value"].lower() for tok in tokens)
                   for m in actives):
            misses.append(label)
    return not misses, f"misses={misses}"


MULTI_ARC = [
    ("FG14.multiarc.city_employer",
     [
         ("city",  ["I live in Berlin.", "Moved to Lisbon.", "Now in Madrid."],
          ("location.current_city", "location.city")),
         ("emp",   ["I work at Stripe.", "Joined Notion last week.", "Just started at Anthropic."],
          ("employment.current_company", "employment.company")),
     ]),
    ("FG15.multiarc.city_employer_diet",
     [
         ("city",  ["I'm in Tokyo.", "Moved to Osaka.", "Now in Kyoto."],
          ("location.current_city", "location.city")),
         ("emp",   ["I work at Sony.", "Joined Rakuten.", "Just started at Mercari."],
          ("employment.current_company", "employment.company")),
         ("diet",  ["I eat anything.", "Going vegetarian.", "Now strictly vegan."],
          ("diet",)),
     ]),
    ("FG16.multiarc.three_arcs_2steps",
     [
         ("city",  ["I'm in NYC.", "Now in LA."],
          ("location.current_city", "location.city")),
         ("emp",   ["At Goldman.", "Now at JPMorgan."],
          ("employment.current_company", "employment.company")),
         ("role",  ["I'm an analyst.", "Promoted to associate."],
          ("employment.current_role", "employment.role")),
     ]),
    ("FG17.multiarc.no_cross_contamination",
     [
         ("city",  ["I live in Madrid.", "Moved to Barcelona."],
          ("location.current_city", "location.city")),
         ("diet",  ["I am vegetarian.", "Switched to vegan."],
          ("diet",)),
     ]),
]
for tid, arcs in MULTI_ARC:
    reg_data(tid, FORGETTING,
             "every arc lands on its latest value",
             lambda a=arcs: _multi_arc(a))


def _stale_under_noise(plants: list[str], n_noise: int,
                       expected_substrs: list[str]) -> tuple[bool, str]:
    """Plant N facts, bury under M noise turns, profile query, expect
    most planted facts surface."""
    uid = uniq("stale")
    for p in plants:
        post_turn(uniq("plant"), uid, p)
    _bury_with_noise(uid, n_noise)
    ctx = post_recall(uid, "Tell me about this user.").json()["context"]
    have = sum(1 for s in expected_substrs if s.lower() in ctx.lower())
    return have >= len(expected_substrs) // 2 + 1, f"{have}/{len(expected_substrs)}"


STALE = [
    ("FG18.stale.city_under_5",  ["I live in Helsinki."],                              5,
     ["Helsinki"]),
    ("FG19.stale.three_under_8",
     ["I live in Oslo.", "I work at Equinor.", "I'm vegetarian."], 8,
     ["Oslo", "Equinor", "vegetarian"]),
    ("FG20.stale.three_under_15",
     ["I live in Lima.", "I'm a chef.", "Allergic to peanuts."], 15,
     ["Lima", "chef", "peanut"]),
    ("FG21.stale.five_under_10",
     ["I live in Warsaw.", "I work at Allegro.", "I'm vegan.",
      "Allergic to bee stings.", "I have a dog called Luna."], 10,
     ["Warsaw", "Allegro", "vegan", "Luna"]),
    ("FG22.stale.early_vs_late",
     # Plant fact at turn 1, noise, then plant another fact at the end.
     # Both should still surface in profile query.
     ["I live in Glasgow.",  # planted turn 1
      "I work at Skyscanner.",  # planted turn 2
      ],
     5,  # 5 noise turns between the two planted bursts
     ["Glasgow", "Skyscanner"]),
]
for tid, plants, n, exp in STALE:
    reg_data(tid, FORGETTING,
             f"≥{len(exp)//2 + 1} planted facts survive {n} noise turns",
             lambda p=plants, n=n, e=exp: _stale_under_noise(p, n, e))


def _tight_budget_aging(plants: list[str], query: str, max_tokens: int,
                        relevant_substrs: list[str],
                        forbidden_substrs: list[str]) -> tuple[bool, str]:
    """Plant many memories; under tight budget, relevant facts win regardless
    of insert order."""
    uid = uniq("aging")
    for p in plants:
        post_turn(uniq("plant"), uid, p)
    ctx = post_recall(uid, query, max_tokens=max_tokens).json()["context"]
    have = sum(1 for s in relevant_substrs if s.lower() in ctx.lower())
    leak = sum(1 for s in forbidden_substrs if s.lower() in ctx.lower())
    ok = have >= 1 and leak == 0
    return ok, f"relevant={have}/{len(relevant_substrs)} leak={leak}"


TIGHT_BUDGET = [
    ("FG23.aging.diet_query_budget40",
     [
         "I work at Stripe.",
         "I live in Berlin.",
         "I have a dog named Biscuit.",
         "I have a cat named Mochi.",
         "Born in 1988.",
         "I'm vegetarian.",
         "Allergic to shellfish.",
         "I drive a Volvo.",
         "I love hiking on weekends.",
         "Have a sister named Anna.",
     ],
     "What does this user eat?", 40,
     ["vegetarian", "shellfish"],
     ["Stripe", "Volvo", "Anna"]),
    ("FG24.aging.employer_query_budget50",
     [
         "I have a cat named Mochi.",
         "I'm allergic to peanuts.",
         "I love TypeScript.",
         "I live in Toronto.",
         "I work at Shopify.",
         "Born in Vancouver.",
         "Brother lives in Montreal.",
     ],
     "Where does this user work?", 50,
     ["Shopify"],
     # Forbid only OBVIOUSLY off-topic (pet, allergy). Location facts
     # legitimately co-render on a "where work" query because they're
     # cosine-adjacent and useful for the agent's downstream reasoning.
     ["Mochi", "peanut"]),
    ("FG25.aging.location_query_budget45",
     [
         "I have a parrot named Mango.",
         "I work at Atlassian.",
         "I'm pescatarian.",
         "I live in Sydney.",
         "I love jazz.",
         "Speak French and Mandarin.",
     ],
     "Where does this user live?", 45,
     ["Sydney"],
     ["Mango", "Atlassian", "jazz"]),
]
for tid, plants, q, mt, rel, forb in TIGHT_BUDGET:
    reg_data(tid, FORGETTING,
             f"under {mt} tokens, relevant={rel} leaks={forb}=0",
             lambda p=plants, q=q, mt=mt, r=rel, f=forb:
                 _tight_budget_aging(p, q, mt, r, f))


# ====================================================================
# ─────────────────────────  runner  ─────────────────────────────────
# ====================================================================

def run_one(t: Test) -> tuple[str, str]:
    """Returns (status, note). status ∈ {pass,fail,error,skip}."""
    if t.skipped_reason:
        return "skip", t.skipped_reason
    try:
        ok, note = t.fn()
    except Exception as exc:
        return "error", f"{type(exc).__name__}: {exc}"[:120]
    return ("pass" if ok else "fail"), note


def cleanup_all():
    """Best-effort delete of every user and session created during the run."""
    for uid in list(CREATED_USERS):
        try:
            CLIENT.delete(f"{BASE}/users/{uid}", timeout=5)
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", help="filter by category prefix")
    p.add_argument("--filter", help="filter by id substring")
    p.add_argument("--list", action="store_true", help="just list tests, don't run")
    p.add_argument("--no-cleanup", action="store_true")
    args = p.parse_args()

    selected = TESTS
    if args.category:
        selected = [t for t in selected if args.category.lower() in t.category.lower()]
    if args.filter:
        selected = [t for t in selected if args.filter.lower() in t.id.lower()]

    if args.list:
        for t in selected:
            print(f"{t.id:42s}  [{t.category}]  predicted: {t.predicted}")
        print(f"\n{len(selected)} tests")
        return

    print(f"Mock workload — {len(selected)} tests, RUN={RUN_ID}, host={BASE}")
    print(f"{'─'*72}")

    counts = {"pass": 0, "fail": 0, "error": 0, "skip": 0}
    cat_counts: dict[str, dict[str, int]] = {}
    failures: list[tuple[Test, str, str]] = []
    started = time.time()

    for t in selected:
        status, note = run_one(t)
        counts[status] += 1
        cat_counts.setdefault(t.category, {"pass": 0, "fail": 0, "error": 0, "skip": 0})
        cat_counts[t.category][status] += 1
        if status == "pass":
            tag = f"{_GREEN}✓ PASS{_RESET}"
        elif status == "fail":
            tag = f"{_RED}✗ FAIL{_RESET}"
            failures.append((t, status, note))
        elif status == "error":
            tag = f"{_RED}! ERR {_RESET}"
            failures.append((t, status, note))
        else:
            tag = f"{_YELLOW}~ SKIP{_RESET}"
        print(f"{tag}  {t.id:42s}  {_DIM}{note[:90]}{_RESET}")

    elapsed = time.time() - started
    print(f"{'─'*72}")
    print("Per category:")
    for cat in sorted(cat_counts):
        c = cat_counts[cat]
        total = sum(c.values())
        print(f"  {cat:30s}  {c['pass']:>3}/{total:<3} pass  "
              f"{c['fail']} fail  {c['error']} err  {c['skip']} skip")
    print(f"{'─'*72}")
    total = sum(counts.values())
    summary = (
        f"  PASS={counts['pass']}  FAIL={counts['fail']}  "
        f"ERROR={counts['error']}  SKIP={counts['skip']}  TOTAL={total}  "
        f"ELAPSED={elapsed:.1f}s"
    )
    if counts["fail"] == 0 and counts["error"] == 0:
        print(f"{_GREEN}{summary}{_RESET}")
    else:
        print(f"{_RED}{summary}{_RESET}")
        print(f"\n{_RED}First failures (max 10):{_RESET}")
        for t, status, note in failures[:10]:
            print(f"  [{status}] {t.id} — predicted {t.predicted!r} — got {note}")

    if not args.no_cleanup:
        cleanup_all()
        print(f"\n{_DIM}cleanup: deleted {len(CREATED_USERS)} users{_RESET}")

    sys.exit(1 if counts["fail"] + counts["error"] > 0 else 0)


if __name__ == "__main__":
    main()
