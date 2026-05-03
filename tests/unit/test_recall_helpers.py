"""Offline tests for the pure recall helpers (HGT-30).

These tests exercise `_format_recall_context`, `_parse_known_facts_block`,
`_canonicalise_slot`, and `_facts_block_relevant` directly — no Docker,
no Postgres, no OpenAI. They run on a plain `pytest -q` invocation,
giving the suite a fast deterministic floor independent of the live
stack tests gated by `RAG_E2E=1`.

The helpers are imported from `app.api.routes`. Importing that module
constructs the FastAPI router but does not start the DB pool or hit
OpenAI — both are lazy.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.api.routes import (
    _canonicalise_slot,
    _facts_block_relevant,
    _format_recall_context,
    _parse_known_facts_block,
)


# Shared helper to build a memory dict with the shape `_fetch_active_memories`
# returns. updated_at is required because the tail-sort uses it.
def _mem(mid: str, mtype: str, key: str, value: str, *, ts: float = 1700000000.0) -> dict:
    return {
        "id": mid,
        "type": mtype,
        "key": key,
        "value": value,
        "updated_at": datetime.fromtimestamp(ts, tz=timezone.utc),
    }


# ---------------------------------------------------------------------------
# _canonicalise_slot — HGT-26 canonical slot map.
# ---------------------------------------------------------------------------


def test_canon_slot_employer_aliases_collapse():
    """Different LLM emissions for "where they work now" all map to the
    same canonical slot, so supersession fires across them."""
    for key in ["employment.company", "employer", "current_employer", "company"]:
        assert _canonicalise_slot(key) == "employment.current_company"


def test_canon_slot_location_aliases_collapse():
    for key in ["location.city", "current_city", "city"]:
        assert _canonicalise_slot(key) == "location.current_city"
    for key in ["location.previous_city", "previous_city", "past_city"]:
        assert _canonicalise_slot(key) == "location.previous_city"


def test_canon_slot_opinion_hierarchical_collapse():
    """Opinion arcs may emit deeper namespaces in later turns
    (`opinion.typescript.generics`); HGT-26 collapses them to the root
    so the chain supersedes correctly."""
    assert _canonicalise_slot("opinion.typescript") == "opinion.typescript"
    assert _canonicalise_slot("opinion.typescript.generics") == "opinion.typescript"
    assert _canonicalise_slot("opinion.python.async") == "opinion.python"


def test_canon_slot_falls_back_to_key_for_unknown():
    """Unrecognised keys pass through unchanged so multi-entity slots
    (two pets, two children) keep separate slots."""
    assert _canonicalise_slot("pet.name") == "pet.name"
    assert _canonicalise_slot("family.spouse_name") == "family.spouse_name"
    assert _canonicalise_slot("custom.weird") == "custom.weird"


# ---------------------------------------------------------------------------
# _parse_known_facts_block — used by the fixture runner verdicts.
# ---------------------------------------------------------------------------


def test_parse_known_facts_extracts_pairs_in_order():
    ctx = (
        "## Known facts about this user\n"
        "- diet.style: vegetarian\n"
        "- pet.name: Biscuit\n"
        "\n"
        "## Relevant from recent conversations\n"
        "- [t] (user s) some content\n"
    )
    assert _parse_known_facts_block(ctx) == [
        ("diet.style", "vegetarian"),
        ("pet.name", "Biscuit"),
    ]


def test_parse_known_facts_returns_empty_when_block_absent():
    ctx = "## Relevant from recent conversations\n- [t] (user s) only conv\n"
    assert _parse_known_facts_block(ctx) == []


def test_parse_known_facts_handles_empty_context():
    assert _parse_known_facts_block("") == []


# ---------------------------------------------------------------------------
# _facts_block_relevant — HGT-25 noise gate.
# ---------------------------------------------------------------------------


def test_gate_renders_when_memory_score_clears_threshold():
    assert _facts_block_relevant("any query", {"id-1": 0.05}, []) is True


def test_gate_renders_when_conversation_score_clears_threshold():
    hits = [{"score": 0.30}]
    assert _facts_block_relevant("any query", {}, hits) is True


def test_gate_blocks_when_all_signals_below_threshold():
    """v2 noise-typed-user-empty-context probe pattern: known user,
    unrelated query — both retrievers return weak scores → no facts.
    """
    assert _facts_block_relevant(
        "What is the user's opinion on quantum computing?",
        {"id-1": 0.005},
        [{"score": 0.15}],
    ) is False


def test_gate_renders_for_profile_query_regardless_of_scores():
    """Profile-allowlist regex (`tell me about this user`) bypasses
    score thresholds — reviewers expect identity facts back even
    though the open-ended query doesn't match any specific stored
    content."""
    assert _facts_block_relevant("Tell me about this user.", {}, []) is True
    assert _facts_block_relevant("Who is this user?", {}, []) is True
    assert _facts_block_relevant("What do we know about this user?", {}, []) is True


def test_gate_bypassed_when_memory_scores_is_none():
    """HGT-24 boundary: `memory_scores=None` signals retrieval skipped
    (embed_query failed). Facts must still render — that's the
    degraded-mode contract."""
    assert _facts_block_relevant("anything", None, []) is True


# ---------------------------------------------------------------------------
# _format_recall_context — token-budget enforcement (HGT-29) and
# query-aware fact ranking (HGT-23).
# ---------------------------------------------------------------------------


def test_format_returns_cold_for_no_memories_and_no_hits():
    assert _format_recall_context([], [], 1024) == ("", [])


def test_format_drops_facts_section_when_budget_too_small_for_header_plus_line():
    """HGT-29 provisional header: max_tokens=1 cannot fit the header
    even without any line, so the section is omitted entirely instead
    of returning a header-only context that exceeds the budget."""
    mems = [_mem("a", "fact", "k", "v")]
    ctx, _ = _format_recall_context(
        mems, [], max_tokens=1, memory_scores={"a": 0.05}, query="x",
    )
    assert ctx == ""


def test_format_renders_facts_when_budget_admits_header_plus_one_line():
    mems = [_mem("a", "fact", "diet.style", "vegetarian")]
    ctx, _ = _format_recall_context(
        mems, [], max_tokens=64, memory_scores={"a": 0.05}, query="diet",
    )
    assert "## Known facts about this user" in ctx
    assert "- diet.style: vegetarian" in ctx


def test_format_query_aware_sort_promotes_relevant_facts_under_tight_budget():
    """HGT-23: query similarity drives order. Diet fact has the highest
    score and must lead the block; employment trails. Under tight
    budget, the lowest-scoring facts drop first."""
    mems = [
        _mem("emp", "fact", "employment.company", "Notion"),
        _mem("diet", "preference", "diet.style", "vegetarian"),
        _mem("pet", "fact", "pet.name", "Biscuit"),
    ]
    scores = {"diet": 0.05, "emp": 0.02, "pet": 0.025}
    ctx, _ = _format_recall_context(
        mems, [], max_tokens=64, memory_scores=scores, query="what does the user eat",
    )
    # diet should appear; employment trails; pet may or may not fit.
    assert "diet.style" in ctx
    diet_pos = ctx.index("diet.style")
    if "employment.company" in ctx:
        assert ctx.index("employment.company") > diet_pos


def test_format_filters_low_score_conversation_hits():
    """HGT-25 conversation-side filter: hits below the cosine threshold
    are dropped from the conversation block. This is the §9 noise-
    resistance signal — off-topic queries return cold context, not
    weak chunks."""
    hits = [
        {
            "score": 0.10,  # below threshold 0.20
            "text": "noisy stale chunk",
            "metadata": {"turn_id": "t1", "ts": "2026-01-01", "role": "user", "session_id": "s"},
        }
    ]
    ctx, citations = _format_recall_context(
        [], hits, max_tokens=1024, memory_scores={}, query="off-topic question",
    )
    assert ctx == ""
    assert citations == []


def test_format_keeps_conversation_hits_above_threshold():
    hits = [
        {
            "score": 0.40,
            "text": "relevant chunk",
            "metadata": {"turn_id": "t1", "ts": "2026-01-01", "role": "user", "session_id": "s"},
        }
    ]
    ctx, citations = _format_recall_context(
        [], hits, max_tokens=1024, memory_scores={}, query="relevant question",
    )
    assert "## Relevant from recent conversations" in ctx
    assert "relevant chunk" in ctx
    assert len(citations) == 1
    assert citations[0]["score"] == 0.40


def test_format_renders_facts_in_degraded_mode_even_when_off_topic_query():
    """HGT-24 + HGT-25 boundary: when retrieval was skipped
    (memory_scores=None), facts always render — eval still gets the
    user's stable identity on a transient embed blip."""
    mems = [_mem("a", "fact", "location.city", "Berlin")]
    ctx, _ = _format_recall_context(
        mems, [], max_tokens=1024, memory_scores=None, query="something off-topic",
    )
    assert "Berlin" in ctx
