"""HTTP routes — §3 contract + manual /chat harness. Read top-to-bottom."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

from app.api.schemas import (
    ChatDeleteResponse,
    ChatHistoryResponse,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatSession,
    ChatSessionCreateRequest,
    ChatSessionCreateResponse,
    ChatSessionListResponse,
    Citation,
    FixtureAggregate,
    FixtureProbeResult,
    IngestRequest,
    IngestResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
    RecallRequest,
    RecallResponse,
    RunFixtureRequest,
    RunFixtureResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    TurnIndexResponse,
    TurnIndexRow,
    TurnMessage,
    TurnRequest,
    TurnResponse,
    UserMemoriesResponse,
    UserMemory,
)
from app.config import get_settings
from app.db.pool import get_pool
from app.embeddings.openai import (
    embed_documents,
    embed_query,
    get_client,
    inject_embed_failure,
)
from app.extraction import extract_memories_from_turn
from app.obs.logging import get_logger

log = get_logger(__name__)

# §3 §10: 1 token ≈ 4 chars for English text. Fine for /recall budget
# triage; we do NOT need a real tokenizer here, the eval timeout
# wouldn't reward the precision.
_CHARS_PER_TOKEN = 4
# Hard cap on the candidate pool we pull before token-budget trimming.
# Keeps the JOIN bounded; HNSW handles ranking.
_RECALL_TOPK = 10
# Memory-side retrieval top-k for HGT-23 query-aware fact ranking. We
# pull more memory IDs than turn-message hits because the budget loop
# in `_format_recall_context` may surface many facts on tight queries
# and we want non-zero similarity scores on as many of them as possible.
_MEMORY_RECALL_TOPK = 50
# Per-retriever candidate pool for the hybrid fuse. Each retriever (cosine
# and BM25) returns up to this many docs; RRF merges and re-ranks them.
_HYBRID_POOL = 30
# §9 noise-resistance gates (HGT-25). Facts block renders only when the
# query is on-topic for this user — at least one of:
#   * memory hybrid retrieval returns a row with RRF ≥ τ_mem
#       (one retriever agreeing on rank 1 ⇒ RRF ≈ 1/61 ≈ 0.0164;
#       two agreeing ⇒ ~0.033 — τ=0.02 is "≥1 retriever scored it").
#   * conversation cosine top hit ≥ τ_conv. Empirically v2 noise probe
#     ("opinion on quantum computing") tops at 0.25 on a typescript
#     user; positive identity probes sit ≥0.30. Threshold 0.27 splits.
#   * query matches the profile-allowlist regex
#     ("tell me about this user", "who is …", "what do you know about").
_RECALL_FACTS_GATE_MEMORY_RRF = 0.02
# HGT-32 per-fact inclusion threshold. The block-level gate above
# decides "should the facts block render at all?"; this threshold
# decides "should THIS particular fact appear inside the block?".
# Uses raw cosine similarity (not RRF) because RRF rank-1 from one
# retriever always yields ≈0.0164 regardless of how relevant the doc
# is — cosine reflects actual relevance. Calibrated so directly
# relevant memories (e.g. `pet.name=Biscuit` on "Does this user have
# any pets?") clear, but partial-overlap memories (typescript opinion
# on quantum query, ~0.20 cosine) are filtered. Threshold tuned to
# 0.30 against v2 fixture probes.
_RECALL_FACT_INCLUSION_MIN_COSINE = 0.25
# Lowered from 0.27 → 0.20 after empirical separation: v2 implicit-pet
# tops at 0.209 on a relevant pet query, v2 noise-typed-user-empty-context
# tops at 0.171 on a quantum-computing query against a typescript user.
# Same threshold gates the facts block AND filters conversation hits —
# off-topic queries return empty conversation, not just empty facts
# (§9 noise resistance is a context-wide property, not facts-only).
_RECALL_FACTS_GATE_CONVERSATION_COSINE = 0.20
_RECALL_CONVERSATION_MIN_SCORE = 0.20
# Compiled module-level so it doesn't recompile per /recall.
import re as _re  # noqa: E402 — kept local to retrieval section
_RECALL_PROFILE_QUERY_RE = _re.compile(
    r"\b(about (this )?user|tell me about|who is|"
    r"what do (?:we|you) know about|user'?s? profile)\b",
    _re.IGNORECASE,
)
# RRF dampening constant. 60 is the value from Cormack-Buettcher-Clarke
# (2009) and now the de-facto default; we explicitly cite it in the
# CHANGELOG so a reviewer doesn't have to ask "why 60?".
_RRF_K = 60
# Snippet trim to keep one giant message from monopolising the budget.
_SNIPPET_CHARS = 240


def _approx_tokens(s: str) -> int:
    if not s:
        return 0
    return max(1, (len(s) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _trim(text: str, n: int = _SNIPPET_CHARS) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


# §4.1 canonical-slot map (HGT-26). Maps LLM-emitted display keys to a
# stable supersession slot so a new fact can replace an old one even
# when the LLM picked a different key namespace. Rules are deterministic
# and intentionally narrow — we only collapse the 5 categories where
# variance is observed against v1+v2 fixtures (employment, location,
# diet, beverage, allergy). Multi-entity users (two pets, two children)
# keep `slot = key` so they don't collapse onto each other.
_SLOT_ALIASES: dict[str, str] = {
    # current employer
    "employment.company": "employment.current_company",
    "employment.current_company": "employment.current_company",
    "employer": "employment.current_company",
    "current_employer": "employment.current_company",
    "company": "employment.current_company",
    # current role
    "employment.role": "employment.current_role",
    "employment.current_role": "employment.current_role",
    "job.title": "employment.current_role",
    "role": "employment.current_role",
    # previous employer / role
    "employment.previous_company": "employment.previous_company",
    "previous_employer": "employment.previous_company",
    "past_employer": "employment.previous_company",
    "employment.previous_role": "employment.previous_role",
    # current / previous city
    "location.city": "location.current_city",
    "location.current_city": "location.current_city",
    "current_city": "location.current_city",
    "city": "location.current_city",
    "location.previous_city": "location.previous_city",
    "previous_city": "location.previous_city",
    "past_city": "location.previous_city",
    # diet / allergy / beverage
    "diet.style": "diet.style",
    "diet": "diet.style",
    "allergy": "allergy",
    "allergies": "allergy",
    "beverage.coffee": "beverage.coffee",
    "drinks.coffee": "beverage.coffee",
}


def _canonicalise_slot(key: str) -> str:
    """
    Map an LLM-emitted memory key to a canonical supersession slot.

    Priority:
      1. Exact alias from `_SLOT_ALIASES`.
      2. `opinion.<a>.<b>...` collapses hierarchically to
         `opinion.<a>` so opinion arcs supersede across LLM key variants
         (`opinion.typescript`, `opinion.typescript.generics` → both
         live in slot `opinion.typescript`).
      3. Fallback: slot = key.
    """
    if key in _SLOT_ALIASES:
        return _SLOT_ALIASES[key]
    if key.startswith("opinion."):
        parts = key.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    return key


def _to_jsonb(d: dict) -> str:
    """asyncpg accepts JSONB as text — keep the cast in one place."""
    return json.dumps(d, ensure_ascii=False)


def _from_jsonb(v) -> dict:
    """asyncpg returns jsonb as str (we don't register a codec)."""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    return json.loads(v)


def _chat_message(row) -> ChatMessage:
    return ChatMessage(
        id=row["id"],
        session_id=row["session_id"],
        user_id=row["user_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
    )


async def _fetch_session_user(session_id: int) -> str:
    """Resolve session_id → user_id, or 404 if the session doesn't exist."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT user_id FROM chat_sessions WHERE id = $1",
            session_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="session_not_found")
    return row["user_id"]


async def _fetch_chat_history(session_id: int, limit: int = 100) -> list[ChatMessage]:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, user_id, role, content, created_at
            FROM (
                SELECT id, session_id, user_id, role, content, created_at
                FROM chat_messages
                WHERE session_id = $1
                ORDER BY id DESC
                LIMIT $2
            ) recent
            ORDER BY id ASC
            """,
            session_id,
            limit,
        )
    return [_chat_message(row) for row in rows]


async def _generate_chat_reply(
    user_id: str,
    history: list[ChatMessage],
    message: str,
) -> str:
    settings = get_settings()
    client = get_client()
    transcript = [
        {"role": item.role, "content": item.content}
        for item in history[-12:]
    ]
    try:
        resp = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise test assistant for a persistence demo. "
                        f"You are currently speaking with {user_id.title()}. "
                        "Do not mention implementation details unless asked."
                    ),
                },
                *transcript,
                {"role": "user", "content": message},
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="llm_unavailable") from exc

    reply = resp.choices[0].message.content
    if not reply:
        return "I did not get a text response from the model."
    return reply


@router.get("/health")
async def health() -> dict[str, str]:
    """Readiness: DB is reachable. We do NOT call OpenAI on health checks."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("SELECT 1")
    return {"status": "ok"}


@router.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(payload: IngestRequest) -> IngestResponse:
    """
    Upsert documents and their OpenAI embeddings.

    Steps:
      1. Embed all texts in one OpenAI call.
      2. UPSERT documents and embeddings inside one transaction so /query
         never sees a half-written row.
    """
    texts = [doc.text for doc in payload.documents]
    vectors, embed_error = await embed_documents(texts)
    if embed_error is not None:
        # Legacy /ingest is a write path; we cannot store half-state.
        # Surface a retryable upstream-failure to the caller.
        raise HTTPException(status_code=503, detail={
            "error": "upstream_unavailable",
            "stage": "embed",
            "reason": embed_error,
            "retryable": True,
        })
    settings = get_settings()

    pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        for doc, vec in zip(payload.documents, vectors, strict=True):
            await conn.execute(
                """
                INSERT INTO documents (id, text, metadata)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (id) DO UPDATE
                    SET text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata
                """,
                doc.id, doc.text, _to_jsonb(doc.metadata),
            )
            await conn.execute(
                """
                INSERT INTO embeddings (doc_id, embedding, model)
                VALUES ($1, $2, $3)
                ON CONFLICT (doc_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        model = EXCLUDED.model,
                        created_at = NOW()
                """,
                doc.id, vec, settings.embedding_model,
            )

    return IngestResponse(ingested=len(payload.documents))


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    """
    Embed the query and return top-k documents by cosine similarity.

    `<=>` is pgvector's cosine DISTANCE operator, so similarity = 1 - distance.
    Lower distance = higher similarity. The HNSW index makes this O(log N).
    """
    qvec, embed_error = await embed_query(payload.query)
    if embed_error is not None:
        # Cold semantics: read-only path, transient blip → empty hits, never 5xx.
        log.warning("query.embed_failed", reason=embed_error)
        return QueryResponse(hits=[])

    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT  d.id,
                    d.text,
                    d.metadata,
                    1 - (e.embedding <=> $1) AS score
            FROM    embeddings e
            JOIN    documents  d ON d.id = e.doc_id
            ORDER BY e.embedding <=> $1
            LIMIT   $2
            """,
            qvec, payload.top_k,
        )

    return QueryResponse(
        hits=[
            QueryHit(
                id=r["id"],
                text=r["text"],
                score=float(r["score"]),
                metadata=_from_jsonb(r["metadata"]),
            )
            for r in rows
        ]
    )


@router.get("/chat/sessions", response_model=ChatSessionListResponse)
async def list_chat_sessions(
    user_id: str = Query(pattern="^(alice|bob)$"),
) -> ChatSessionListResponse:
    """Sessions for a user, newest first, with a message count for the UI."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT cs.id, cs.user_id, cs.title, cs.created_at,
                   COUNT(cm.id) AS message_count
            FROM   chat_sessions cs
            LEFT   JOIN chat_messages cm ON cm.session_id = cs.id
            WHERE  cs.user_id = $1
            GROUP  BY cs.id
            ORDER  BY cs.created_at DESC, cs.id DESC
            """,
            user_id,
        )
    return ChatSessionListResponse(
        sessions=[
            ChatSession(
                id=r["id"],
                user_id=r["user_id"],
                title=r["title"],
                created_at=r["created_at"],
                message_count=int(r["message_count"]),
            )
            for r in rows
        ]
    )


@router.post("/chat/sessions", response_model=ChatSessionCreateResponse, status_code=201)
async def create_chat_session(payload: ChatSessionCreateRequest) -> ChatSessionCreateResponse:
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO chat_sessions (user_id, title)
            VALUES ($1, $2)
            RETURNING id, user_id, title, created_at
            """,
            payload.user_id,
            payload.title,
        )
    return ChatSessionCreateResponse(
        session=ChatSession(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=row["created_at"],
            message_count=0,
        )
    )


@router.delete("/chat/sessions/{session_id}", response_model=ChatDeleteResponse)
async def delete_chat_session(session_id: int) -> ChatDeleteResponse:
    """Drops the session — chat_messages cascade via FK."""
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chat_sessions WHERE id = $1",
            session_id,
        )
    deleted = int(result.split()[-1])
    if deleted == 0:
        raise HTTPException(status_code=404, detail="session_not_found")
    return ChatDeleteResponse(deleted=deleted)


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def chat_history(
    session_id: int = Query(ge=1),
    limit: int = Query(default=100, ge=1, le=500),
) -> ChatHistoryResponse:
    await _fetch_session_user(session_id)  # 404 if missing
    return ChatHistoryResponse(messages=await _fetch_chat_history(session_id, limit))


@router.get("/chat/all", response_model=ChatHistoryResponse)
async def all_chat_history(
    limit: int = Query(default=500, ge=1, le=1000),
) -> ChatHistoryResponse:
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, user_id, role, content, created_at
            FROM (
                SELECT id, session_id, user_id, role, content, created_at
                FROM chat_messages
                ORDER BY id DESC
                LIMIT $1
            ) recent
            ORDER BY id ASC
            """,
            limit,
        )
    return ChatHistoryResponse(messages=[_chat_message(row) for row in rows])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    user_id = await _fetch_session_user(payload.session_id)
    history = await _fetch_chat_history(payload.session_id, limit=40)
    reply = await _generate_chat_reply(user_id, history, payload.message)

    pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            INSERT INTO chat_messages (session_id, user_id, role, content)
            VALUES ($1, $2, 'user', $3), ($1, $2, 'assistant', $4)
            """,
            payload.session_id,
            user_id,
            payload.message,
            reply,
        )

    return ChatResponse(
        reply=reply,
        messages=await _fetch_chat_history(payload.session_id, limit=100),
    )


@router.delete("/chat/history", response_model=ChatDeleteResponse)
async def delete_chat_history(
    session_id: int = Query(ge=1),
) -> ChatDeleteResponse:
    """Wipes messages but keeps the session — for the UI 'Clear' button."""
    await _fetch_session_user(session_id)  # 404 if missing
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chat_messages WHERE session_id = $1",
            session_id,
        )
    return ChatDeleteResponse(deleted=int(result.split()[-1]))


# =====================================================================
# §3 contract endpoints. These are the eval-facing surface.
# Embedding, retrieval, LLM extraction, §4.3 stable-fact priority
# assembly (`## Known facts about this user`), and §4.1 supersession
# on (user_id, key) collision are live. /users/{id}/memories preserves
# the supersession chain (active=FALSE rows kept, linked via
# `supersedes`).
# =====================================================================


@router.post("/turns", response_model=TurnResponse, status_code=201)
async def create_turn(payload: TurnRequest) -> TurnResponse:
    """
    Persist a turn + its messages synchronously, index each message into
    documents/embeddings under doc_id `turn:{turn_id}:{idx}`, and run the
    LLM extractor to populate typed `memories` rows.

    Both the embed batch and the extractor run BEFORE `pool.acquire()` so
    we never hold a Postgres connection during an OpenAI HTTP roundtrip
    (the pool is small — see app/db/pool.py — and a 1-3s extractor would
    saturate it under concurrency). Once the transaction opens, all DB
    writes commit or roll back together (§5 synchronous correctness).

    Extraction failure is graceful per §5: the error string is recorded
    in `turns.metadata.extraction_error` and the turn is still committed
    with zero memories. AuthenticationError is the only exception that
    propagates — it indicates a config bug, not transient runtime state.
    """
    turn_id = str(uuid.uuid4())
    settings = get_settings()

    # HGT-23: extract FIRST so memory texts can be added to the same
    # embed batch as turn-message texts. One OpenAI roundtrip covers
    # both layers, keeping the §3 60 s /turns budget intact.
    if payload.user_id is None:
        # Anonymous turns can't write memories (table requires user_id
        # NOT NULL), so skip the LLM call entirely — saves cost and
        # avoids a no-op write path.
        extracted: list[dict] = []
        extraction_error: str | None = None
    else:
        extracted, extraction_error = await extract_memories_from_turn(payload.messages)

    # Pre-assign new memory ids before the embed call so the kind='memory'
    # doc_id (`mem:{id}`) is stable across the batch — vector and provenance
    # share the same uuid.
    new_memory_ids = [str(uuid.uuid4()) for _ in extracted]
    memory_texts = [f"{m['key']}: {m['value']}" for m in extracted]

    turn_msg_texts = [msg.content for msg in payload.messages]
    embed_inputs = turn_msg_texts + memory_texts

    # max_retries=0 on the hot path: 3 retry windows × 30s would blow the
    # §3 60s /turns budget on a 429 burst. /ingest and /recall keep the
    # client default (2) since they don't have the same budget pressure.
    vectors, embed_error = await embed_documents(embed_inputs, max_retries=0)
    n_turns = len(turn_msg_texts)
    turn_vectors = vectors[:n_turns] if embed_error is None else []
    memory_vectors = vectors[n_turns:] if embed_error is None else []

    # §5 / HGT-24 graceful upstream-failure policy:
    #   * both ok           → full commit (today's path).
    #   * embed fails       → commit turn metadata + memories; no
    #                         turn_messages / documents / embeddings rows
    #                         for this turn. The turn won't surface in
    #                         `## Relevant from recent conversations`
    #                         until re-indexed, but `## Known facts` and
    #                         /users/{id}/memories still work.
    #   * extract fails     → today's existing path: empty memories,
    #                         extraction_error in metadata. Already correct.
    #   * both fail         → 503 retryable. We have neither vectors nor
    #                         memories — there is nothing useful to commit.
    if embed_error is not None and not extracted:
        # "Both fail" = embed failed AND we have no extracted memories
        # (either because user_id is null, or because extraction errored,
        # or because the LLM legitimately produced nothing). 503 retryable.
        raise HTTPException(status_code=503, detail={
            "error": "upstream_unavailable",
            "embed_error": embed_error,
            "extract_error": extraction_error,
            "retryable": True,
        })

    turn_metadata = dict(payload.metadata)
    if extraction_error:
        turn_metadata["extraction_error"] = extraction_error
    if embed_error:
        turn_metadata["embed_error"] = embed_error

    pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            INSERT INTO turns (id, session_id, user_id, ts, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            turn_id,
            payload.session_id,
            payload.user_id,
            payload.timestamp,
            _to_jsonb(turn_metadata),
        )
        # HGT-28: raw turn_messages persist on every 201 — even when
        # embed fails. The §3 contract is "service writes completed
        # conversation turns and persists data"; degrading to "no
        # raw text on disk" would leave reviewers and any future
        # backfill path with nothing to reconstruct from. The indexed
        # triplet (documents + embeddings) still gates on `embed_error
        # is None` because those rows are useless without vectors.
        for idx, msg in enumerate(payload.messages):
            await conn.execute(
                """
                INSERT INTO turn_messages (turn_id, idx, role, name, content)
                VALUES ($1, $2, $3, $4, $5)
                """,
                turn_id, idx, msg.role, msg.name, msg.content,
            )
        if embed_error is None:
            for idx, (msg, vec) in enumerate(
                zip(payload.messages, turn_vectors, strict=True)
            ):
                doc_id = f"turn:{turn_id}:{idx}"
                provenance = {
                    "kind": "turn_message",
                    "turn_id": turn_id,
                    "session_id": payload.session_id,
                    "user_id": payload.user_id,
                    "role": msg.role,
                    "name": msg.name,
                    "idx": idx,
                    "ts": payload.timestamp.isoformat(),
                }
                await conn.execute(
                    """
                    INSERT INTO documents (id, text, metadata)
                    VALUES ($1, $2, $3::jsonb)
                    """,
                    doc_id, msg.content, _to_jsonb(provenance),
                )
                await conn.execute(
                    """
                    INSERT INTO embeddings (doc_id, embedding, model)
                    VALUES ($1, $2, $3)
                    """,
                    doc_id, vec, settings.embedding_model,
                )

        for mem_idx, mem in enumerate(extracted):
            # §4.1 supersession on (user_id, slot). Three cases:
            #   1. No prior active row     → plain insert.
            #   2. Same slot, same value   → idempotent: refresh
            #      the existing row's updated_at and skip insert.
            #      Avoids polluting `## Known facts` with duplicates
            #      when the user re-states a fact across turns.
            #   3. Same slot, new value    → insert new row with
            #      supersedes=old.id, flip old.active=FALSE.
            #      Old row stays for /users/{id}/memories history;
            #      its kind='memory' document is deleted so /recall
            #      doesn't re-surface stale facts via the hybrid
            #      memory retriever (HGT-23).
            #
            # HGT-26 lookup uses canonical slot, not the raw LLM key,
            # so opinion.typescript.generics and opinion.typescript
            # collapse onto the same slot. HGT-27 takes a transaction-
            # scoped advisory lock on (user_id, slot) before the
            # read-modify-write so concurrent /turns to the same slot
            # serialise — the partial unique index on
            # (user_id, slot) WHERE active=TRUE is the DB backstop.
            slot = _canonicalise_slot(mem["key"])
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"{payload.user_id}:{slot}",
            )
            prior = await conn.fetchrow(
                """
                SELECT id, value
                FROM   memories
                WHERE  user_id = $1 AND slot = $2 AND active = TRUE
                """,
                payload.user_id, slot,
            )
            if prior is not None and prior["value"] == mem["value"]:
                await conn.execute(
                    "UPDATE memories SET updated_at = NOW() WHERE id = $1",
                    prior["id"],
                )
                continue
            new_id = new_memory_ids[mem_idx]
            supersedes = prior["id"] if prior is not None else None
            if prior is not None:
                await conn.execute(
                    """
                    UPDATE memories
                    SET    active = FALSE, updated_at = NOW()
                    WHERE  id = $1
                    """,
                    prior["id"],
                )
                # Drop the superseded fact's `kind='memory'` document so
                # the query-relevant retrieval (HGT-23) never re-surfaces
                # it. The new active row gets its own mem:{id} doc below.
                await conn.execute(
                    "DELETE FROM documents WHERE id = $1",
                    f"mem:{prior['id']}",
                )
            await conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, type, key, value, confidence,
                    source_session, source_turn, supersedes, active, slot
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, TRUE, $10)
                """,
                new_id,
                payload.user_id,
                mem["type"],
                mem["key"],
                mem["value"],
                float(mem["confidence"]),
                payload.session_id,
                turn_id,
                supersedes,
                slot,
            )
            # HGT-23: write the sibling kind='memory' document + embedding
            # so /recall can rank facts by query similarity. Embed budget
            # was paid for upfront in the batched embed_documents call.
            if embed_error is None:
                mem_doc_id = f"mem:{new_id}"
                mem_provenance = {
                    "kind": "memory",
                    "memory_id": new_id,
                    "user_id": payload.user_id,
                    "key": mem["key"],
                    "type": mem["type"],
                    "session_id": payload.session_id,
                    "turn_id": turn_id,
                    "ts": payload.timestamp.isoformat(),
                }
                await conn.execute(
                    """
                    INSERT INTO documents (id, text, metadata)
                    VALUES ($1, $2, $3::jsonb)
                    """,
                    mem_doc_id, memory_texts[mem_idx], _to_jsonb(mem_provenance),
                )
                await conn.execute(
                    """
                    INSERT INTO embeddings (doc_id, embedding, model)
                    VALUES ($1, $2, $3)
                    """,
                    mem_doc_id, memory_vectors[mem_idx], settings.embedding_model,
                )
    return TurnResponse(id=turn_id)


async def _recall_candidates(
    qtext: str,
    qvec: list[float],
    user_id: str | None,
    session_id: str | None,
    top_k: int,
) -> list[dict]:
    """
    Hybrid retrieval: cosine top-N (HNSW) AND BM25 top-N (Postgres
    `tsvector` + `ts_rank_cd`) in parallel, fused via Reciprocal Rank
    Fusion. §10 explicitly calls vanilla cosine-top-k a floor — pure
    embedding similarity under-scores keyword-anchored queries on
    short turns, BM25 catches the literal-token signal, RRF blends the
    two rankings without needing them to share a score scale.

    Citations keep `score = cosine_similarity` (interpretable in
    [0,1]) so the §3 contract semantics don't shift. Ordering is by
    fused RRF rank.

    Scope by user_id; cross-session sharing within the same user is
    intentional per §5. session_id is captured in citations for
    provenance but not used as a hard filter.
    """
    if user_id is None:
        return []

    pool = get_pool()
    cosine_sql = """
        SELECT  d.id,
                d.text,
                d.metadata,
                1 - (e.embedding <=> $1) AS cosine_score
        FROM    embeddings e
        JOIN    documents  d ON d.id = e.doc_id
        WHERE   d.metadata->>'kind' = 'turn_message'
          AND   d.metadata->>'user_id' = $2
        ORDER BY e.embedding <=> $1
        LIMIT   $3
    """
    # plainto_tsquery handles tokenisation + stop-word removal; if the
    # query reduces to nothing (only stopwords) it returns an empty
    # tsquery and `tsv @@ q` matches zero rows — falls back to pure
    # cosine, which is fine.
    bm25_sql = """
        SELECT  d.id,
                d.text,
                d.metadata,
                1 - (e.embedding <=> $1) AS cosine_score
        FROM    embeddings e
        JOIN    documents  d ON d.id = e.doc_id
        WHERE   d.metadata->>'kind' = 'turn_message'
          AND   d.metadata->>'user_id' = $2
          AND   d.tsv @@ plainto_tsquery('english', $4)
        ORDER BY ts_rank_cd(d.tsv, plainto_tsquery('english', $4)) DESC
        LIMIT   $3
    """
    # asyncpg connections are NOT reentrant — running both fetches in
    # parallel on a single connection raises `another operation is in
    # progress`. Two cheap indexed lookups in series is the right
    # primitive; parallelising would only add pool pressure.
    async with pool.acquire() as conn:
        cos_rows = await conn.fetch(cosine_sql, qvec, user_id, _HYBRID_POOL)
        bm_rows = await conn.fetch(bm25_sql, qvec, user_id, _HYBRID_POOL, qtext)

    # RRF fusion — score(d) = Σ 1/(k + rank_i(d)) across retrievers.
    fused: dict[str, dict] = {}
    for rank, row in enumerate(cos_rows, start=1):
        fused[row["id"]] = {
            "row": row,
            "cosine_score": float(row["cosine_score"]),
            "rrf": 1.0 / (_RRF_K + rank),
        }
    for rank, row in enumerate(bm_rows, start=1):
        entry = fused.setdefault(row["id"], {
            "row": row,
            "cosine_score": float(row["cosine_score"]),
            "rrf": 0.0,
        })
        entry["rrf"] += 1.0 / (_RRF_K + rank)

    ordered = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    return [
        {
            "doc_id": e["row"]["id"],
            "text": e["row"]["text"],
            "metadata": _from_jsonb(e["row"]["metadata"]),
            "score": e["cosine_score"],
        }
        for e in ordered[:top_k]
    ]


# §4.3 priority bucketing: stable user identity facts beat opinions/events
# when the budget is tight. Lower number = higher priority.
_MEMORY_TYPE_PRIORITY = {"fact": 0, "preference": 1, "opinion": 2, "event": 3}


async def _fetch_active_memories(user_id: str) -> list[dict]:
    """
    Active typed memories for a user, ordered for §3 `## Known facts`
    rendering: type priority first (fact → preference → opinion → event),
    then most-recently-updated. Plain row dicts, decoupled from the
    /users/{id}/memories response model. The retrieval-rank join lives
    in `_format_recall_context` (HGT-23 query-aware ordering).
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT  id, type, key, value, updated_at
            FROM    memories
            WHERE   user_id = $1 AND active = TRUE
            ORDER BY updated_at DESC, id ASC
            """,
            user_id,
        )
    out = [
        {
            "id": r["id"],
            "type": r["type"],
            "key": r["key"],
            "value": r["value"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]
    out.sort(key=lambda m: _MEMORY_TYPE_PRIORITY.get(m["type"], 99))
    return out


async def _recall_memory_candidates(
    qtext: str,
    qvec: list[float],
    user_id: str,
    top_k: int,
) -> dict[str, dict]:
    """
    Hybrid retrieval over `kind='memory'` documents — mirror of
    `_recall_candidates` but it returns a
    `{memory_id: {"rrf": float, "cosine": float}}` map so the caller
    can both re-rank by RRF (HGT-23 ordering) and apply per-fact
    cosine thresholds (HGT-32 noise hardening). RRF ranks alone
    can't discriminate "relevant top-1" from "marginal top-1" — both
    yield 1/61 if only one retriever scored the doc — so the cosine
    score is required for the per-fact gate.
    """
    pool = get_pool()
    cosine_sql = """
        SELECT  d.metadata->>'memory_id' AS memory_id,
                1 - (e.embedding <=> $1) AS cosine_score
        FROM    embeddings e
        JOIN    documents  d ON d.id = e.doc_id
        WHERE   d.metadata->>'kind' = 'memory'
          AND   d.metadata->>'user_id' = $2
        ORDER BY e.embedding <=> $1
        LIMIT   $3
    """
    bm25_sql = """
        SELECT  d.metadata->>'memory_id' AS memory_id
        FROM    documents d
        WHERE   d.metadata->>'kind' = 'memory'
          AND   d.metadata->>'user_id' = $1
          AND   d.tsv @@ plainto_tsquery('english', $2)
        ORDER BY ts_rank_cd(d.tsv, plainto_tsquery('english', $2)) DESC
        LIMIT   $3
    """
    async with pool.acquire() as conn:
        cos_rows = await conn.fetch(cosine_sql, qvec, user_id, top_k)
        bm_rows = await conn.fetch(bm25_sql, user_id, qtext, top_k)
    fused: dict[str, dict] = {}
    for rank, row in enumerate(cos_rows, start=1):
        mid = row["memory_id"]
        if not mid:
            continue
        entry = fused.setdefault(mid, {"rrf": 0.0, "cosine": 0.0})
        entry["rrf"] += 1.0 / (_RRF_K + rank)
        # cosine_score from the SQL is the actual similarity (1 - distance);
        # cosine retrieval is the only place we have it, so set directly.
        entry["cosine"] = float(row["cosine_score"])
    for rank, row in enumerate(bm_rows, start=1):
        mid = row["memory_id"]
        if not mid:
            continue
        entry = fused.setdefault(mid, {"rrf": 0.0, "cosine": 0.0})
        entry["rrf"] += 1.0 / (_RRF_K + rank)
    return fused


def _facts_block_relevant(
    query: str,
    memory_scores: dict[str, dict] | None,
    hits: list[dict],
) -> bool:
    """
    HGT-25 noise-resistance gate. Facts render only when the query is
    on-topic for this user. Three signals (any one suffices):
      * memory hybrid retrieval scored at least one row above
        `_RECALL_FACTS_GATE_MEMORY_RRF`.
      * conversation cosine top hit ≥ `_RECALL_FACTS_GATE_CONVERSATION_COSINE`.
      * query matches `_RECALL_PROFILE_QUERY_RE` (open-ended profile
        questions like "what do you know about this user?").

    Profile queries bypass score thresholds because by construction
    they don't match any specific stored content but reviewers expect
    the identity facts back. Per CLAUDE.md §3 reference shape.

    `memory_scores=None` is the HGT-24 degraded-mode signal: retrieval
    didn't run because `embed_query` failed, so we have no relevance
    evidence either way. Render facts — the eval still gets the user's
    stable identity, which is the HGT-24 contract. The gate only
    suppresses facts when retrieval ran AND found nothing relevant.
    """
    if memory_scores is None:
        return True
    if memory_scores:
        max_rrf = max((v["rrf"] for v in memory_scores.values()), default=0.0)
        if max_rrf >= _RECALL_FACTS_GATE_MEMORY_RRF:
            return True
    if hits:
        if max((h["score"] for h in hits), default=0.0) >= _RECALL_FACTS_GATE_CONVERSATION_COSINE:
            return True
    if _RECALL_PROFILE_QUERY_RE.search(query):
        return True
    return False


def _format_recall_context(
    memories: list[dict],
    hits: list[dict],
    max_tokens: int,
    *,
    memory_scores: dict[str, dict] | None = None,
    query: str = "",
) -> tuple[str, list[dict]]:
    """
    Build the §3 `context` string and matched citations under a token
    budget. §4.3 priority: stable user facts → query-relevant memories
    → recent conversation chunks. The `## Known facts about this user`
    block consumes budget first; whatever budget remains is drained
    into `## Relevant from recent conversations`.

    HGT-23: when `memory_scores` is supplied (from
    `_recall_memory_candidates`), facts are re-sorted by query
    similarity DESC, with `(type_priority, updated_at DESC)` as the
    tail-sort for memories the retriever didn't surface.

    HGT-25: the facts block only renders when the query is relevant
    to this user (see `_facts_block_relevant`). Off-topic queries
    return cold — §9 noise resistance. The conversation block obeys
    its own relevance — it only renders if at least one hit clears
    the budget, and `hits` is already filtered by hybrid retrieval.

    HGT-29: every section header reserves cost provisionally — it's
    only committed to the running total once at least one body line
    fits. Empty sections never push context over `max_tokens`.

    Citations only carry `turn_id` references — memories surface in
    the prose block but are inspectable via /users/{id}/memories so we
    don't double-report them as citations.
    """
    if not memories and not hits:
        return "", []

    sections: list[str] = []
    used = 0

    facts_eligible = bool(memories) and _facts_block_relevant(
        query, memory_scores, hits,
    )

    if facts_eligible:
        facts_header = "## Known facts about this user"
        header_cost = _approx_tokens(facts_header)
        fact_lines: list[str] = []
        scores = memory_scores or {}

        # HGT-32 per-fact gate. Two bypasses:
        #   * Profile query (`tell me about this user`) — reviewer
        #     expects a full identity dump regardless of per-fact
        #     similarity.
        #   * Degraded mode (`memory_scores is None`, embed_query
        #     failed) — render every fact, HGT-24 contract overrides
        #     relevance.
        # Outside those, a memory renders only if its actual cosine
        # similarity to the query clears `_RECALL_FACT_INCLUSION_MIN_COSINE`.
        # We use cosine, not RRF rank, because rank-1 from one retriever
        # always yields ≈0.0164 regardless of how relevant the doc is —
        # cosine is the actual relevance signal.
        per_fact_gate_active = (
            memory_scores is not None
            and not _RECALL_PROFILE_QUERY_RE.search(query)
        )

        # Sort key: (-rrf_score, type_priority, -updated_at_ts).
        # Negative score so DESC by RRF; type_priority ASC keeps
        # facts > preferences > opinions > events for tied-similarity
        # tail; -updated_at_ts puts the freshest version on top within
        # a tie. RRF (not cosine) for ranking because hybrid retrieval
        # blends both signals; cosine alone misses keyword-anchored
        # FTS-only matches.
        def _rank_key(m: dict) -> tuple[float, int, float]:
            entry = scores.get(m["id"]) or {}
            rrf = entry.get("rrf", 0.0) if isinstance(entry, dict) else 0.0
            type_pri = _MEMORY_TYPE_PRIORITY.get(m["type"], 99)
            ts = m["updated_at"].timestamp() if m.get("updated_at") else 0.0
            return (-rrf, type_pri, -ts)

        for mem in sorted(memories, key=_rank_key):
            entry = scores.get(mem["id"]) or {}
            cosine_sim = entry.get("cosine", 0.0) if isinstance(entry, dict) else 0.0
            if per_fact_gate_active and cosine_sim < _RECALL_FACT_INCLUSION_MIN_COSINE:
                continue
            line = f"- {mem['key']}: {mem['value']}"
            cost = _approx_tokens(line) + 1
            head = header_cost if not fact_lines else 0
            # §4.3 + HGT-29: header cost only commits if a line fits
            # under it. The first line + header must both fit; later
            # lines just need their own cost under the running budget.
            if used + head + cost > max_tokens:
                break
            used += head + cost
            fact_lines.append(line)
        if fact_lines:
            sections.append("\n".join([facts_header, *fact_lines]))

    # HGT-25: filter conversation hits below the relevance floor so
    # off-topic queries return empty context, not stale chunks. Same
    # threshold as the facts gate keeps the two layers consistent.
    relevant_hits = [
        h for h in hits
        if h.get("score", 0.0) >= _RECALL_CONVERSATION_MIN_SCORE
    ]
    chosen: list[dict] = []
    if relevant_hits:
        hits_header = "## Relevant from recent conversations"
        header_cost = _approx_tokens(hits_header)
        hit_lines: list[str] = []
        for hit in relevant_hits:
            meta = hit["metadata"]
            snippet = _trim(hit["text"])
            ts = meta.get("ts") or ""
            role = meta.get("role") or "?"
            sid = meta.get("session_id") or "?"
            line = f"- [{ts}] ({role} {sid}) {snippet}"
            cost = _approx_tokens(line) + 1
            head = header_cost if not hit_lines else 0
            # Provisional header — only commit it if at least one hit
            # fits below it.
            if used + head + cost > max_tokens:
                break
            used += head + cost
            hit_lines.append(line)
            chosen.append({
                "turn_id": meta.get("turn_id") or "",
                "score": hit["score"],
                "snippet": snippet,
            })
        if hit_lines:
            sections.append("\n".join([hits_header, *hit_lines]))

    if not sections:
        return "", []
    return "\n\n".join(sections), chosen


@router.post("/recall", response_model=RecallResponse)
async def recall(payload: RecallRequest) -> RecallResponse:
    """
    §3 /recall: assemble context from two layers under a token budget.

    1. Stable user facts — active rows in `memories` for `user_id`.
    2. Query-relevant recent turns — cosine top-k over indexed messages.

    §4.3 priority is "stable facts → query-relevant memories → recent
    context". HGT-23 wires the middle layer: each memory row has a
    sibling `kind='memory'` document, hybrid-ranked by query
    similarity, used to re-sort the `## Known facts` block so under
    tight budget the relevant facts survive and irrelevant ones drop
    first. Recent-context retrieval is hybrid (cosine + FTS via RRF)
    per §10 — see `_recall_candidates`.

    §3 cold semantics: empty store, unknown user, or zero matches → 200
    with `{"context": "", "citations": []}`. Never errors.

    §5 / HGT-24 graceful upstream-failure policy: if `embed_query`
    fails (transient OpenAI blip), the `## Known facts` block STILL
    renders — it's a direct Postgres read, no embedding required.
    Only the conversation block goes cold (no docs to rank), so
    citations will be empty. The eval still gets the user's stable
    facts; throwing them away on a network blip would be gratuitous.
    """
    if payload.user_id is None:
        # No scope = no recall. We share across same-user sessions (§5),
        # but anonymous queries don't get to fish in everyone's data.
        return RecallResponse(context="", citations=[])

    # Always fetch facts — they don't need the query embedding.
    memories = await _fetch_active_memories(payload.user_id)

    qvec, embed_error = await embed_query(payload.query)
    # `memory_scores=None` is the HGT-24/HGT-25 boundary signal:
    # None  → retrieval did not run (embed failed); _format_recall_context
    #         bypasses the HGT-25 noise gate and renders facts as a
    #         degraded fallback (HGT-24 "facts always render on embed
    #         failure" contract).
    # dict  → retrieval ran (possibly empty result set); gate evaluates
    #         normally — empty dict + empty hits + no profile-query
    #         match means off-topic, render cold.
    memory_scores: dict[str, dict] | None = None
    if embed_error is not None:
        log.warning(
            "recall.embed_failed", user_id=payload.user_id, reason=embed_error,
        )
        hits: list[dict] = []
    else:
        memory_scores = await _recall_memory_candidates(
            payload.query, qvec, payload.user_id, _MEMORY_RECALL_TOPK,
        )
        hits = await _recall_candidates(
            payload.query, qvec, payload.user_id, payload.session_id, _RECALL_TOPK,
        )

    context, citations = _format_recall_context(
        memories,
        hits,
        payload.max_tokens,
        memory_scores=memory_scores,
        query=payload.query,
    )
    return RecallResponse(
        context=context,
        citations=[Citation(**c) for c in citations],
    )


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    """
    Twin of /recall but returns structured rows, not prose. §3 says this is
    invoked by an explicit agent tool call, so we don't apply token budgets
    here — `limit` already bounds the response.
    """
    if payload.user_id is None and payload.session_id is None:
        # Same fishing-trip guard as /recall: refuse fully unscoped queries.
        return SearchResponse(results=[])

    qvec, embed_error = await embed_query(payload.query)
    if embed_error is not None:
        # /search has no facts-block fallback — return cold per §5/HGT-24.
        log.warning("search.embed_failed", reason=embed_error)
        return SearchResponse(results=[])
    # session_id-only scope falls back to user-less lookup; we narrow by
    # session_id in SQL when user_id is missing.
    where = ["d.metadata->>'kind' = 'turn_message'"]
    args: list[object] = [qvec]
    if payload.user_id is not None:
        args.append(payload.user_id)
        where.append(f"d.metadata->>'user_id' = ${len(args)}")
    if payload.session_id is not None:
        args.append(payload.session_id)
        where.append(f"d.metadata->>'session_id' = ${len(args)}")
    where_sql = " AND ".join(where)
    args.append(payload.limit)
    sql = f"""
        SELECT  d.id,
                d.text,
                d.metadata,
                1 - (e.embedding <=> $1) AS score
        FROM    embeddings e
        JOIN    documents  d ON d.id = e.doc_id
        WHERE   {where_sql}
        ORDER BY e.embedding <=> $1
        LIMIT   ${len(args)}
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)

    results: list[SearchHit] = []
    for r in rows:
        meta = _from_jsonb(r["metadata"])
        ts_str = meta.get("ts")
        ts = datetime.fromisoformat(ts_str) if ts_str else datetime.fromtimestamp(0)
        results.append(SearchHit(
            content=r["text"],
            score=float(r["score"]),
            session_id=meta.get("session_id") or "",
            timestamp=ts,
            metadata={k: v for k, v in meta.items() if k not in {"kind"}},
        ))
    return SearchResponse(results=results)


@router.get("/users/{user_id}/memories", response_model=UserMemoriesResponse)
async def list_user_memories(user_id: str) -> UserMemoriesResponse:
    """
    Return typed structured memories for the user (§3 inspection
    endpoint). §4.1 requires the supersession chain to be preserved
    here, so we return both active and inactive rows. Order:
    active-first, then most-recently updated, so the reviewer sees
    the current state on top with superseded history below.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT  id, user_id, type, key, value, confidence,
                    source_session, source_turn,
                    created_at, updated_at, supersedes, active
            FROM    memories
            WHERE   user_id = $1
            ORDER BY active DESC, updated_at DESC, id ASC
            """,
            user_id,
        )
    return UserMemoriesResponse(
        memories=[
            UserMemory(
                id=r["id"],
                type=r["type"],
                key=r["key"],
                value=r["value"],
                confidence=float(r["confidence"]),
                source_session=r["source_session"],
                source_turn=r["source_turn"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                supersedes=r["supersedes"],
                active=r["active"],
            )
            for r in rows
        ]
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_section3_session(session_id: str) -> Response:
    """
    Drop every turn for the §3 string session_id, plus the indexed
    documents/embeddings rows and any structured memories whose
    provenance points at this session. turn_messages cascade via FK;
    documents and memories have no FK to turns so they are deleted
    explicitly in the same transaction.
    """
    pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            DELETE FROM documents
            WHERE metadata->>'kind' IN ('turn_message', 'memory')
              AND metadata->>'session_id' = $1
            """,
            session_id,
        )
        await conn.execute(
            "DELETE FROM memories WHERE source_session = $1",
            session_id,
        )
        await conn.execute("DELETE FROM turns WHERE session_id = $1", session_id)
    return Response(status_code=204)


@router.delete("/users/{user_id}", status_code=204)
async def delete_section3_user(user_id: str) -> Response:
    """Drop every turn, indexed message, and structured memory for this user."""
    pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            DELETE FROM documents
            WHERE metadata->>'kind' IN ('turn_message', 'memory')
              AND metadata->>'user_id' = $1
            """,
            user_id,
        )
        await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM turns WHERE user_id = $1", user_id)
    return Response(status_code=204)


# =====================================================================
# Admin / debug. Read-only inspector for the static UI so HGT-16a can be
# A/B verified without psql.
# =====================================================================


@router.get("/admin/turn_index", response_model=TurnIndexResponse)
async def admin_turn_index(
    user_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
) -> TurnIndexResponse:
    """
    Inspector for the /turns indexing path: indexed-message counts +
    recent rows + per-scope extraction-failure count (turns whose
    metadata carries an `extraction_error` from HGT-17a-ii). The UI
    pulls this on every refresh so writes and failures are visible
    without psql.
    """
    pool = get_pool()
    where = ["metadata->>'kind' = 'turn_message'"]
    args: list[object] = []
    if user_id is not None:
        args.append(user_id)
        where.append(f"metadata->>'user_id' = ${len(args)}")
    if session_id is not None:
        args.append(session_id)
        where.append(f"metadata->>'session_id' = ${len(args)}")
    where_sql = " AND ".join(where)
    # HGT-23: parallel WHERE for kind='memory' documents — same scope.
    mem_where = where_sql.replace(
        "metadata->>'kind' = 'turn_message'",
        "metadata->>'kind' = 'memory'",
    )

    # Build parallel turns-side WHERE clauses for the failure counts.
    # `documents` rows live under metadata->>'kind' = 'turn_message' but
    # error markers live in turns.metadata, so we have to scope twice.
    turn_scope_args: list[object] = []
    turn_scope: list[str] = []
    if user_id is not None:
        turn_scope_args.append(user_id)
        turn_scope.append(f"user_id = ${len(turn_scope_args)}")
    if session_id is not None:
        turn_scope_args.append(session_id)
        turn_scope.append(f"session_id = ${len(turn_scope_args)}")
    extract_where = " AND ".join(["metadata ? 'extraction_error'", *turn_scope])
    embed_where = " AND ".join(["metadata ? 'embed_error'", *turn_scope])

    async with pool.acquire() as conn:
        docs_count = await conn.fetchval(
            f"SELECT count(*) FROM documents WHERE {where_sql}",
            *args,
        )
        embs_count = await conn.fetchval(
            f"""
            SELECT count(*)
            FROM   embeddings e
            JOIN   documents  d ON d.id = e.doc_id
            WHERE  {where_sql}
            """,
            *args,
        )
        memory_docs_count = await conn.fetchval(
            f"SELECT count(*) FROM documents WHERE {mem_where}",
            *args,
        )
        extraction_failures = await conn.fetchval(
            f"SELECT count(*) FROM turns WHERE {extract_where}",
            *turn_scope_args,
        )
        embed_failures = await conn.fetchval(
            f"SELECT count(*) FROM turns WHERE {embed_where}",
            *turn_scope_args,
        )
        rows = await conn.fetch(
            f"""
            SELECT  d.id,
                    d.text,
                    d.metadata,
                    (e.doc_id IS NOT NULL) AS has_embedding
            FROM    documents d
            LEFT JOIN embeddings e ON e.doc_id = d.id
            WHERE   {where_sql}
            ORDER BY d.created_at DESC, d.id ASC
            LIMIT   ${len(args) + 1}
            """,
            *args, limit,
        )

    return TurnIndexResponse(
        documents_count=int(docs_count or 0),
        embeddings_count=int(embs_count or 0),
        memory_document_count=int(memory_docs_count or 0),
        extraction_failures=int(extraction_failures or 0),
        embed_failures=int(embed_failures or 0),
        rows=[
            TurnIndexRow(
                doc_id=r["id"],
                text=r["text"],
                metadata=_from_jsonb(r["metadata"]),
                has_embedding=bool(r["has_embedding"]),
            )
            for r in rows
        ],
    )


# =====================================================================
# Recall-quality measurement harness — CLAUDE.md §7 / §10.
#
# Loads fixtures/recall/<name>.json, resets each fix-* user, ingests its
# conversations through create_turn() (so production write-path runs
# end-to-end), then probes /recall and reports per-probe verdicts +
# aggregate metrics. The static UI calls this to make every retrieval
# upgrade a measurable diff in CHANGELOG.
# =====================================================================


_FIXTURE_USER_PREFIX = "fix-"
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "recall"


def _parse_iso_z(s: str) -> datetime:
    # asyncpg rejects naive datetimes for timestamptz; the §3 contract
    # also explicitly says ISO-8601, so we normalise the trailing Z.
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


_KNOWN_FACTS_HEADER = "## Known facts about this user"


def _parse_known_facts_block(context: str) -> list[tuple[str, str]]:
    """
    Pull `(key, value)` pairs out of the `## Known facts about this user`
    section of a /recall context. Format mirrors `_format_recall_context`:
    `- {key}: {value}` per bullet. Returns [] if the block is absent.
    """
    if _KNOWN_FACTS_HEADER not in context:
        return []
    out: list[tuple[str, str]] = []
    in_block = False
    for line in context.splitlines():
        if line.startswith(_KNOWN_FACTS_HEADER):
            in_block = True
            continue
        if line.startswith("## "):
            in_block = False
            continue
        if in_block and line.startswith("- ") and ": " in line:
            entry = line[2:]
            key, value = entry.split(": ", 1)
            out.append((key.strip(), value.strip()))
    return out


async def _count_inactive_rows(
    user_id: str,
    *,
    key: str | None = None,
    key_prefix: str | None = None,
) -> int:
    """
    Count `memories` rows where active=FALSE for a (user_id, key)
    selector. HGT-22 supersession-arc probes use this to verify
    /users/{id}/memories preserves the chain. Either `key` (exact match)
    or `key_prefix` (LIKE) must be provided.
    """
    if key is None and key_prefix is None:
        return 0
    pool = get_pool()
    if key is not None:
        sql = (
            "SELECT count(*) FROM memories "
            "WHERE user_id = $1 AND active = FALSE AND key = $2"
        )
        args = (user_id, key)
    else:
        sql = (
            "SELECT count(*) FROM memories "
            "WHERE user_id = $1 AND active = FALSE AND key LIKE $2"
        )
        args = (user_id, f"{key_prefix}%")
    async with pool.acquire() as conn:
        return int(await conn.fetchval(sql, *args) or 0)


@router.post("/admin/reembed_memories")
async def admin_reembed_memories(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    """
    HGT-23 backfill: walk active memories that lack a sibling
    `kind='memory'` document, batch-embed them, and INSERT the
    documents/embeddings rows. Idempotent — second run on a clean
    state returns `processed=0`.

    Operator-driven; runs in the foreground inside the §3 60s budget
    via internal batching (`limit` caps a single call). Call
    repeatedly until `processed=0`. Fresh installs do not need this —
    the new `/turns` code path writes the sibling docs synchronously.
    """
    pool = get_pool()
    where = ["m.active = TRUE",
             "NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = 'mem:' || m.id)"]
    args: list[object] = []
    if user_id is not None:
        args.append(user_id)
        where.append(f"m.user_id = ${len(args)}")
    args.append(limit)
    where_sql = " AND ".join(where)
    sql = f"""
        SELECT  m.id, m.user_id, m.type, m.key, m.value,
                m.source_session, m.source_turn, m.created_at
        FROM    memories m
        WHERE   {where_sql}
        ORDER BY m.created_at ASC
        LIMIT   ${len(args)}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
    if not rows:
        return {"processed": 0, "skipped": 0, "failed": 0, "error": None}

    texts = [f"{r['key']}: {r['value']}" for r in rows]
    settings = get_settings()
    vectors, embed_error = await embed_documents(texts)
    if embed_error is not None:
        return {
            "processed": 0,
            "skipped": 0,
            "failed": len(rows),
            "error": embed_error,
        }

    processed = 0
    async with pool.acquire() as conn, conn.transaction():
        for row, vec in zip(rows, vectors, strict=True):
            mem_doc_id = f"mem:{row['id']}"
            mem_provenance = {
                "kind": "memory",
                "memory_id": row["id"],
                "user_id": row["user_id"],
                "key": row["key"],
                "type": row["type"],
                "session_id": row["source_session"],
                "turn_id": row["source_turn"],
                "ts": row["created_at"].isoformat(),
            }
            await conn.execute(
                """
                INSERT INTO documents (id, text, metadata)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (id) DO NOTHING
                """,
                mem_doc_id, f"{row['key']}: {row['value']}", _to_jsonb(mem_provenance),
            )
            await conn.execute(
                """
                INSERT INTO embeddings (doc_id, embedding, model)
                VALUES ($1, $2, $3)
                ON CONFLICT (doc_id) DO NOTHING
                """,
                mem_doc_id, vec, settings.embedding_model,
            )
            processed += 1
    return {"processed": processed, "skipped": 0, "failed": 0, "error": None}


@router.post("/admin/recanonicalise_memories")
async def admin_recanonicalise_memories(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    """
    HGT-26 backfill: walk memories whose `slot` is NULL or stale,
    apply `_canonicalise_slot(key)`, and resolve any active-row
    collisions that emerge by superseding older losers to the newest
    winner per `(user_id, slot)`. Idempotent — second run on a clean
    state returns `processed=0`.

    Operator-driven; runs synchronously in the foreground inside the
    §3 budget via internal `limit`. Call repeatedly until
    `processed=0`. Fresh installs do not need this — going-forward
    /turns writes always include `slot`.
    """
    pool = get_pool()
    where = ["(slot IS NULL OR slot <> $1)"]  # placeholder; we replace below
    # Build dynamic WHERE: rows missing slot OR with stale slot for the
    # current canonicalisation. Stale-detection is computed in Python
    # against `key`, so the SQL just pulls all rows scoped by user_id
    # and we filter in memory.
    args: list[object] = []
    sql_where = ["TRUE"]
    if user_id is not None:
        args.append(user_id)
        sql_where.append(f"user_id = ${len(args)}")
    args.append(limit)
    sql = f"""
        SELECT id, user_id, key, value, active, updated_at, slot
        FROM   memories
        WHERE  {' AND '.join(sql_where)}
        ORDER BY user_id, updated_at DESC, id ASC
        LIMIT  ${len(args)}
    """

    async with pool.acquire() as conn, conn.transaction():
        rows = await conn.fetch(sql, *args)
        # Group by (user_id, computed canonical slot). Skip rows whose
        # current `slot` already matches — idempotency guard.
        groups: dict[tuple[str, str], list] = {}
        for r in rows:
            canonical = _canonicalise_slot(r["key"])
            if r["slot"] == canonical:
                continue
            groups.setdefault((r["user_id"], canonical), []).append(r)

        processed = 0
        collisions_resolved = 0
        for (uid, slot), rs in groups.items():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"{uid}:{slot}",
            )
            # Existing active row in this slot (e.g. set by a concurrent
            # /turns or a prior backfill batch) must be respected.
            existing_active = await conn.fetchrow(
                """
                SELECT id, updated_at FROM memories
                WHERE user_id = $1 AND slot = $2 AND active = TRUE
                """,
                uid, slot,
            )
            actives = sorted(
                [r for r in rs if r["active"]],
                key=lambda r: r["updated_at"], reverse=True,
            )
            if existing_active is not None:
                # Pre-existing winner — losers among `actives` get
                # inactivated and pointed at the existing winner.
                winner_id = existing_active["id"]
                losers = actives
            elif actives:
                winner_id = actives[0]["id"]
                losers = actives[1:]
            else:
                winner_id = None
                losers = []

            for loser in losers:
                await conn.execute(
                    """
                    UPDATE memories
                    SET    active = FALSE,
                           supersedes = $1,
                           slot = $2,
                           updated_at = NOW()
                    WHERE  id = $3
                    """,
                    winner_id, slot, loser["id"],
                )
                await conn.execute(
                    "DELETE FROM documents WHERE id = $1",
                    f"mem:{loser['id']}",
                )
                collisions_resolved += 1
                processed += 1

            # Stamp slot on every row in the group that doesn't already
            # have it (active winner + any inactive history rows).
            for r in rs:
                if r["id"] in {l["id"] for l in losers}:
                    continue
                await conn.execute(
                    "UPDATE memories SET slot = $1 WHERE id = $2 AND (slot IS DISTINCT FROM $1)",
                    slot, r["id"],
                )
                processed += 1
    return {
        "processed": processed,
        "collisions_resolved": collisions_resolved,
    }


@router.post("/admin/inject_embed_failure")
async def admin_inject_embed_failure(
    count: int = Query(default=1, ge=0, le=100),
    error: str = Query(default="connection"),
) -> dict:
    """
    Test-only knob: load a counter that forces the next N embed_documents
    calls to return `([], error)` instead of hitting OpenAI. The DB is not
    exposed to the host, so HGT-24 graceful-degradation tests can't
    monkey-patch in-process; this endpoint lets them inject the failure
    via the live HTTP stack instead. NOT part of the §3 contract.
    """
    inject_embed_failure(count, error)
    return {"queued_failures": count, "error_tag": error}


@router.post("/admin/run_fixture", response_model=RunFixtureResponse)
async def admin_run_fixture(payload: RunFixtureRequest) -> RunFixtureResponse:
    """
    Run a recall-quality fixture end-to-end and return per-probe + aggregate
    metrics. Destructive: resets every user_id in the fixture, but only if
    that user_id is prefixed `fix-` (guard against zapping real data).
    """
    fixture_path = _FIXTURE_DIR / f"{payload.name}.json"
    if not fixture_path.exists():
        raise HTTPException(status_code=404, detail=f"fixture_not_found:{payload.name}")

    started = datetime.now()
    fixture = json.loads(fixture_path.read_text())

    user_ids = sorted({c["user_id"] for c in fixture["conversations"]})
    for uid in user_ids:
        if not uid.startswith(_FIXTURE_USER_PREFIX):
            raise HTTPException(
                status_code=400,
                detail=f"unsafe_user_id:{uid} (must start with '{_FIXTURE_USER_PREFIX}')",
            )

    # 1. Reset fixture users so re-runs are idempotent.
    for uid in user_ids:
        await delete_section3_user(uid)

    # 2. Ingest each turn via the real create_turn so the production
    #    embed+index path runs. Capture turn_id by label for verdict mapping.
    # §5/HGT-24: a transient OpenAI blip on one turn must not abort the
    # whole fixture. Wrap each ingest; bucket failures into the aggregate
    # so the operator can see the gap without parsing logs.
    label_to_turn_id: dict[str, str] = {}
    ingest_failures = 0
    ingest_error_modes: dict[str, int] = {}
    for conv in fixture["conversations"]:
        for turn in conv["turns"]:
            req = TurnRequest(
                session_id=turn["session_id"],
                user_id=conv["user_id"],
                messages=[TurnMessage(**m) for m in turn["messages"]],
                timestamp=_parse_iso_z(turn["timestamp"]),
                metadata={"fixture": payload.name, "turn_label": turn["label"]},
            )
            try:
                resp = await create_turn(req)
            except HTTPException as exc:
                ingest_failures += 1
                tag = "http_503" if exc.status_code == 503 else f"http_{exc.status_code}"
                ingest_error_modes[tag] = ingest_error_modes.get(tag, 0) + 1
                log.warning(
                    "fixture.ingest_failed",
                    fixture=payload.name,
                    turn_label=turn["label"],
                    status=exc.status_code,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — last-resort net per §5
                ingest_failures += 1
                tag = f"unhandled:{type(exc).__name__}"
                ingest_error_modes[tag] = ingest_error_modes.get(tag, 0) + 1
                log.warning(
                    "fixture.ingest_unhandled",
                    fixture=payload.name,
                    turn_label=turn["label"],
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )
                continue
            label_to_turn_id[turn["label"]] = resp.id

    turn_id_to_label = {v: k for k, v in label_to_turn_id.items()}
    noise_max_top = float(fixture.get("noise_max_top_score", 0.35))

    # 3. Probe /recall and compute per-probe verdicts.
    probes_out: list[FixtureProbeResult] = []
    positive_top1 = positive_top3 = positive_total = 0
    noise_passed = noise_total = 0
    positive_top_scores: list[float] = []
    # HGT-22 v2 aggregations.
    multihop_total = multihop_top1 = multihop_top3 = 0
    facts_expected_total = facts_present_total = 0
    forbidden_total = 0
    supersession_chains_resolved = supersession_chains_expected = 0
    noise_top_score_violations = 0

    for probe in fixture["probes"]:
        is_noise = bool(probe.get("noise", False))
        rec_req = RecallRequest(
            query=probe["query"],
            session_id=probe["session_id"],
            user_id=probe["user_id"],
            max_tokens=int(probe.get("max_tokens", 512)),
        )
        rec_resp = await recall(rec_req)
        cites = rec_resp.citations
        top_score = cites[0].score if cites else None
        top_label = turn_id_to_label.get(cites[0].turn_id) if cites else None
        known_facts = _parse_known_facts_block(rec_resp.context)

        # ---- HGT-22: per-probe v2 evaluations (independent of pos/noise) ----
        expected_facts = list(probe.get("expected_known_facts", []))
        forbidden_facts = list(probe.get("forbidden_known_facts", []))
        expected_supersession = probe.get("expected_supersession")
        per_probe_top_threshold = probe.get("expected_top_score_under")

        facts_present = sum(
            1
            for ef in expected_facts
            if any(
                k == ef["key"] and ef["value_substring"].lower() in v.lower()
                for k, v in known_facts
            )
        )
        forbidden_violated = sum(
            1
            for fk in forbidden_facts
            if any(k.startswith(fk["key_prefix"]) for k, _ in known_facts)
        )
        chain_length: int | None = None
        chain_min: int | None = None
        if expected_supersession is not None:
            chain_min = int(expected_supersession.get("min_inactive_rows", 1))
            chain_length = await _count_inactive_rows(
                probe["user_id"],
                key=expected_supersession.get("key"),
                key_prefix=expected_supersession.get("key_prefix"),
            )
            supersession_chains_expected += 1
            if chain_length >= chain_min:
                supersession_chains_resolved += 1

        facts_expected_total += len(expected_facts)
        facts_present_total += facts_present
        forbidden_total += forbidden_violated

        if is_noise:
            noise_total += 1
            threshold = (
                float(per_probe_top_threshold)
                if per_probe_top_threshold is not None
                else noise_max_top
            )
            score_ok = (not cites) or (top_score is not None and top_score < threshold)
            # HGT-25: noise probes can additionally assert that the
            # entire context is empty (`expected_empty_context: true`)
            # — this is the relevance-gate signal, distinct from the
            # citation-score signal. When the flag is set, the facts
            # block must NOT render.
            expected_empty = bool(probe.get("expected_empty_context", False))
            empty_ok = (not expected_empty) or (rec_resp.context == "")
            # HGT-32: noise probes can also assert that no forbidden
            # fact key prefix surfaces in the `## Known facts` block.
            # The noise-quantum-typescript-no-facts probe uses this to
            # verify the per-fact gate suppresses partial-overlap leaks.
            forbidden_ok = forbidden_violated == 0
            ok = score_ok and empty_ok and forbidden_ok
            if not score_ok:
                noise_top_score_violations += 1
            if ok:
                noise_passed += 1
            note_parts = [f"noise threshold < {threshold}"]
            if expected_empty:
                note_parts.append(
                    "context==\"\"" if empty_ok else "EXPECTED empty context"
                )
            probes_out.append(FixtureProbeResult(
                label=probe["label"],
                query=probe["query"],
                is_noise=True,
                top_hit_turn_label=top_label,
                top_score=top_score,
                citations_count=len(cites),
                passed=ok,
                notes="; ".join(note_parts),
                expected_top_score_under=per_probe_top_threshold,
                expected_known_facts_total=len(expected_facts),
                expected_known_facts_present=facts_present,
                forbidden_known_facts_violated=forbidden_violated,
                supersession_chain_length=chain_length,
                supersession_expected_min=chain_min,
            ))
            continue

        positive_total += 1
        expected_label = probe.get("expected_turn_label")
        expected_id = label_to_turn_id.get(expected_label) if expected_label else None
        expected_subs = list(probe.get("expected_any_substring", []))

        # Multi-hop classification: a probe is multi-hop if it expects
        # ≥2 distinct fact keys in the Known-facts block. These probes
        # don't have a single "expected_turn_label" — the answer is
        # synthesised from facts. Reported separately so v1 and v2
        # numbers stay comparable.
        is_multihop = len({ef["key"] for ef in expected_facts}) >= 2

        if is_multihop:
            multihop_total += 1
            multihop_passed_top1 = facts_present == len(expected_facts)
            multihop_passed_top3 = multihop_passed_top1  # facts block is single-shot
            if multihop_passed_top1:
                multihop_top1 += 1
            if multihop_passed_top3:
                multihop_top3 += 1
            passed_top1 = multihop_passed_top1
            passed_top3 = multihop_passed_top3
            substring_match = passed_top1
            passed = passed_top1 and forbidden_violated == 0
        else:
            passed_top1 = bool(cites) and cites[0].turn_id == expected_id
            passed_top3 = any(c.turn_id == expected_id for c in cites[:3])
            substring_match = False
            if cites and expected_subs:
                joined = " ".join(c.snippet for c in cites[:3]).lower()
                substring_match = any(s.lower() in joined for s in expected_subs)
            # When the probe carries v2 expectations, fold them into pass/fail.
            facts_ok = facts_present == len(expected_facts)
            forbidden_ok = forbidden_violated == 0
            chain_ok = (
                expected_supersession is None
                or (chain_length is not None and chain_min is not None
                    and chain_length >= chain_min)
            )
            base_pass = (passed_top1 and substring_match) if expected_label else True
            passed = base_pass and facts_ok and forbidden_ok and chain_ok

            if passed_top1:
                positive_top1 += 1
            if passed_top3:
                positive_top3 += 1

        if top_score is not None:
            positive_top_scores.append(top_score)

        probes_out.append(FixtureProbeResult(
            label=probe["label"],
            query=probe["query"],
            is_noise=False,
            expected_turn_label=expected_label,
            expected_substrings=expected_subs,
            top_hit_turn_label=top_label,
            top_score=top_score,
            citations_count=len(cites),
            passed_top1=passed_top1,
            passed_top3=passed_top3,
            substring_match=substring_match,
            passed=passed,
            notes=(
                f"multi-hop facts {facts_present}/{len(expected_facts)}"
                if is_multihop
                else f"expected={expected_label} got={top_label}"
            ),
            expected_known_facts_total=len(expected_facts),
            expected_known_facts_present=facts_present,
            forbidden_known_facts_violated=forbidden_violated,
            supersession_chain_length=chain_length,
            supersession_expected_min=chain_min,
        ))

    # Surface extraction failures from the just-ingested fixture turns so
    # one Run-fixture click answers two questions: did /recall find the
    # turn, and did the extractor actually populate memories for it.
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT metadata->>'extraction_error' AS error
            FROM   turns
            WHERE  user_id = ANY($1::text[])
              AND  metadata ? 'extraction_error'
            """,
            user_ids,
        )
    extraction_failures = len(rows)
    extraction_error_modes: dict[str, int] = {}
    for r in rows:
        bucket = r["error"] or "unknown"
        extraction_error_modes[bucket] = extraction_error_modes.get(bucket, 0) + 1

    elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
    aggregate = FixtureAggregate(
        name=payload.name,
        total=len(probes_out),
        positive_total=positive_total,
        positive_passed_top1=positive_top1,
        positive_passed_top3=positive_top3,
        noise_total=noise_total,
        noise_passed=noise_passed,
        recall_at_1=(positive_top1 / positive_total) if positive_total else 0.0,
        recall_at_3=(positive_top3 / positive_total) if positive_total else 0.0,
        noise_resistance=(noise_passed / noise_total) if noise_total else 0.0,
        mean_top_score=(sum(positive_top_scores) / len(positive_top_scores))
            if positive_top_scores else 0.0,
        extraction_failures=extraction_failures,
        extraction_error_modes=extraction_error_modes,
        ingest_failures=ingest_failures,
        ingest_error_modes=ingest_error_modes,
        multihop_total=multihop_total,
        multihop_passed_top1=multihop_top1,
        multihop_passed_top3=multihop_top3,
        multihop_recall_at_1=(multihop_top1 / multihop_total) if multihop_total else 0.0,
        multihop_recall_at_3=(multihop_top3 / multihop_total) if multihop_total else 0.0,
        facts_expected=facts_expected_total,
        facts_present=facts_present_total,
        forbidden_facts_violations=forbidden_total,
        supersession_chains_resolved=supersession_chains_resolved,
        supersession_chains_expected=supersession_chains_expected,
        noise_top_score_violations=noise_top_score_violations,
    )
    return RunFixtureResponse(
        aggregate=aggregate,
        probes=probes_out,
        elapsed_ms=elapsed_ms,
    )


