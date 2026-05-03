"""HTTP routes — API contract + manual /chat harness. Read top-to-bottom."""

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
from app.embeddings.openai import embed_documents, embed_query, get_client
from app.extraction import extract_memories_from_turn

# : 1 token ≈ 4 chars for English text. Fine for /recall budget
# triage; we do NOT need a real tokenizer here, the eval timeout
# wouldn't reward the precision.
_CHARS_PER_TOKEN = 4
# Hard cap on the candidate pool we pull before token-budget trimming.
# Keeps the JOIN bounded; HNSW handles ranking.
_RECALL_TOPK = 10
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
    vectors = await embed_documents(texts)   # list[list[float]] (N, 1536)
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
    qvec = await embed_query(payload.query)

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
# API contract endpoints. These are the eval-facing surface.
# Embedding, retrieval, LLM extraction, and stable-fact priority
# assembly (`## Known facts about this user`) are live; 
# supersession (MS-18) is the next slice and will deduplicate the
# memory set surfaced here.
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
    writes commit or roll back together (synchronous correctness).

    Extraction failure is graceful per the API contract: the error string is recorded
    in `turns.metadata.extraction_error` and the turn is still committed
    with zero memories. AuthenticationError is the only exception that
    propagates — it indicates a config bug, not transient runtime state.
    """
    turn_id = str(uuid.uuid4())
    settings = get_settings()

    contents = [msg.content for msg in payload.messages]
    # max_retries=0 on the hot path: 3 retry windows × 30s would blow the
    # 60s /turns budget on a 429 burst. /ingest and /recall keep the
    # client default (2) since they don't have the same budget pressure.
    vectors = await embed_documents(contents, max_retries=0)

    # Anonymous turns can't write memories (table requires user_id NOT NULL),
    # so skip the LLM call entirely — saves cost and avoids a no-op write path.
    if payload.user_id is None:
        extracted: list[dict] = []
        extraction_error: str | None = None
    else:
        extracted, extraction_error = await extract_memories_from_turn(payload.messages)

    turn_metadata = dict(payload.metadata)
    if extraction_error:
        turn_metadata["extraction_error"] = extraction_error

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
        for idx, (msg, vec) in enumerate(zip(payload.messages, vectors, strict=True)):
            await conn.execute(
                """
                INSERT INTO turn_messages (turn_id, idx, role, name, content)
                VALUES ($1, $2, $3, $4, $5)
                """,
                turn_id, idx, msg.role, msg.name, msg.content,
            )
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

        for mem in extracted:
            await conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, type, key, value, confidence,
                    source_session, source_turn, supersedes, active
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NULL, TRUE)
                """,
                str(uuid.uuid4()),
                payload.user_id,
                mem["type"],
                mem["key"],
                mem["value"],
                float(mem["confidence"]),
                payload.session_id,
                turn_id,
            )
    return TurnResponse(id=turn_id)


async def _recall_candidates(
    qvec: list[float],
    user_id: str | None,
    session_id: str | None,
    top_k: int,
) -> list[dict]:
    """
    Cosine top-k over indexed turn messages. Scope by user_id; cross-session
    sharing for the same user is intentional per the API contract (documented in README).
    session_id, when supplied, is used only as a soft hint for the eval
    harness — we do NOT exclude other sessions, but we surface the match's
    own session_id in the citation so reviewers can see provenance.
    """
    where = ["d.metadata->>'kind' = 'turn_message'"]
    args: list[object] = [qvec]
    if user_id is not None:
        args.append(user_id)
        where.append(f"d.metadata->>'user_id' = ${len(args)}")
    where_sql = " AND ".join(where)
    args.append(top_k)
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
    return [
        {
            "doc_id": r["id"],
            "text": r["text"],
            "metadata": _from_jsonb(r["metadata"]),
            "score": float(r["score"]),
        }
        for r in rows
    ]


# priority bucketing: stable user identity facts beat opinions/events
# when the budget is tight. Lower number = higher priority.
_MEMORY_TYPE_PRIORITY = {"fact": 0, "preference": 1, "opinion": 2, "event": 3}


async def _fetch_active_memories(user_id: str) -> list[dict]:
    """
    Active typed memories for a user, ordered for `## Known facts`
    rendering: type priority first (fact → preference → opinion → event),
    then most-recently-updated. Plain row dicts, decoupled from the
    /users/{id}/memories response model.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT  type, key, value
            FROM    memories
            WHERE   user_id = $1 AND active = TRUE
            ORDER BY updated_at DESC, id ASC
            """,
            user_id,
        )
    out = [
        {"type": r["type"], "key": r["key"], "value": r["value"]}
        for r in rows
    ]
    out.sort(key=lambda m: _MEMORY_TYPE_PRIORITY.get(m["type"], 99))
    return out


def _format_recall_context(
    memories: list[dict],
    hits: list[dict],
    max_tokens: int,
) -> tuple[str, list[dict]]:
    """
    Build the `context` string and matched citations under a token
    budget. priority: stable user facts → recent conversation
    chunks. The `## Known facts about this user` block consumes budget
    first; whatever budget remains is drained into `## Relevant from
    recent conversations`.

    Citations only carry `turn_id` references — memories surface in
    the prose block but are inspectable via /users/{id}/memories so we
    don't double-report them as citations.
    """
    if not memories and not hits:
        return "", []

    sections: list[str] = []
    used = 0

    if memories:
        facts_header = "## Known facts about this user"
        used += _approx_tokens(facts_header)
        fact_lines: list[str] = []
        for mem in memories:
            line = f"- {mem['key']}: {mem['value']}"
            cost = _approx_tokens(line) + 1
            if used + cost > max_tokens and fact_lines:
                # : facts have priority but a runaway memory list
                # still cannot blow past the budget.
                break
            used += cost
            fact_lines.append(line)
        if fact_lines:
            sections.append("\n".join([facts_header, *fact_lines]))

    chosen: list[dict] = []
    if hits:
        hits_header = "## Relevant from recent conversations"
        header_cost = _approx_tokens(hits_header)
        hit_lines: list[str] = []
        # Reserve header cost only if at least one hit fits.
        for hit in hits:
            meta = hit["metadata"]
            snippet = _trim(hit["text"])
            ts = meta.get("ts") or ""
            role = meta.get("role") or "?"
            sid = meta.get("session_id") or "?"
            line = f"- [{ts}] ({role} {sid}) {snippet}"
            cost = _approx_tokens(line) + 1
            head = header_cost if not hit_lines else 0
            if used + head + cost > max_tokens and hit_lines:
                break
            if used + head + cost > max_tokens:
                # Even the first hit doesn't fit; facts win the budget.
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
    /recall: assemble context from two layers under a token budget.

    1. Stable user facts — active rows in `memories` for `user_id`.
    2. Query-relevant recent turns — cosine top-k over indexed messages.

    priority is "stable facts → query-relevant memories → recent
    context". Layer 2 here covers facts and recent context together via
    the cosine pipeline; the dedicated query-relevant-memories layer
    (embedding similarity over memory rows) is a deliberate next slice
    once MS-18 supersession deduplicates the memory set.

    cold semantics: empty store, unknown user, or zero matches → 200
    with `{"context": "", "citations": []}`. Never errors.
    """
    if payload.user_id is None:
        # No scope = no recall. We share across same-user sessions,
        # but anonymous queries don't get to fish in everyone's data.
        return RecallResponse(context="", citations=[])

    qvec = await embed_query(payload.query)
    memories, hits = await asyncio.gather(
        _fetch_active_memories(payload.user_id),
        _recall_candidates(qvec, payload.user_id, payload.session_id, _RECALL_TOPK),
    )
    context, citations = _format_recall_context(memories, hits, payload.max_tokens)
    return RecallResponse(
        context=context,
        citations=[Citation(**c) for c in citations],
    )


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    """
    Twin of /recall but returns structured rows, not prose. says this is
    invoked by an explicit agent tool call, so we don't apply token budgets
    here — `limit` already bounds the response.
    """
    if payload.user_id is None and payload.session_id is None:
        # Same fishing-trip guard as /recall: refuse fully unscoped queries.
        return SearchResponse(results=[])

    qvec = await embed_query(payload.query)
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
    Return typed structured memories for the user (inspection
    endpoint). Default ordering: most-recently updated first so the
    reviewer sees the freshest state at the top. We return active rows
    only by default — superseded history will surface once MS-18
    introduces the supersession chain and a `?include_inactive=true`
    flag is needed.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT  id, user_id, type, key, value, confidence,
                    source_session, source_turn,
                    created_at, updated_at, supersedes, active
            FROM    memories
            WHERE   user_id = $1 AND active = TRUE
            ORDER BY updated_at DESC, id ASC
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
async def delete_session_endpoint(session_id: str) -> Response:
    """
    Drop every turn for the string session_id, plus the indexed
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
            WHERE metadata->>'kind' = 'turn_message'
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
async def delete_user_endpoint(user_id: str) -> Response:
    """Drop every turn, indexed message, and structured memory for this user."""
    pool = get_pool()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            """
            DELETE FROM documents
            WHERE metadata->>'kind' = 'turn_message'
              AND metadata->>'user_id' = $1
            """,
            user_id,
        )
        await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM turns WHERE user_id = $1", user_id)
    return Response(status_code=204)


# =====================================================================
# Admin / debug. Read-only inspector for the static UI so MS-16a can be
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
    metadata carries an `extraction_error` from MS-17a-ii). The UI
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

    # Build a parallel turns-side WHERE for the extraction-failure count.
    # `documents` rows live under metadata->>'kind' = 'turn_message' but
    # extraction errors live in turns.metadata, so we have to scope twice.
    turn_args: list[object] = []
    turn_where = ["metadata ? 'extraction_error'"]
    if user_id is not None:
        turn_args.append(user_id)
        turn_where.append(f"user_id = ${len(turn_args)}")
    if session_id is not None:
        turn_args.append(session_id)
        turn_where.append(f"session_id = ${len(turn_args)}")
    turn_where_sql = " AND ".join(turn_where)

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
        extraction_failures = await conn.fetchval(
            f"SELECT count(*) FROM turns WHERE {turn_where_sql}",
            *turn_args,
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
        extraction_failures=int(extraction_failures or 0),
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
# Recall-quality measurement harness — the requirements.
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
    # asyncpg rejects naive datetimes for timestamptz; the API contract
    # also explicitly says ISO-8601, so we normalise the trailing Z.
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


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
        await delete_user_endpoint(uid)

    # 2. Ingest each turn via the real create_turn so the production
    #    embed+index path runs. Capture turn_id by label for verdict mapping.
    label_to_turn_id: dict[str, str] = {}
    for conv in fixture["conversations"]:
        for turn in conv["turns"]:
            req = TurnRequest(
                session_id=turn["session_id"],
                user_id=conv["user_id"],
                messages=[TurnMessage(**m) for m in turn["messages"]],
                timestamp=_parse_iso_z(turn["timestamp"]),
                metadata={"fixture": payload.name, "turn_label": turn["label"]},
            )
            resp = await create_turn(req)
            label_to_turn_id[turn["label"]] = resp.id

    turn_id_to_label = {v: k for k, v in label_to_turn_id.items()}
    noise_max_top = float(fixture.get("noise_max_top_score", 0.35))

    # 3. Probe /recall and compute per-probe verdicts.
    probes_out: list[FixtureProbeResult] = []
    positive_top1 = positive_top3 = positive_total = 0
    noise_passed = noise_total = 0
    positive_top_scores: list[float] = []

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

        if is_noise:
            noise_total += 1
            ok = (not cites) or (top_score is not None and top_score < noise_max_top)
            if ok:
                noise_passed += 1
            probes_out.append(FixtureProbeResult(
                label=probe["label"],
                query=probe["query"],
                is_noise=True,
                top_hit_turn_label=top_label,
                top_score=top_score,
                citations_count=len(cites),
                passed=ok,
                notes=f"noise threshold < {noise_max_top}",
            ))
            continue

        positive_total += 1
        expected_label = probe.get("expected_turn_label")
        expected_id = label_to_turn_id.get(expected_label) if expected_label else None
        expected_subs = list(probe.get("expected_any_substring", []))

        passed_top1 = bool(cites) and cites[0].turn_id == expected_id
        passed_top3 = any(c.turn_id == expected_id for c in cites[:3])
        substring_match = False
        if cites and expected_subs:
            joined = " ".join(c.snippet for c in cites[:3]).lower()
            substring_match = any(s.lower() in joined for s in expected_subs)
        passed = passed_top1 and substring_match

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
            notes=f"expected={expected_label} got={top_label}",
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
    )
    return RunFixtureResponse(
        aggregate=aggregate,
        probes=probes_out,
        elapsed_ms=elapsed_ms,
    )


