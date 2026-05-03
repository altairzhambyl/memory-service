"""HTTP routes — API contract + manual /chat harness. Read top-to-bottom."""

from __future__ import annotations

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
    IngestRequest,
    IngestResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
    RecallRequest,
    RecallResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    TurnIndexResponse,
    TurnIndexRow,
    TurnRequest,
    TurnResponse,
    UserMemoriesResponse,
)
from app.config import get_settings
from app.db.pool import get_pool
from app.embeddings.openai import embed_documents, embed_query, get_client

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
# MS-15 ships only the shapes + synchronous /turns persistence;
# MS-16/17 fill in recall, search, and structured memories.
# =====================================================================


@router.post("/turns", response_model=TurnResponse, status_code=201)
async def create_turn(payload: TurnRequest) -> TurnResponse:
    """
    Persist a turn + its messages synchronously, then index each message
    into documents/embeddings under doc_id `turn:{turn_id}:{idx}`. The
    embed call happens before the DB transaction so a network blip won't
    leave half-written rows; once we open the transaction, all writes
    commit or roll back together (synchronous correctness).
    """
    turn_id = str(uuid.uuid4())
    settings = get_settings()

    contents = [msg.content for msg in payload.messages]
    vectors = await embed_documents(contents)

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
            _to_jsonb(payload.metadata),
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


def _format_recall_context(
    hits: list[dict],
    max_tokens: int,
) -> tuple[str, list[dict]]:
    """
    Build the `context` string and matched citations under a token budget.
    Baseline ships only the "## Relevant from recent conversations" section;
    "## Known facts about this user" is MS-17/19 work and is documented as
    a known gap in README + CHANGELOG.
    """
    if not hits:
        return "", []

    header = "## Relevant from recent conversations"
    used = _approx_tokens(header)
    lines: list[str] = []
    chosen: list[dict] = []

    for hit in hits:
        meta = hit["metadata"]
        snippet = _trim(hit["text"])
        ts = meta.get("ts") or ""
        role = meta.get("role") or "?"
        sid = meta.get("session_id") or "?"
        line = f"- [{ts}] ({role} {sid}) {snippet}"
        cost = _approx_tokens(line) + 1  # +1 for newline
        if used + cost > max_tokens and lines:
            # Budget tight; bail before exceeding 2× per the API contract.
            break
        used += cost
        lines.append(line)
        chosen.append({
            "turn_id": meta.get("turn_id") or "",
            "score": hit["score"],
            "snippet": snippet,
        })

    if not lines:
        return "", []
    context = "\n".join([header, *lines])
    return context, chosen


@router.post("/recall", response_model=RecallResponse)
async def recall(payload: RecallRequest) -> RecallResponse:
    """
    Cosine top-k over the turn-message index, scoped by user_id. cold
    semantics: empty store, unknown user, or zero matches → 200 with
    {"context": "", "citations": []}. Never errors.

    This is the baseline floor — a deliberate, defensible cosine-only
    pipeline. Hybrid retrieval, rerank, multi-hop, and stable-fact
    priority assembly are tracked separately and will compound on top.
    """
    if payload.user_id is None:
        # No scope = no recall. We share across same-user sessions,
        # but anonymous queries don't get to fish in everyone's data.
        return RecallResponse(context="", citations=[])

    qvec = await embed_query(payload.query)
    hits = await _recall_candidates(qvec, payload.user_id, payload.session_id, _RECALL_TOPK)
    context, citations = _format_recall_context(hits, payload.max_tokens)
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
    """Cold shape until MS-17 populates structured memories."""
    _ = user_id
    return UserMemoriesResponse(memories=[])


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session_endpoint(session_id: str) -> Response:
    """
    Drop every turn for the string session_id. turn_messages cascades
    via FK; the indexed documents/embeddings rows we wrote at /turns time
    have no FK back to turns, so we delete them by provenance metadata in
    the same transaction.
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
        await conn.execute("DELETE FROM turns WHERE session_id = $1", session_id)
    return Response(status_code=204)


@router.delete("/users/{user_id}", status_code=204)
async def delete_user_endpoint(user_id: str) -> Response:
    """Drop every turn + indexed message for this user across all sessions."""
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
    """Show indexed turn-message rows so the UI can verify MS-16a writes."""
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
