"""HTTP routes — health, ingest, query. Read top-to-bottom; that's the whole API."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
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
    IngestRequest,
    IngestResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
)
from app.config import get_settings
from app.db.pool import get_pool
from app.embeddings.openai import embed_documents, embed_query, get_client

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
