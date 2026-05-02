"""Request/response shapes — three endpoints, three pairs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------- POST /ingest ----------

class Document(BaseModel):
    id: str = Field(min_length=1, description="Stable caller-supplied id")
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[Document] = Field(min_length=1)


class IngestResponse(BaseModel):
    ingested: int


# ---------- POST /query ----------

class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class QueryHit(BaseModel):
    id: str
    text: str
    score: float            # cosine similarity in [-1, 1]; with normalized vectors, ~[0, 1]
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    hits: list[QueryHit]


# ---------- Manual Alice/Bob chat UI ----------

UserId = Literal["alice", "bob"]
ChatRole = Literal["user", "assistant"]


class ChatRequest(BaseModel):
    session_id: int = Field(ge=1)
    message: str = Field(min_length=1, max_length=4000)


class ChatMessage(BaseModel):
    id: int
    session_id: int
    user_id: UserId
    role: ChatRole
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    reply: str
    messages: list[ChatMessage]


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessage]


class ChatDeleteResponse(BaseModel):
    deleted: int


class ChatSession(BaseModel):
    id: int
    user_id: UserId
    title: str
    created_at: datetime
    message_count: int


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSession]


class ChatSessionCreateRequest(BaseModel):
    user_id: UserId
    title: str = Field(default="New chat", min_length=1, max_length=120)


class ChatSessionCreateResponse(BaseModel):
    session: ChatSession
