"""Request/response shapes — §3 contract + the manual /chat harness."""

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


# ---------- §3 contract: POST /turns ----------

Section3Role = Literal["user", "assistant", "tool"]


class TurnMessage(BaseModel):
    role: Section3Role
    name: str | None = None
    content: str = Field(min_length=1)


class TurnRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_id: str | None = None
    messages: list[TurnMessage] = Field(min_length=1)
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnResponse(BaseModel):
    id: str


# ---------- §3 contract: POST /recall ----------

class RecallRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    user_id: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=32768)


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation]


# ---------- §3 contract: POST /search ----------

class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None
    user_id: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchHit]


# ---------- §3 contract: GET /users/{user_id}/memories ----------

MemoryType = Literal["fact", "preference", "opinion", "event"]


class UserMemory(BaseModel):
    id: str
    type: MemoryType
    key: str
    value: str
    confidence: float
    source_session: str
    source_turn: str
    created_at: datetime
    updated_at: datetime
    supersedes: str | None = None
    active: bool


class UserMemoriesResponse(BaseModel):
    memories: list[UserMemory]
