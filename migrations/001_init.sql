-- =========================================================
-- Simple RAG schema. Two tables, one job:
--   documents   — the text we want to retrieve.
--   embeddings  — OpenAI text-embedding-3-small vectors, one per document.
-- HNSW cosine index for fast top-k.
-- =========================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,         -- caller-supplied stable id
    text        TEXT NOT NULL,            -- raw content we'll search over
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embeddings (
    doc_id      TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    -- 1536 = native dim of OpenAI text-embedding-3-small.
    -- text-embedding-3-large is 3072. If you switch, change this and rebuild.
    embedding   VECTOR(1536) NOT NULL,
    model       TEXT NOT NULL DEFAULT 'text-embedding-3-small',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW with cosine distance. m=16 / ef_construction=64 are pgvector defaults
-- and are appropriate for the row counts a single-developer demo will hit.
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Manual persistence harness: Alice/Bob chat history lives in the same named
-- Postgres volume, so it should survive `docker compose down && up`.
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL CHECK (user_id IN ('alice', 'bob')),
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created
    ON chat_messages (user_id, created_at, id);
