-- =========================================================
-- — typed structured memories. One row per extracted
-- fact/preference/opinion/event. The columns mirror the 
-- reference shape for GET /users/{user_id}/memories so the
-- response can stream straight from the row.
--
-- MS-17a-i ships the table + read path; MS-17a-ii will fill
-- it via an LLM extractor inside POST /turns. MS-18 will wire
-- supersession by promoting `supersedes` to a self-referential
-- FK and turning the unique index on (user_id, key, active=TRUE).
-- For now `supersedes` is just a TEXT pointer (nullable) so the
-- migration stays self-contained and idempotent.
-- =========================================================

CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    type            TEXT NOT NULL CHECK (type IN ('fact', 'preference', 'opinion', 'event')),
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    source_session  TEXT NOT NULL,
    source_turn     TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    supersedes      TEXT,
    active          BOOLEAN NOT NULL DEFAULT TRUE
);

-- Composite covers the hot path: list active memories for a user,
-- look up an existing active row by (user_id, key) for MS-18.
CREATE INDEX IF NOT EXISTS idx_memories_user_active_key
    ON memories (user_id, active, key);
