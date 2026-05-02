-- =========================================================
-- API contract storage. The eval harness writes turns and reads
-- recall/memories against THESE tables.
--   turns           — one row per POST /turns (string id, string session_id).
--   turn_messages   — flattened message log; cascades on turn delete.
-- session_id is a caller-supplied string and is intentionally
-- separate from the int chat_sessions.id used by the manual harness.
-- =========================================================

CREATE TABLE IF NOT EXISTS turns (
    id          TEXT PRIMARY KEY,                       -- service-generated uuid
    session_id  TEXT NOT NULL,                          -- caller-supplied per the API contract
    user_id     TEXT,                                   -- nullable per the API contract
    ts          TIMESTAMPTZ NOT NULL,                   -- caller-supplied
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id);
CREATE INDEX IF NOT EXISTS idx_turns_user    ON turns (user_id);

CREATE TABLE IF NOT EXISTS turn_messages (
    turn_id     TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    idx         INT  NOT NULL,                          -- 0-based order within the turn
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    name        TEXT,                                   -- tool name, if role='tool'
    content     TEXT NOT NULL,
    PRIMARY KEY (turn_id, idx)
);
