-- =========================================================
-- Multi-session chat for the Alice/Bob harness.
--   chat_sessions   — one row per (user, named chat).
--   chat_messages   — gains session_id; user_id stays for fast diagnostic queries.
-- Backfill: every existing user gets a single "Default" session
-- and all their prior messages are linked to it.
-- =========================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL CHECK (user_id IN ('alice', 'bob')),
    title       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_created
    ON chat_sessions (user_id, created_at, id);

ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS session_id BIGINT
        REFERENCES chat_sessions(id) ON DELETE CASCADE;

-- Backfill in one shot: create a Default session per user that has
-- orphan messages, then point those messages at it. Wrapped in a DO
-- so it stays idempotent across restarts.
DO $$
DECLARE
    u TEXT;
    new_id BIGINT;
BEGIN
    FOR u IN
        SELECT DISTINCT user_id
        FROM chat_messages
        WHERE session_id IS NULL
    LOOP
        INSERT INTO chat_sessions (user_id, title)
        VALUES (u, 'Default')
        RETURNING id INTO new_id;

        UPDATE chat_messages
        SET    session_id = new_id
        WHERE  user_id = u AND session_id IS NULL;
    END LOOP;
END
$$;

ALTER TABLE chat_messages
    ALTER COLUMN session_id SET NOT NULL;

DROP INDEX IF EXISTS idx_chat_messages_user_created;

CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
    ON chat_messages (session_id, id);

CREATE INDEX IF NOT EXISTS idx_chat_messages_user_created
    ON chat_messages (user_id, created_at, id);
