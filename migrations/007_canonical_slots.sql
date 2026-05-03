-- fact evolution + concurrency safety (MS-26 + MS-27).
--
-- Adds a canonical `slot` column to `memories`. Canonicalisation maps
-- variable LLM-emitted keys (opinion.typescript, opinion.typescript.generics,
-- language.favourite, programming.language) to a stable slot
-- (opinion.typescript) so supersession fires across key variants. The
-- display `key` is preserved for /users/{id}/memories inspection — only
-- the supersession lookup uses `slot`.
--
-- A partial unique index `(user_id, slot) WHERE active = TRUE` enforces
-- "at most one active row per slot per user" at the DB level. This is
-- the concurrency backstop: even if two /turns race past the
-- application's read-modify-write, Postgres rejects the second insert
-- and the application catches the unique-violation, retries the
-- supersession path, and converges to a single active row.
--
-- Canonicalisation backfill happens in Python (POST /admin/recanonicalise_memories)
-- because the rule set lives in the application — keeping it in code
-- means a reviewer reads one Python dict instead of bouncing between
-- SQL and Python. SQL here only declares structure.

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS slot TEXT;

-- Partial unique index. Postgres treats NULL as distinct in unique
-- indices, so unbackfilled rows (slot IS NULL) don't trip the
-- constraint until the operator runs the backfill endpoint.
CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_user_slot_active
    ON memories (user_id, slot)
    WHERE active = TRUE AND slot IS NOT NULL;

-- The legacy (non-unique) index on (user_id, active, key) stays for
-- the /users/{id}/memories listing path — it's still the right cover
-- for "all rows for a user, active first" queries.
