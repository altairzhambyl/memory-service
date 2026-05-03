-- : hybrid retrieval prep.
-- Adds a Postgres FTS tsvector over documents.text and a GIN index so
-- /recall can fuse cosine top-k with BM25 top-k via RRF. Vanilla
-- cosine on text-embedding-3-small under-scores keyword-anchored
-- queries (e.g. "Stripe", "shellfish") — see CHANGELOG entry MS-16b
-- diagnostic where literal-token queries top out around 0.22-0.40.
-- BM25 catches that signal, RRF fuses both rankings.
--
-- `english` config is hard-coded — fixtures and the reference shape
-- are English-only. A multi-lingual setup would compute tsvector per
-- language column, out of scope for v1.
--
-- Generated STORED column: Postgres backfills automatically on ALTER
-- TABLE for existing rows, so reruns on populated volumes are safe.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX IF NOT EXISTS idx_documents_tsv
    ON documents USING GIN (tsv);
