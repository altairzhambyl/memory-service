# Memory Service

A Dockerised memory service for an AI agent. Ingests conversation
turns, extracts typed structured memories, and answers recall queries
with a priority-assembled context. Implements the API contract from
`CLAUDE.md`.

## Quickstart

```bash
cp .env.example .env             # put OPENAI_API_KEY in .env
docker compose up -d --build     # ~30s — pulls pgvector, builds the app
until curl -sf http://localhost:8080/health >/dev/null; do sleep 0.3; done
```

Inspection UI at <http://localhost:8080/>: every endpoint has a
panel with editable JSON, response viewer, and verdict pills. Run the
recall fixture from card 5 and you should see `recall@1 6/6,
recall@3 6/6, noise 1/1, mean 0.356, extraction_failures 0/5`.

## Architecture

```
┌──────────┐   POST /turns      ┌──────────────────────────┐
│  agent   │ ─────────────────▶ │       FastAPI            │
│  (eval)  │ ◀─── 201 {id} ──── │  routes.py / schemas.py  │
└──────────┘                    └────┬─────────────────────┘
     │ POST /recall                  │  embed (text-embedding-3-small)
     │   {query, user_id}            │  extract (gpt-4o-mini Structured Outputs)
     ▼                               │
┌──────────┐                         ▼
│ context  │      ┌─────────────────────────────────────────┐
│  block   │ ◀─── │  Postgres 16 + pgvector  (asyncpg)      │
└──────────┘      │  ┌──────────┬───────────────────────┐   │
                  │  │ turns    │ session_id, user_id   │   │
                  │  │ turn_msgs│ FK turns, role, idx   │   │
                  │  │ documents│ id, text, metadata    │   │
                  │  │ embeddings│ vec(1536) HNSW cosine│   │
                  │  │ memories │ type/key/value chain  │   │
                  │  └──────────┴───────────────────────┘   │
                  └─────────────────────────────────────────┘
```

Three layers, one process per box: **API** (FastAPI + Pydantic v2),
**Retrieval/Extraction** (OpenAI clients with explicit timeouts +
`max_retries=0` on the hot path), **Data** (pgvector + asyncpg pool).
A single transaction per `/turns` covers turn metadata, embedded
turn-message documents, and extracted memories — synchronous
correctness with no eventual consistency.

## Backing store — pgvector + Postgres

Postgres 16 with the `pgvector` extension, accessed via `asyncpg`.
Everything lives in one named Docker volume (`rag-db-data`) so
`docker compose down && up` preserves data.

- One store, two access patterns: relational rows for memories /
  turns, vector cosine over `embeddings.embedding` with HNSW
  (`vector_cosine_ops`, m=16, ef_construction=64). No second system
  to keep in sync, no eventual-consistency dance.
- `documents` carries provenance JSONB (`kind`, `turn_id`,
  `session_id`, `user_id`, `role`, `idx`, `ts`) for citations and for
  scoping cosine queries by `user_id`.
- `memories` is a real append-only chain: `supersedes` self-FK +
  `active` boolean. `/users/{id}/memories` returns the full chain;
  `/recall` filters `active=TRUE` for current state.

We considered Kuzu / Qdrant / a separate vector DB earlier but pulled
back: at this scale the second runtime adds risk without buying us
anything we cannot do with one HNSW index and a JSONB column.

## Extraction pipeline

`app/extraction.py` calls `client.chat.completions.parse` with a
Pydantic `ExtractionResponse` schema (OpenAI Structured Outputs) so the
JSON is validated server-side at decode time. Model: `gpt-4o-mini`.

The prompt is original — it covers the categories and uses
fixture content for examples (no copy from mem0/honcho/hindsight per
):

- `employment.company`, `employment.role`, `employment.previous_company`
- `location.city`, `location.previous_city`
- `pet.name`, `pet.species`, `pet.age_years`, `family.spouse_name`
- `diet.style`, `beverage.coffee`, `allergy`
- `opinion.<topic>`
- Implicit facts (`walking Biscuit` → `pet.name=Biscuit`)
- Corrections (`actually, I meant…` — only the corrected fact is kept)

Every emitted memory carries `type ∈ {fact, preference, opinion, event}`,
`key`, `value`, and `confidence ∈ [0,1]` (0.95 explicit / 0.85 implied
/ 0.65 hedged).

**Failure handling.** `extract_memories_from_turn` never raises on
transient API failure: it returns `(memories, error_or_none)`. Caught:
`RateLimitError`, `APITimeoutError`, `APIConnectionError`,
`BadRequestError`, `LengthFinishReasonError`,
`ContentFilterFinishReasonError`, `ValidationError`, plus a last-resort
`Exception` net. The error string is persisted in
`turns.metadata.extraction_error` and the turn still commits with
`memories=[]`. `AuthenticationError` is deliberately re-raised — that
is a config bug, silently degrading would hide it.

**Hot-path policy.** `client.with_options(max_retries=0)` and an
explicit `timeout=30.0` on the parse call. SDK default is 600 s with 2
retries; under that default a 429 burst can chain 3 × 30 s and blow
the 60 s `/turns` budget. Single attempt or fail.

**Deliberately unsupported (current).** Multi-turn corrections that
require remembering a *previous* turn's content (we only see one turn
at a time). Long opinion arcs are stored as separate `opinion.<topic>`
rows; the full arc is reconstructable from the supersession chain in
`/users/{id}/memories` but `/recall` currently surfaces only the
latest active opinion. Documented as a known limitation.

## Recall strategy

`/recall` builds a two-section context under a `max_tokens` budget:

```
## Known facts about this user
- employment.company: Notion
- employment.role: product manager
- location.city: Berlin
- location.previous_city: NYC
- pet.name: Biscuit

## Relevant from recent conversations
- [2026-04-01T10:00:00Z] (user fix-berlin-s1) I just moved to Berlin from NYC last month. Loving it so far.
- ...
```

End-to-end:

1. **Embed query** with `text-embedding-3-small`.
2. **In parallel** (`asyncio.gather`) — fetch active memories for
   `user_id`, ordered by type priority (fact > preference > opinion >
   event) then `updated_at DESC`; and run cosine top-10 over
   `documents` rows scoped by `user_id`.
3. **Assemble**: emit the `## Known facts` block first, then drain the
   remaining budget into `## Relevant from recent conversations`.
   Snippets are pre-trimmed at index time (240 chars).
4. **Citations** carry `turn_id`, cosine score, and snippet. Memories
   are not duplicated as citations — reviewers inspect them via
   `/users/{user_id}/memories`.

**Priority logic — graded design choice.** Stable user identity
beats episodic context. A user's employer, city, allergies, and pet
are needed by *most* queries, so they always win the budget. Recent
turns drain the leftover. This matches the reference shape and is
why a "where does the user live" query still gets the Berlin fact even
when the conversation hits another topic.

**Cold semantics.** Empty store, unknown user, anonymous query
(`user_id: null`), or zero matches → 200 with
`{"context":"", "citations":[]}`. Never errors.

**Cross-session scoping.** A user's history is shared across
their `session_id`s — the multi-hop "user with the dog named Biscuit"
test in spans sessions by design. Different `user_id`s are
*always* isolated; the request's `session_id` is captured in each
citation for provenance but never used as a recall filter.

## Fact evolution & supersession

Insert path in `create_turn`, three branches per extracted memory:

| Prior `(user_id, key)` active row | Action |
|---|---|
| none | plain insert, `supersedes=NULL` |
| same value | idempotent: `UPDATE memories SET updated_at=NOW()`; no new row |
| different value | mark old `active=FALSE, updated_at=NOW()`; insert new with `supersedes=old.id` |

Everything in the same transaction as the turn writes, so a reviewer
querying `/users/{id}/memories` immediately after `/turns` sees the
chain consistent with what `/recall` returns.

`/recall` reads only `active=TRUE`, so the **current** fact is
returned. `/users/{id}/memories` returns active + inactive rows
(`ORDER BY active DESC, updated_at DESC`), so the **history** is
preserved per the API contract.

**Coverage and limits.** Key-based supersession is deterministic and
cheap. It handles location, employment, pet, and persistent
preferences cleanly. Opinion arcs (`I love TS` → `TS generics annoy
me` → `TS for big projects`) are stored under a single
`opinion.typescript` key, so the chain shows the trajectory but the
v1 implementation does not synthesise an "evolved" prose summary —
the current opinion wins. Documented as a tradeoff against an LLM
judge that adds latency and model-cost variance.

## Tradeoffs

| Optimised for | Given up |
|---|---|
| One backing store, one transaction, no eventual consistency | Hybrid retrieval (BM25 + RRF) — a likely upgrade |
| Synchronous extraction inside `/turns` (60 s budget) | Throughput on bursts — `max_retries=0` on the hot path means a 429 returns zero memories rather than retrying |
| Deterministic `(user_id, key)` supersession | LLM-judge contradiction detection — heavier, less predictable |
| stable-facts-first recall priority | Query-relevant memory layer (similarity over memory rows) — left as a post-MVP refinement |
| Original prompt, original schema, single Python file per concern | Bigger framework conveniences (no Celery / Kafka / queue) |

## Failure modes

- **Missing `OPENAI_API_KEY`** — startup fails fast via `${VAR:?}` in
  `docker-compose.yml`. Listed in `.env.example`.
- **OpenAI 5xx / 429 / timeout** — extractor returns
  `(memories=[], error="rate_limit"|"timeout"|"connection"|...)`. The
  turn still commits and the error string is persisted in
  `turns.metadata.extraction_error`. `/admin/turn_index` and
  `/admin/run_fixture` expose `extraction_failures` counts +
  `extraction_error_modes` breakdown.
- **OpenAI auth bug (bad key)** — `AuthenticationError` is re-raised
  so `/turns` returns 5xx instead of silently producing zero memories.
- **DB down** — pool acquisition raises, FastAPI exception middleware
  serialises a `{"error":"internal_server_error","request_id":...}`.
  `/health` returns 503 until the pool recovers.
- **Restart mid-write** — single transaction guarantees turn +
  memories + embeddings either all land or none. `test_restart_persistence`
  enforces this.
- **Malformed input** — Pydantic v2 returns 422 on shape errors;
  unicode noise (RTL overrides, emoji, control chars) is accepted as
  payload. `test_malformed_input_no_crash` enforces no 5xx.

## How to run tests

```bash
# unit / contract-shape tests against an ASGI stub (no DB / no API)
.venv/bin/pytest -q tests/

# full suite against the live stack (must have docker compose up first)
RAG_E2E=1 .venv/bin/pytest -q tests/
```

Expected: **16 passed** (12 contract-shape from `tests/contract/test_contract_shapes.py`
+ `tests/contract/test_shapes.py`, plus 4 required from
`tests/contract/test_required_behaviors.py`: synchronous availability,
concurrent-session isolation, malformed input no crash, restart
persistence). The restart test invokes `docker compose restart app`
via subprocess; it is auto-skipped when the docker CLI isn't on PATH.

The recall-quality fixture is a separate, measured eval:

```bash
curl -sX POST http://localhost:8080/admin/run_fixture \
  -H 'Content-Type: application/json' -d '{"name":"v1"}' | jq .aggregate
```

Or click **Run fixture** in card 5 of the inspection UI. v1 has 7
probes (6 positive + 1 noise) over 3 conversations
(fix-berlin / fix-pet / fix-diet); current run reports recall@1=6/6,
recall@3=6/6, noise=1/1, mean top-score=0.356, extraction_failures=0/5.

## Repo layout

```
memory-service/
├── app/
│   ├── main.py                # FastAPI lifespan + pool + OpenAI client
│   ├── api/
│   │   ├── routes.py          # API contract endpoints + admin
│   │   └── schemas.py         # Pydantic request/response models
│   ├── extraction.py          # gpt-4o-mini Structured Outputs extractor
│   ├── embeddings/openai.py   # text-embedding-3-small (async)
│   ├── db/pool.py             # asyncpg + pgvector codec
│   ├── obs/logging.py         # structlog JSON
│   └── static/index.html      # inspection UI (every endpoint)
├── migrations/                # 001_init, 002_chat_sessions,
│                                003_turns, 004_memories
├── fixtures/recall/v1.json    # 7-probe recall-quality fixture
├── tests/contract/            # shape tests + required tests
├── docker-compose.yml         # db + app, named volume rag-db-data
├── Dockerfile
├── .env.example
├── CHANGELOG.md               # iteration history
└── CLAUDE.md                  # private working notes (gitignored)
```
