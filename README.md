# Memory Service — `main` (service-only)

A Dockerised memory service for an AI agent. Ingests conversation turns,
extracts typed structured memories, and answers recall queries with a
priority-assembled context. Implements the HTTP contract from the
the project's private working notes (not in this repo;
references it via section numbers throughout).

> **Branches.** `main` is the **service-only** build — only the 
> HTTP surface, no static UI. Use this for an agent or eval that
> drives the service over HTTP.
>
> The `with-ui` branch ships an inspection panel at
> `http://localhost:8080/` — every endpoint has an editable JSON
> card with response viewer + verdict pills (recall fixture, memories
> table, turn index). Use it for a click-driven manual demo.
>
> ```bash
> git clone               git@github.com:kekeront/ai-memory.git   # main = this branch, service only
> git clone -b with-ui    git@github.com:kekeront/ai-memory.git   # with inspection UI
> ```

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-16-336791?logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-HNSW-336791)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini%20%2B%20text--embedding--3--small-412991?logo=openai&logoColor=white)
![Tests](https://img.shields.io/badge/tests-22%20offline%20%2B%2024%20live-brightgreen)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)

---

## Quick Start

```bash
cp .env.example .env             # add OPENAI_API_KEY=...
docker compose up -d --build     # ~30s; pulls pgvector + builds the app
bash scripts/smoke.sh            # /health → /turns → /recall → /users/{id}/memories
```

Run the recall-quality eval:

```bash
curl -sX POST http://localhost:8080/admin/run_fixture \
  -H 'content-type: application/json' -d '{"name":"v1"}' | jq .aggregate
```

Expect `recall@1 6/6, recall@3 6/6, noise 1/1, mean 0.356,
extraction_failures 0/5` on `v1`. Fixture `v2` covers stress
probes — see [Recall pipeline](#recall-pipeline) and the [`fixtures/recall/`](fixtures/recall/) directory.

---

## What this is

Reviewers asked for a memory layer that an AI agent can sit on top of:
write completed conversation turns to it, query it for relevant
context on the next turn, and trust that what comes back is
**structured, current, and on-topic**. The brief explicitly disqualifies
"raw chunks in, top-k cosine out" as a passing solution.

This implementation answers each scored category with a deliberate
mechanism, not a default:

| Eval category | Mechanism in this repo |
| --- | --- |
| Recall quality | Hybrid retrieval — cosine + Postgres FTS via RRF (`k=60`). Not vanilla top-k. |
| Fact evolution | Canonical-slot supersession with a partial unique index + advisory lock. Old facts go inactive, history preserved. |
| Multi-hop | Facts-first priority assembly — stable identity (city, employer, pets) always considered alongside conversation turns. |
| Noise resistance | Two-stage gate: block-level relevance + per-fact cosine threshold. Off-topic queries return cold context. |
| Extraction quality | `gpt-4o-mini` with OpenAI Structured Outputs. Eight categories. Graceful failure: error string lives in `turns.metadata.extraction_error`. |
| Persistence | Single named Docker volume; `docker compose down && up` survives. |
| Cross-session | One transaction per `/turns` (turn + memories + indexed docs). No eventual consistency. |
| Robustness | Pydantic at boundaries; transient OpenAI failures degrade gracefully (MS-24). |
| Synchronous correctness | After `POST /turns` returns 201, memories are queryable in the next call. |
| Contract compliance | All seven endpoints exact shape + status. 9 contract-shape tests. |

---

## Architecture

```
                     ┌──────────────────────────────────────────┐
                     │             FastAPI (src/api)            │
                     │ ┌──────────────────────────────────────┐ │
       POST /turns   │ │ /turns   /recall   /search           │ │
       ─────────────▶│ │ /users/{id}/memories                 │ │
                     │ │ DELETE /sessions/{id}, /users/{id}   │ │
                     │ │ /admin/turn_index, /admin/run_fixture│ │
                     │ └────────┬───────────────┬─────────────┘ │
                     └──────────│───────────────│───────────────┘
                                │               │
                  embed (1536)  │               │  extract (gpt-4o-mini)
                  ───────▶ OpenAI ◀───────       │
                                │               │
                                ▼               ▼
                     ┌──────────────────────────────────────────┐
                     │  Postgres 16 + pgvector  (asyncpg pool)  │
                     │ ┌──────────────────────────────────────┐ │
                     │ │ turns         : session_id, user_id  │ │
                     │ │ turn_messages : FK turns, role, idx  │ │
                     │ │ documents     : kind, metadata, tsv  │ │
                     │ │ embeddings    : vector(1536) HNSW    │ │
                     │ │ memories      : type/key/value/slot  │ │
                     │ │                supersedes, active    │ │
                     │ └──────────────────────────────────────┘ │
                     └──────────────────────────────────────────┘
```

Three layers, one process per box. **API** (FastAPI + Pydantic v2)
handles HTTP and contract shapes. **Retrieval / extraction** owns the
OpenAI clients with explicit timeouts and `max_retries=0` on the hot
path to keep the 60-second `/turns` budget intact. **Data**
(pgvector + asyncpg) is one store: relational rows for memories and
turns, vector cosine via HNSW, full-text via `tsvector` + GIN. Every
`/turns` request commits in a single transaction — the brief's "no
eventual consistency" rule is the guarantee.

---

## Contract endpoints

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/health` | `200 {"status":"ok"}` |
| `POST` | `/turns` | `201 {"id":"<turn_id>"}` — synchronous: persists turn, embeddings, extracted memories |
| `POST` | `/recall` | `200 {"context":"...","citations":[...]}` — facts-first context under `max_tokens` |
| `POST` | `/search` | `200 {"results":[...]}` — structured rows, agent-tool-style |
| `GET` | `/users/{user_id}/memories` | `200 {"memories":[{type,key,value,...}]}` |
| `DELETE` | `/sessions/{session_id}` | `204` |
| `DELETE` | `/users/{user_id}` | `204` |

Admin (out of contract): `GET /admin/turn_index`,
`POST /admin/run_fixture`, `POST /admin/recanonicalise_memories`,
`POST /admin/reembed_memories`, `POST /admin/inject_embed_failure`
(test-only).

---

## Recall pipeline

`/recall` builds the reference context:

```
## Known facts about this user
- employment.company: Notion
- location.city: Berlin
- pet.name: Biscuit

## Relevant from recent conversations
- [2026-04-01T10:00:00Z] (user fix-berlin-s1) I just moved to Berlin from NYC last month. Loving it so far.
- ...
```

Five steps, all under one `max_tokens` budget:

1. **Fetch active memories** for `user_id` from Postgres — direct read, no embedding needed.
2. **Embed query** with `text-embedding-3-small`. On transient failure, the facts block still renders (MS-24 degraded mode).
3. **Hybrid retrieval over conversation chunks** — cosine top-30 + Postgres FTS top-30, fused via Reciprocal Rank Fusion (`k=60`, Cormack-Buettcher-Clarke 2009). Replaces vanilla cosine top-k per the API contract.
4. **Hybrid retrieval over memory rows** — same RRF over `kind='memory'` documents, returning a `{memory_id: {rrf, cosine}}` map for query-aware fact ranking.
5. **Noise + per-fact gates** — facts only render when (a) memory RRF or conversation cosine clears the relevance threshold or (b) the query matches a profile-allowlist regex (`tell me about this user` etc.). Inside the block, each individual fact is gated on its own cosine ≥ 0.25 (`MS-32`). Off-topic queries return cold context — noise resistance.

Citations carry `turn_id`, cosine score, and snippet.

---

## Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Web | FastAPI + Pydantic v2 | Async-native, schema validation at boundaries |
| DB | Postgres 16 + pgvector + `tsvector`/FTS | One store for relational + vector + keyword |
| ORM | asyncpg (raw SQL) | Single-file pool, no ORM ceremony |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) | Cheap, ~50 ms, swappable in one file |
| Extraction | OpenAI `gpt-4o-mini` Structured Outputs | Schema-validated JSON, no parse failures |
| Container | Docker Compose | `up -d` boots both services; named volume persists |
| Tests | pytest + httpx (live ASGI + offline pure helpers) | Fast offline floor + live-stack contract suite |

---

## Backing store — why pgvector, not a separate vector DB

One Postgres instance via `asyncpg` covers three access patterns:
relational rows (`turns`, `turn_messages`, `memories`), vector cosine
via HNSW (`vector_cosine_ops`, `m=16`, `ef_construction=64`), and
full-text via `tsvector` + GIN.

**Why this over Qdrant / Weaviate / Kuzu:**

- **Synchronous correctness.** Each `/turns` call writes turn
  metadata, embedded turn-message documents, extracted memories, and
  `kind='memory'` sibling docs in **one transaction**. A two-store
  design (Postgres for memories, vector DB for vectors) cannot give
  this guarantee without a write-ahead log or eventual-consistency
  dance — both ruled out by the brief.
- **One thing to operate.** Single named Docker volume
  (`rag-db-data`) preserves everything. No second backup story, no
  second connection-pool ceiling, no second migration runner.
- **Scale ceiling honest.** asyncpg pool is sized to 10; HNSW handles
  the row counts a developer demo will hit (≤100 RPS comfortably).
  Beyond that, sharding pgvector or moving to a dedicated index is a
  future concern, not a hackathon one (out-of-scope).
- **HNSW is fast enough.** Cosine top-30 over 5–500 docs/user runs in
  single-digit ms; the 60 s `/turns` budget is dominated by the
  OpenAI embed + extract roundtrips, not the DB.

Schema lives in `migrations/`: `001_init` (documents + embeddings),
`003_turns` (turn store), `004_memories` (typed memory
chain), `005_bm25_hybrid` (FTS column + GIN), `006_memory_documents`
(kind='memory' discriminator), `007_canonical_slots` (slot column +
partial unique index for supersession concurrency).

---

## Extraction pipeline

`src/extraction.py` calls `client.chat.completions.parse` with a
Pydantic `ExtractionResponse` schema (OpenAI Structured Outputs). The
model returns JSON validated against the schema server-side — no
custom parser, no `json_object` mode failures, no schema drift between
identical calls. Model: `gpt-4o-mini` with `timeout=30 s` and
`max_retries=0` on the hot path (the SDK default of 2 retries × 30 s
would blow the 60 s `/turns` budget on a 429 burst; per-call retry
is the eval's job, not ours).

**Eight categories covered, all in the prompt with examples drawn
from the v1 fixture:**

| Category | Example keys |
| --- | --- |
| Employment | `employment.company`, `employment.role`, `employment.previous_company` |
| Location | `location.city`, `location.previous_city` |
| Family / pets | `pet.name`, `pet.species`, `pet.age_years`, `family.spouse_name` |
| Preferences | `diet.style`, `beverage.coffee` |
| Opinions | `opinion.<topic>` |
| Allergies | `allergy` |
| Implicit facts | `walking Biscuit` → `pet.name=Biscuit` |
| Corrections | `actually, I meant…` — the corrected fact only |

Every emitted memory carries `type ∈ {fact, preference, opinion,
event}`, `key`, `value`, and `confidence ∈ [0, 1]` (0.95 explicit,
0.85 implied, 0.65 hedged).

**What is deliberately not extracted:**

- Multi-turn corrections that need previous-turn context. The
  extractor sees one turn at a time; cross-turn judging is MS-26's
  canonical-slot supersession layer instead.
- Free-form narrative summarisation. Each memory is a single atomic
  fact, not a paragraph.
- Sensitive data classification (PII tagging). Out of scope for .

**Failure handling.**
`extract_memories_from_turn(messages) → (memories, error_or_none)`
never raises on transient failure. Caught: `RateLimitError`,
`APITimeoutError`, `APIConnectionError`, `BadRequestError`,
`LengthFinishReasonError`, `ContentFilterFinishReasonError`,
`ValidationError`, plus a last-resort `Exception` net. The error
string is persisted in `turns.metadata.extraction_error` and the turn
still commits with `memories=[]`. `AuthenticationError` is the **only
exception that propagates** — it indicates a config bug, not transient
runtime state, and silently degrading would hide a missing API key.

---

## Fact evolution & supersession

The `memories` table is append-only. Updates happen via the
*supersession chain*: a new fact for the same canonical slot flips
the prior row's `active=FALSE` and sets `supersedes=<old.id>`. Old
rows are not deleted — `/users/{id}/memories` returns the full
chain so reviewers can audit history.

**Insert path inside `create_turn`** (MS-18 + MS-26 + MS-27):

| Prior active row for `(user_id, slot)` | Action |
| --- | --- |
| None | Plain insert. `slot` computed via canonical-alias map. |
| Same slot, identical value | Idempotent: `UPDATE memories SET updated_at=NOW()`. No new row. Avoids polluting `## Known facts` with duplicates when the user re-states a fact. |
| Same slot, different value | Mark old row `active=FALSE, updated_at=NOW()`. Insert new row with `supersedes=old.id`. Delete the old `kind='memory'` document so `/recall` doesn't surface stale facts. |

**Canonical slot map (MS-26).** The LLM emits varying keys for the
same semantic slot — `opinion.typescript`, `opinion.typescript.generics`,
`language.favourite` could all describe the same opinion arc.
`_canonicalise_slot(key)` collapses them deterministically:

- 25-entry alias dict for employment / location / diet / allergy /
  beverage variants. Example: `employment.company`, `employer`,
  `current_employer`, `company` → `employment.current_company`.
- Hierarchical opinion collapse: `opinion.<a>.<b>...` → `opinion.<a>`.
  Closes opinion-arc supersession across LLM phrasings.
- Fallback `slot=key` for anything unknown (preserves multi-entity
  slots like `pet.name` / `family.spouse_name`).

**Concurrency safety (MS-27).**
`pg_advisory_xact_lock(hashtextextended($1, 0))` keyed on
`f"{user_id}:{slot}"` serialises read-modify-write across concurrent
`/turns` to the same slot. The partial unique index
`UNIQUE (user_id, slot) WHERE active=TRUE` is the DB backstop — even
if the application logic ever bugs, Postgres rejects a second active
row for the same slot.

**Opinion arcs.** The brief calls these out as harder than clean
overwrites. Current implementation: hierarchical slot collapse handles
across-phrasing supersession; the chain stays linear (love →
annoyed → pragmatic = three rows, two superseded, one active). A
multi-step LLM judge that synthesises arc summaries is documented as
out of scope and tracked for follow-up.

---

## Tradeoffs

| Optimised for | Given up |
| --- | --- |
| One backing store, one transaction, no eventual consistency | LLM-judge canonicalisation — extra latency + non-determinism vs deterministic alias map |
| Synchronous extraction inside `/turns` (60 s budget) | Throughput on bursts — `max_retries=0` on the hot path means a 429 returns zero memories rather than retrying |
| Deterministic `(user_id, slot)` supersession | Cross-encoder reranker (Cohere / local) — extra latency we do not yet have eval evidence to justify (tracked MS-35) |
| query-aware fact ranking + per-fact relevance gate | Explicit query rewriting / multi-hop graph traversal (tracked MS-34) |
| Original prompt, original schema, single-file-per-concern | Bigger framework conveniences (no Celery / Kafka / queue) |
| Hybrid retrieval (cosine + Postgres FTS via RRF) | Field-weighted BM25, IDF-precision tuning (would need `pg_search` / ParadeDB or external index) |

The brief weights iteration history. Every tradeoff above has a
CHANGELOG entry recording why we picked it and a Linear ticket
recording when (or whether) we revisit.

---

## Failure modes

| Failure | Behaviour | Trace |
| --- | --- | --- |
| Missing `OPENAI_API_KEY` | `docker compose up` fails fast on `${OPENAI_API_KEY:?…}` interpolation. Listed in `.env.example`. | docker error log |
| OpenAI 5xx / 429 / DNS / timeout on `/turns` | Persist turn metadata + extracted memories. Skip embedding writes. `turns.metadata.embed_error="rate_limit"` (or `timeout` / `connection` / `bad_request:{status}`). Return 201. | `/admin/turn_index` `embed_failures` count + warn log |
| OpenAI failure on extraction | Extraction returns `(memories=[], error="…")`. Turn still commits with empty memories. `turns.metadata.extraction_error` set. | `/admin/turn_index` `extraction_failures` count |
| Both embed AND extract fail | 503 with `{error:"upstream_unavailable", retryable:true, embed_error, extract_error}`. Eval may retry. | response body |
| OpenAI auth bug (invalid key) | `AuthenticationError` re-raises → 5xx. Loud failure, not silent zero memories. | uvicorn error log |
| `embed_query` failure on `/recall` | `## Known facts` block still renders from Postgres (no embedding needed). Conversation block goes cold; citations empty. 200 response. | `recall.embed_failed` warn log |
| `embed_query` failure with no memories | Fully cold per the API contract: `{context:"", citations:[]}`. | warn log |
| DB pool acquisition fails | FastAPI exception middleware serialises `{error:"internal_server_error", request_id}`. `/health` returns 503 until pool recovers. | request_id in log |
| Restart mid-write | Single transaction guarantees turn + memories + embeddings either all land or none. Verified by `test_restart_persistence`. | transaction log |
| Malformed input | Pydantic v2 returns 422. Unicode oddities (RTL override, emoji, control chars) accepted as payload. Verified by `test_malformed_input_no_crash`. | Pydantic error |
| Concurrent `/turns` to the same canonical slot | Advisory lock serialises; partial unique index is the DB backstop. Verified by `test_concurrent_turns_to_same_slot_no_double_active`. | — |

Five live-stack tests in `tests/contract/test_required_behaviors.py` cover the
graceful-upstream paths (MS-24); a sixth covers concurrency (MS-27).

---

## Project structure

```
memory-service/
├── README.md                   # architecture, backing store, recall, tradeoffs
├── CHANGELOG.md                # iteration history per the API contract (Russian)
├── docker-compose.yml          # db + app service, named volume `rag-db-data`
├── Dockerfile                  # service container; runs `uvicorn src.main:app`
├── src/                        # service code
│   ├── main.py                 # FastAPI lifespan + DB pool + OpenAI client
│   ├── api/
│   │   ├── routes.py           # API contract + admin endpoints
│   │   └── schemas.py          # Pydantic request/response models
│   ├── extraction.py           # gpt-4o-mini Structured Outputs extractor
│   ├── embeddings/openai.py    # text-embedding-3-small (async)
│   ├── db/pool.py              # asyncpg + pgvector codec + migration runner
│   └── obs/logging.py          # structlog JSON
├── migrations/                 # 001_init … 007_canonical_slots
├── fixtures/recall/            # v1 (easy floor) + v2 (stress probes)
├── tests/
│   ├── unit/                   # offline pure-helper tests (no Docker)
│   └── contract/               # live-stack contract tests (RAG_E2E=1)
├── scripts/smoke.sh            # reference smoke flow
└── .env.example                # OPENAI_API_KEY + Postgres knobs
```

---

## How to run tests

Two tiers.

```bash
# Offline floor — 22 unit tests on pure recall helpers. <1s. No Docker, no OpenAI.
.venv/bin/pytest -q tests/

# Full suite — live stack. Requires `docker compose up` + a real OPENAI_API_KEY.
RAG_E2E=1 .venv/bin/pytest -q tests/
```

**Offline (`tests/unit/test_recall_helpers.py`, 22 tests):**
canonical-slot collapse, `## Known facts` parsing, noise gate signals,
provisional-header budget enforcement, query-aware fact sort,
per-fact cosine gate.

**Live (`tests/contract/`, 24 tests):**
- 12 contract-shape tests covering request/response shapes.
- 4 required tests (MS-20): synchronous availability,
  concurrent-session isolation, malformed input no-crash, restart
  persistence.
- 5 graceful-upstream tests (MS-24): `/turns` survives embed blip,
  `/recall` returns facts when embed fails, fully cold when no
  memories, fixture runner aggregates failures, both-fail returns 503.
- 1 query-aware ranking test (MS-23).
- 1 concurrency-canonical-slot guard (MS-27).
- 1 partial-overlap noise gate (MS-32 via v2 fixture).

The restart test invokes `docker compose restart app` via subprocess
and is auto-skipped when the docker CLI is absent on PATH.

### Recall-quality fixtures

```bash
curl -sX POST http://localhost:8080/admin/run_fixture \
  -H 'content-type: application/json' -d '{"name":"v1"}' | jq .aggregate
```

- **v1** — 7 probes / 3 conversations. Floor:
  `recall@1=6/6, mean=0.356`.
- **v2** — 10 probes covering categories (multi-hop linkage,
  keyword-anchored vs distractor, supersession arcs for employer +
  opinion, tight-budget priority assembly, adversarial noise,
  partial-overlap noise, implicit fact). All probes pass:
  `multihop@1=1/1, facts=6/6, forbidden=0, supersession=1/1, noise=1/1`.

---

## What ships

- All seven endpoints with exact shapes + status codes; persistence across restarts.
- LLM extraction (`gpt-4o-mini` Structured Outputs) covering all eight categories with graceful upstream-failure handling.
- One-store data layer (Postgres + pgvector); single transaction per `/turns`.
- Canonical-slot supersession with a partial unique index and `pg_advisory_xact_lock`. Chain visible via `/users/{id}/memories`.
- Hybrid retrieval (cosine + Postgres FTS + RRF). Not vanilla top-k.
- Query-aware fact ranking via sibling `kind='memory'` documents.
- Two-stage noise gate: block-level + per-fact cosine threshold.
- Graceful upstream-failure: embed/extract blip → 201 with metadata flag, never 5xx; both fail → 503 retryable.
- 22 offline + 24 live tests = 46 passed. Two recall-quality fixtures (v1 + v2).

## What does not ship

- Async/background re-embed or re-slot jobs. Out of scope per the API contract.
- LLM-judge canonicalisation. Deterministic slot map is sufficient on the current fixture; see [MS-37 ADRs](https://linear.app/hgtz/issue/MS-37) for revisit criteria.
- Cross-encoder reranker (Cohere `rerank-3` / local cross-encoder). Tracked as [MS-35](https://linear.app/hgtz/issue/MS-35); only meaningful with a `COHERE_API_KEY`.
- Query rewriting / multi-hop subquery decomposition. Tracked as [MS-34](https://linear.app/hgtz/issue/MS-34); current pipeline solves 's verbatim multi-hop example via facts-first assembly but a more general solver is the next ladder rung.

---

## Iteration log

`CHANGELOG.md` (Russian) is the per-decision log — each entry follows
a six-part template (Проблема / Ход мыслей / Рассмотренные варианты /
Причина выбора / Результат / Дальше) and cites the area of the
requirements it addresses. The project weights iteration history — read CHANGELOG for the *why* behind every choice this
README documents structurally.

The internal requirements (referenced as section numbers throughout this README
and CHANGELOG) is private and not committed to this repo.
