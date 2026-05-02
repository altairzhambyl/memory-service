# Simple RAG — pgvector + OpenAI embeddings

A minimal retrieval service. You POST documents, you POST a query, you
get the top-k most similar documents back. That's it. We'll grow
features on top of this baseline only when an A/B test demands it.

## What's inside

```
client ──HTTP──▶ FastAPI ──▶ OpenAI text-embedding-3-small (cloud)
                     │
                     └──asyncpg──▶ Postgres 16 + pgvector  (HNSW cosine)
```

Two tables:

| Table        | What it holds                                             |
| ------------ | --------------------------------------------------------- |
| `documents`  | `id`, `text`, `metadata`, `created_at`                    |
| `embeddings` | `doc_id` (FK), `embedding vector(1536)`, `model`, `created_at` |

One HNSW index on `embeddings.embedding` with `vector_cosine_ops`.

## Endpoints

| Method | Path      | Body                                                              | Returns |
| ------ | --------- | ----------------------------------------------------------------- | ------- |
| GET    | `/health` | —                                                                 | `{"status":"ok"}` |
| POST   | `/ingest` | `{"documents":[{"id":"...","text":"...","metadata":{...}}, ...]}` | `201 {"ingested": N}` |
| POST   | `/query`  | `{"query":"...","top_k":5}`                                       | `{"hits":[{"id","text","score","metadata"}, ...]}` |

`score` is cosine similarity on L2-normalized vectors, so it lives in
roughly `[0, 1]`. Higher = more relevant.

## Run it

```bash
cp .env.example .env               # then put your OPENAI_API_KEY in .env
docker compose up -d --build       # ~30s build — no local model
curl http://localhost:8080/health  # {"status":"ok"}
bash scripts/smoke.sh              # ingest + query end-to-end
```

## Manual persistence UI

Open `http://localhost:8080/` after `docker compose up`. The page is a
small Alice/Bob chat harness for checking that Postgres data survives
`docker compose down && docker compose up`.

Each user can have **multiple chats (sessions)**. The sidebar shows a
"+ New chat" button and the list of that user's existing sessions; the
composer always writes into the selected session. The schema mirrors the
API contract shape so we can swap the harness for the real `/turns` flow
later without throwing the UI away.

| Table           | Holds                                                                  |
| --------------- | ---------------------------------------------------------------------- |
| `chat_sessions` | `id`, `user_id`, `title`, `created_at` — one row per chat              |
| `chat_messages` | `id`, `session_id` (FK → cascade), `user_id`, `role`, `content`, ...   |

Endpoints driving the UI:

| Method | Path                          | Body / Query                                  |
| ------ | ----------------------------- | --------------------------------------------- |
| GET    | `/chat/sessions`              | `?user_id=alice`                              |
| POST   | `/chat/sessions`              | `{"user_id":"alice","title":"trip planning"}` |
| DELETE | `/chat/sessions/{id}`         | —                                             |
| GET    | `/chat/history`               | `?session_id=42`                              |
| POST   | `/chat`                       | `{"session_id":42,"message":"..."}`           |
| DELETE | `/chat/history`               | `?session_id=42` (clears messages, keeps session) |
| GET    | `/chat/all`                   | diagnostic dump across all users              |

Data lives in the named Docker volume `rag-db-data`. This UI is a
development aid, not the memory-service contract.

### Frontend dev workflow (no rebuilds)

Backend in Docker, frontend served from any static server on its own
port. CORS allows any `localhost:*` origin so the API is reachable.

Terminal 1 — backend + DB:

```bash
docker compose up -d --build
```

Terminal 2 — frontend with hot reload (F5):

```bash
cd app/static
python3 -m http.server 5173
# open http://localhost:5173/?api=http://localhost:8080 once;
# the ?api= value is cached in localStorage afterwards.
```

Edit `app/static/index.html` → F5 in the browser. No `--build`, no
`docker compose restart`. For Python edits, `docker compose restart app`
(~2s) or run `uvicorn app.main:app --reload --port 8080` locally
against the Dockerized DB.

## Why these choices

- **OpenAI `text-embedding-3-small`** — 1536-dim, cheap ($0.02/1M tokens),
  fast (~50ms per request), no local model to download or update. We
  isolate the call in `app/embeddings/openai.py` so swapping providers
  later (Voyage, Cohere, local fastembed) is a one-file change.
- **HNSW (m=16, ef_construction=64)** — pgvector defaults; right for the
  row counts a developer demo will hit. Tune later if needed.
- **Single transaction per `/ingest`** — `documents` and `embeddings` go
  in together so `/query` never sees a half-written row.
- **No `/health` API call** — readiness only checks the DB. Hitting
  OpenAI on every health probe would burn money for no signal; the API
  call is exercised on first `/ingest` or `/query`.

## Swap to a bigger model

1. `EMBEDDING_MODEL=text-embedding-3-large` in `.env`
2. `vector(3072)` in `migrations/001_init.sql`
3. `docker compose down -v && docker compose up -d --build` (resets the
   volume because the column type changed)

## Tests

```bash
make test         # unit only — fast
make test-e2e     # against the live stack on :8080
make smoke        # human-readable end-to-end with curl
```

## Repo layout

```
memory-service/
├── app/
│   ├── main.py             # FastAPI lifespan + DB pool + OpenAI client
│   ├── api/                # routes, schemas, request-id middleware
│   ├── db/pool.py          # asyncpg + pgvector codec
│   ├── embeddings/openai.py # embed_documents / embed_query (async)
│   └── obs/logging.py      # structlog JSON logging
├── migrations/001_init.sql # documents + embeddings + HNSW
├── tests/contract/         # /health /ingest /query shape tests
├── scripts/smoke.sh        # the demo curl flow
├── docker-compose.yml      # db (pgvector/pgvector:pg16) + app
└── Dockerfile              # single-stage; no local model download
```

The longer architectural research that led here lives one level up at
`../research/{SYNTHESIS,architecture-report,agentic-memory-internals}.md` —
not load-bearing for the current code, kept for context.
