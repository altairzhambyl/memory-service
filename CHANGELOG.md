# Changelog

2026-05-02 8:00pm-9:00pm
Simple RAG
Decided to use simple RAG as a baseline for the project, looking toward expanding the memory system from a simple PostgreSQL database. Two tables (`documents`, `embeddings`), Qwen3-Embedding-0.6B in-process, HNSW cosine, three endpoints (`/health`, `/ingest`, `/query`).

AI API PIVOT - decided to use Groq API instead of Qwen. Benefits - better models, faster inference, lower load on local machine.

Pivoted to OpenAI API since Groq models don't support embeddings for RAG. Need embeddings for comprehensive RAG test.

Got OpenAI API key.

Checking database with mock data. Decided to make a simple chatbot frontend for testing - simulate multiple different users "Alice" and "Bob" to check if the database can distinguish multiple concurrent sessions.

9pm-10pm
Testing docker up/down persistance. Made simple frontend for visualization, improving dev experience.

10pm-11:45pm
Multi-session test harness — 
Alice/Bob harness was scoped only by `user_id`; a single user couldn't hold parallel chats — this didn't mirror the model where one `user_id` owns many `session_id`s. Added `chat_sessions` table and `chat_messages.session_id` FK with cascade delete (migration `002_chat_sessions.sql`), backfilled existing rows into a "Default" session per user, reshaped `/chat`, `/chat/history`, `DELETE /chat/history` to scope by `session_id`, and added `GET/POST /chat/sessions` + `DELETE /chat/sessions/{id}`. Frontend gained a per-user sessions list + "+ New chat" button; switching sessions doesn't bleed messages (manual A/B with two Alice sessions and one Bob session, then `docker compose down && up`, confirmed). `ensure_schema` now applies all `migrations/*.sql` in lexical order so volumes from earlier sessions migrate forward in place.
Side track: added permissive `localhost:*` CORS + `?api=` override in the static page so the UI can be served from `python3 -m http.server 5173` while the backend stays in Docker — frontend iteration is now F5, no rebuild. Production deploy will tighten the regex.
Result: schema applied idempotently on existing volume; cross-session isolation verified by hand. No quality metric — this is harness work, not the API contract. The gap (`/turns`, `/recall`, `/search`, `/users/{user_id}/memories`, `DELETE /sessions/{id}`, `DELETE /users/{id}`) is the next epic.

Alice - любит карбонару и слушает Adele
Bob - любит пиццу и слушает imagine dragons

baseline тест - Боб спросил что любит есть Alice, но модель ответила что не знает. Memory is isolated.

2026-05-02 17:00-17:35
API contract shell — 7/7 endpoints exposed — 
Wrapped the simple-RAG baseline with the surface: `POST /turns` (synchronous persist via new `turns` + `turn_messages` tables, FK cascade), `POST /recall`, `POST /search`, `GET /users/{user_id}/memories` returning the cold shapes (MS-16/17 fill content), and `DELETE /sessions/{session_id}` / `DELETE /users/{user_id}` (idempotent cascade by `session_id` / `user_id`). string `session_id` kept fully separate from the int `chat_sessions.id` of the manual harness. New `tests/contract/test_contract_shapes.py` (9 cases) plus the 3 baseline cases — `RAG_E2E=1 pytest tests/contract -q` → 12 passed. Closes the gap; MS-15 DoD `7/7 contract endpoints exposed` met.

2026-05-03
MS-16a — embed-on-/turns + cleanup-on-DELETE — 
`POST /turns` now embeds each `messages[i].content` via `text-embedding-3-small` and writes the doc into the existing `documents`/`embeddings` tables under `doc_id = "turn:{turn_id}:{idx}"` with provenance metadata (`kind=turn_message, turn_id, session_id, user_id, role, idx, ts`). Embed-then-transaction ordering keeps a network blip from leaving half-state; the DB writes commit atomically (synchronous correctness). `DELETE /sessions/{id}` and `DELETE /users/{id}` were extended to drop the matching `documents` rows in the same transaction, since those rows have no FK back to `turns` — without this, orphan vectors would have leaked into future sessions. Read-only `GET /admin/turn_index` added so the static UI can A/B verify writes/cleanup without psql. Manual A/B in panel of static UI: 2 docs / 2 embeddings / 2 turn_messages after the Berlin payload; counts drop to 0 after `DELETE /users/user-1`. No retrieval logic yet — `/recall` still empty stub.

2026-05-03
MS-16b — raw recall baseline — 
`POST /recall` and `POST /search` now run cosine top-k (HNSW) over the indexed turn-message vectors, scoped by `user_id`. `/recall` formats the response as a "## Relevant from recent conversations" section under an approximate token budget (chars/4); citations carry `turn_id` + cosine-similarity score + a 240-char snippet. `/search` returns structured `SearchHit` results, no prose. Cold semantics preserved per the API contract: missing user_id, empty store, or zero matches → 200 with empty body, never 500.
Cross-session sharing for the same `user_id` is intentional and now documented in README; `session_id` from the request is captured in citations for provenance, not used as a filter on `/recall`. Per this is the **vanilla cosine baseline** — explicitly the floor, not the target. Stable-fact priority assembly, "## Known facts about this user" section, hybrid retrieval, and rerank are tracked separately (MS-17 / MS-19) and will compound on top.
Result on smoke probe ("Where does this user live?" against the Berlin/NYC turn, user-1 scope): top score **0.273** on the user message, citations=2 (both messages of the turn), ~70 tokens used out of 512 budget; `context` includes both "Berlin" and "NYC" via the snippet — measured in panel after rebuild. Anonymous (`user_id: null`) and cold-after-DELETE paths both return empty context as expected. Static UI gained metric pills on the recall card (top score / citation count / approx tokens used) so the score is visible without inspecting JSON.
Note on the 0.273 floor: `text-embedding-3-small` cosines for short query/message pairs sit lower than my pre-build guess (~0.5). This is the floor signal — it's enough to surface the right turn for this fixture but it's also exactly why hybrid retrieval / rerank (MS-19) need to land before release. Future CHANGELOG entries should beat 0.273 on this same probe; if they don't, we picked the wrong upgrade.