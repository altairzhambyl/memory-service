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

2026-05-03
MS-20a-1 — recall-quality fixture v1 + admin runner + UI verdict table — 
Stood up the measurement layer that was missing from iteration story. New `fixtures/recall/v1.json` holds 3 conversations (Berlin/NYC + Notion employment, Biscuit the beagle across two sessions, vegetarian + shellfish allergy + no-coffee) and 7 probes — 6 positive (where-live / current-job / pet-name / pet-implicit / is-vegetarian / shellfish) and 1 noise probe (`Does this user own a horse?` against pet user, threshold 0.35). New `POST /admin/run_fixture` resets every `fix-*` user idempotently (safety guard rejects non-`fix-` user_ids with 400), ingests turns through the real `create_turn` so production write-path runs end-to-end, then probes `/recall` and reports per-probe verdicts + aggregate. Static UI gained card 5 in panel: pill aggregates + colour-coded table (green/red row stripe + PASS/FAIL/noise OK).
Cross-cutting fix: `Dockerfile` was COPYing only `app/` and `migrations/`, so the fixture file 404'd until I added `COPY fixtures` and bind-mounted `./fixtures:/app/fixtures:ro` in `docker-compose.yml` (mirrors the existing static bind-mount so v2/v3 fixture iteration won't need a rebuild). Discovered via the UI: `Error: {"detail":"fixture_not_found:v1"}` on first run.
Result on v1 fixture (vanilla cosine baseline, `text-embedding-3-small`, HNSW): **recall@1 = 6/6 (100%)**, **recall@3 = 6/6 (100%)**, **noise resistance = 1/1**, **mean top-score = 0.356**, elapsed 7690ms (5 turns × 2 messages embedded + 7 probe embeddings, all serial). Per-probe top scores: where-live 0.273, current-job 0.352, pet-name 0.400, pet-implicit 0.318, is-vegetarian 0.412, shellfish 0.379; noise-horse correctly fell to 0.185 (top-hit got-biscuit because dogs and horses share semantic neighbourhood, but well below the 0.35 noise threshold).
Read of the numbers: vanilla cosine sweeps a hand-built short-conversation fixture, but mean 0.356 is still the floor — every score lives in [0.27, 0.41] and any extra noise pushed in by multi-hop / contradiction probes will eat into that margin. This run is the line every future upgrade (MS-17 typed memories → MS-19 priority + hybrid → rerank) must beat on the same fixture. Next iteration: MS-17 — typed structured memories + a parallel `fixtures/extraction/v1.json`.

2026-05-03
MS-17a-i — memories table + read endpoint + demo seed — 
Stood up the structured-memories storage and inspection path so the `GET /users/{user_id}/memories` endpoint stops being a stub. New `migrations/004_memories.sql` adds the `memories` table with columns matching the reference shape exactly (id, user_id, type ∈ {fact, preference, opinion, event}, key, value, confidence ∈ [0,1], source_session, source_turn, created_at, updated_at, supersedes nullable, active boolean) plus a composite index on `(user_id, active, key)` sized for the MS-18 supersession lookup hot path. `supersedes` is a plain TEXT pointer for now — promoting it to a self-referential FK is part of MS-18, not 17a. `GET /users/{user_id}/memories` now reads real rows (active only, ORDER BY updated_at DESC), `DELETE /users/{id}` and `DELETE /sessions/{id}` were extended to drop matching memories in the same transaction so cleanup stays atomic and the isolation story holds.
Demo seeder: new `POST /admin/seed_memories` writes the ideal-extraction rows for the v1 fixture users (`fix-berlin` 5 rows, `fix-pet` 3 rows, `fix-diet` 3 rows). Idempotent — wipes the user's memories first. 404 on any non-`fix-*` user_id; we don't pretend to have ground-truth extractions for arbitrary users. This is a placeholder for MS-17a-ii (real LLM extractor in `/turns`) so the read path + UI table can be exercised before the LLM pipeline lands. Static UI gained card 6 in panel: user_id input, Refresh/Seed buttons, pill counts (rows + type distribution), table showing type/key/value/conf/session/turn/active/updated_at.
Result on inspection probe (after `Run fixture v1` then `Seed demo memories` for fix-berlin): 5 typed rows visible in UI, all `fact`, with the expected location/employment keys; DELETE-then-fixture-rerun cycle confirms memory rows wipe alongside turns. No quality metric for this slice — it's structural; MS-17a-ii will deliver the first extraction-quality number when the LLM extractor replaces the seeder.