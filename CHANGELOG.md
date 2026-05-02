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