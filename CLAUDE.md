# Coding Supervisor — Higgsfield AI Engineering Challenge

This file is the canonical Technical Requirements for the build. Re-read it
at the start of every session. The §-numbers below match the original
challenge brief; cite them in commits, CHANGELOG entries, and review
comments (e.g. `feat(recall): hybrid + RRF — addresses §10, §6 v3`).

If anything in code conflicts with this document, this document wins.

---

## §0. Working Agreement (how we collaborate)

1. **Engineer first, not over-performer.** Every line of code must be
   explainable in one sentence. Ship the obvious solution, prove it
   works, then iterate.
2. **Fixate every step.** Each sub-step is a discrete, named, reviewable
   unit. No "and while we're here" expansions.
3. **Manageable & understood.** A sub-step is too big if the user can't
   review it in one sitting. If in doubt, split it.
4. **Manual A/B test every step.** After every sub-step, the user runs
   a concrete command (curl, pytest, docker compose) and observes the
   expected output. Claude provides commands AND expected output.
5. **No skipped verification.** "Looks right" is not signoff. Move on
   only after the user confirms the test passed.
6. **Pause for signoff.** After each sub-step, report what changed,
   give the verification commands, and **stop**.
7. **Cite the spec.** Every CHANGELOG entry and non-trivial commit
   references the §-section it addresses.

---

## §1. Mission

Build a **memory service** for an AI agent. The service ingests
conversation turns, persists them, extracts structured knowledge, and
answers recall queries that decide what context the agent sees on the
next turn.

- **Role:** AI Engineer (mid-level)
- **Time-box:** 2 days of focused work
- **Deliverable:** A reproducible Git repository with a Dockerized
  memory service that conforms to §3.
- **Hosting:** Higgsfield hosts it themselves and runs a private eval.
- **Submission:** https://higgsfieldcareers.typeform.com/to/HoCFpdJC

Full design freedom over: backing store, extraction strategy, recall
pipeline, language. The brief explicitly says raw-message-in /
vector-DB-out is **not** extraction, and **vanilla cosine-top-k will
not score well**.

---

## §2. What "ship" means

A single Docker-deployable service that:

1. Conforms to the HTTP contract in §3.
2. Persists data across container restarts via a Docker volume.
3. Comes up with `docker compose up` — no manual setup steps.
4. Has internal tests (incl. a small recall-quality fixture — §7).
5. Ships a `CHANGELOG.md` documenting iteration history (§6).
6. Ships a `README.md` covering architecture, store choice, recall
   strategy.

---

## §3. HTTP Contract (normative)

Auth: optional `Authorization: Bearer <token>`. If we need one, we
require `MEMORY_AUTH_TOKEN`; if we don't, we ignore the header.

### `GET /health`
Liveness/readiness probe. **200** when ready.

### `POST /turns`
Write a completed conversation turn. Service must persist, run
extraction, and return when done. Eval harness uses a **60-second
timeout** for this endpoint — take as long as needed for extraction,
embedding, indexing. **Don't waste time on async orchestration; focus
on extraction quality.**

Request:
```json
{
  "session_id": "string",
  "user_id": "string | null",
  "messages": [
    { "role": "user", "content": "string" },
    { "role": "assistant", "content": "string" },
    { "role": "tool", "name": "string | null", "content": "string" }
  ],
  "timestamp": "ISO-8601 string",
  "metadata": { "...": "..." }
}
```

A turn may contain **one or more** messages — user+assistant is
simplest, multi-message including tool calls is valid. Handle both.

Response: **201 Created**, `{ "id": "string" }`. After this returns,
the turn and its extracted memories **must** be queryable via
`/recall` and `/users/{user_id}/memories`. **No eventual consistency.**

### `POST /recall` — primary eval signal
Return relevant context for the upcoming agent turn. Most of the
eval scores this endpoint.

Request:
```json
{
  "query": "string",
  "session_id": "string",
  "user_id": "string | null",
  "max_tokens": 1024
}
```

Response:
```json
{
  "context": "string",
  "citations": [
    { "turn_id": "string", "score": 0.0, "snippet": "string" }
  ]
}
```

Behavior:
- `context` is the formatted text injected into the agent's prompt —
  readable to a frozen LLM. Format reference:

```
## Known facts about this user
- Works at Notion as a PM (updated 2025-03-15; previously at Stripe as an engineer)
- Vegetarian, allergic to shellfish
- Has a dog named Biscuit
- Prefers concise, direct answers

## Relevant from recent conversations
- [2025-03-10] User was debugging a React performance issue with excessive re-renders in a dashboard component
- [2025-03-14] User mentioned preparing for a system design interview at a FAANG company
```

- Respect `max_tokens` (approximate is fine; don't blow past it by 2×).
- Tight-budget priority: **stable user facts → query-relevant memories
  → recent context.** This priority logic is a graded design decision —
  defend it in the README.
- Cold session: return **200** with `{"context": "", "citations": []}`.
  Never error.

### `POST /search`
Explicit search invoked by an agent tool call. Structured results, not
prose.

Request:
```json
{
  "query": "string",
  "session_id": "string | null",
  "user_id": "string | null",
  "limit": 10
}
```

Response:
```json
{
  "results": [
    {
      "content": "string",
      "score": 0.0,
      "session_id": "string",
      "timestamp": "ISO-8601",
      "metadata": {}
    }
  ]
}
```

### `GET /users/{user_id}/memories`
Inspection endpoint. Reviewers will call this to see what got
extracted. Reference shape (exact schema is up to you, but it must be
**structured**, not raw chunks):

```json
{
  "memories": [
    {
      "id": "string",
      "type": "fact | preference | opinion | event",
      "key": "string",
      "value": "string",
      "confidence": 0.0,
      "source_session": "string",
      "source_turn": "string",
      "created_at": "ISO-8601",
      "updated_at": "ISO-8601",
      "supersedes": "string | null",
      "active": true
    }
  ]
}
```

### `DELETE /sessions/{session_id}`
Eval cleanup. Delete all data for the session. **204 No Content.**

### `DELETE /users/{user_id}`
Eval cleanup. Delete all data (memories, turns, sessions) for the
user. **204 No Content.**

### Extras
You may add admin/metrics/rebuild endpoints. The seven above are the
contract — endpoint paths, request/response shapes, and status codes
are exact.

---

## §4. Hard Problems (where points are won/lost)

### 4.1 Fact evolution and contradiction handling
"I work at Stripe" (s1) → "I just started at Notion" (s3). Service
must:
- Detect they're about the same topic (employment).
- Mark old fact superseded — **not deleted**.
- `/recall` returns the current fact, not the stale one.
- `/users/{user_id}/memories` preserves the supersession chain.

Applies to **all mutable facts**: location, job, relationships,
opinions, preferences. Append-only with no update logic = poor score.

Harder: opinion arcs. "I love TS" → "TS generics annoy me" → "TS for
big projects, Python for scripts." Not a clean overwrite — document
how the system handles arcs even if implementation is partial.

### 4.2 Extraction, not just storage
Extract structured knowledge from raw turns. Minimum coverage:
- Personal facts (employment, location, family, pets)
- Preferences and opinions
- Corrections ("actually, I meant…", "sorry, not X — Y")
- Implicit facts ("walking Biscuit this morning" → has pet Biscuit)

Reviewers will inspect `/users/{user_id}/memories`. Raw message chunks
in there = red flag.

### 4.3 Context assembly under token budget
`/recall` must fit `max_tokens`. When tight, triage. Priority logic is
a core design decision — defend it in README.

---

## §5. Hard Constraints

- **Persistence.** Survives `docker compose down && up`. Named volume.
- **Concurrent sessions.** Multiple `session_id`s active at once must
  not bleed — unless cross-session sharing for the same `user_id` is
  intentional, in which case **document it in README**.
- **Synchronous correctness.** After `POST /turns` returns, data is
  immediately available via `/recall` and
  `/users/{user_id}/memories`. **No eventual consistency.**
- **Recall budget.** `POST /recall` returns within reasonable time.
  Slow pipeline → document why and the optimization plan.
- **Resilience.** No crashes on malformed input, oversized payloads,
  or unicode oddities. Errors are 4xx/5xx, never crashes.
- **LLM usage encouraged.** Any provider (OpenAI, Anthropic, Ollama,
  …). Document models in README. List required keys in `.env.example`.

---

## §6. CHANGELOG.md — most-important deliverable

One entry per significant design iteration: **what you tried, what you
observed, why you changed direction.** A mediocre final score with a
thoughtful 5-entry CHANGELOG **beats** a higher score with no
iteration history.

Reference template the brief uses:

```
## v3 — Hybrid retrieval with reciprocal rank fusion

**What changed:** Added BM25 alongside embedding search and fused results with RRF.

**Why:** Pure embedding search was missing keyword-heavy queries like "what's their
dog's name?" where exact token match matters more than semantic similarity.
Noticed this in my test fixture — queries 4, 7, 11 were all keyword-dependent.

**Result:** Self-eval recall improved from 0.52 to 0.64. Precision stayed flat.
Latency increased ~40ms due to double retrieval — acceptable.

**Next:** Contradiction handling is still broken. "Moved to Berlin" and
"lives in NYC" both come back as active. Need to add supersession logic.
```

Required minimum: **4+ entries with metrics at each step** (per §10).

---

## §7. Testing & Self-Eval

### Required: contract tests
- Contract roundtrip — write a turn, recall it, shape verified.
- Restart persistence — write, restart, recall; data survives.
- Concurrent sessions — two sessions with different users don't bleed.
- Malformed input — bad JSON, missing fields, unicode → 4xx, no crash.

### Required: recall quality fixture
`fixtures/` contains 3–5 scripted conversations + probe queries with
expected facts. Tests ingest the conversations, run probes against
`/recall`, report a basic quality metric (even "X of Y expected facts
appeared in context"). **Build this fixture early; run it after every
change.** Brief explicitly calls this out as a quality predictor.

### Provided: smoke test (must pass)
```bash
# after docker compose up
curl -s http://localhost:8080/health | jq .

curl -X POST http://localhost:8080/turns \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "smoke-1",
    "user_id": "user-1",
    "messages": [
      {"role": "user", "content": "I just moved to Berlin from NYC last month. Loving it so far."},
      {"role": "assistant", "content": "That sounds exciting! Berlin is a great city. How are you settling in?"}
    ],
    "timestamp": "2025-03-15T10:30:00Z",
    "metadata": {}
  }'

curl -X POST http://localhost:8080/recall \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Where does this user live?",
    "session_id": "smoke-2",
    "user_id": "user-1",
    "max_tokens": 512
  }'
# should mention Berlin, ideally note the move from NYC

curl http://localhost:8080/users/user-1/memories | jq .
# should show structured memories, not raw message text
```

If shapes match and smoke test passes, the service runs against eval.

---

## §8. Setup the eval will run (clean machine)

```bash
git clone <your repo> memory-service
cd memory-service
docker compose up -d
# wait for health
until curl -sf http://localhost:8080/health; do sleep 1; done
# eval now points at http://localhost:8080
```

- Default port **8080** (or document the override clearly).
- Port exposed via `docker-compose.yml`.
- **No** `pip install`, **no** `npm run setup` outside the container.
- API keys → `.env.example`. Higgsfield supplies values at eval time.

---

## §9. How it will be tested

### A. Automated private eval — primary
Categories (fixtures + thresholds + judge prompts are private):
- **Recall quality** — does `/recall` surface facts a follow-up
  depends on. **Primary signal.**
- **Fact evolution** — supersession works; current fact returned;
  history preserved in memories endpoint.
- **Multi-hop recall** — "What city does the user with the dog named
  Biscuit live in?" Naive top-k fails.
- **Noise resistance** — queries on undiscussed topics return empty
  context, not hallucinations.
- **Extraction quality** — typed, structured, implicit facts
  captured, corrections handled.
- **Persistence across restarts.**
- **Cross-session scoping** (or documented sharing).
- **Robustness** — malformed input, missing auth, slow/empty store,
  restart mid-write. Stays up.
- **Correctness** — synchronous availability after `/turns`.
- **Contract compliance** — endpoints, shapes, status codes.

### B. Human architecture review
We read code, tests, CHANGELOG, README. Judged on: sound architecture,
justified store choice, real (not faux) extraction, deliberate recall
pipeline, genuine iteration in CHANGELOG, clean and tested code,
6-month maintainability. Be ready to defend every choice in a 30-min
follow-up.

---

## §10. What "Excellent" looks like (target list)

- Contract is **exact** — endpoints, shapes, status codes.
- Memories are **typed + structured** with confidence + provenance.
- Fact evolution: contradictions detected, old facts superseded,
  history preserved.
- Recall ranking is **deliberate** — embedding + reranking, hybrid,
  graph, multi-hop, query rewriting. Not vanilla cosine-top-k.
- Context assembly has **explicit priority logic** under token
  budget. Defended in README.
- `/turns` is **synchronous** — no race, no eventual consistency.
- `/recall` returns well-formatted context within budget.
- Restart is invisible to clients.
- Graceful degradation on failure (no crashes, sensible errors).
- Tests cover contract roundtrip, restart, concurrent sessions,
  malformed input, recall quality on self-built fixture.
- **CHANGELOG has 4+ entries with metrics** showing real iteration.
- README walks a reviewer to understanding in 5 minutes.
- `/users/{user_id}/memories` is clean and inspectable.

---

## §11. Originality

Reading mem0 / hindsight / honcho / mnemonic-agents / etc. for
inspiration is fine. **The submission must be our own design.** Don't
lift another project's API shape or recall pipeline. Resemblance has
to survive a line-by-line interview.

---

## §12. Out of Scope

- No agent-side code.
- No UI.
- No multi-tenant production-readiness.
- No horizontal scaling proofs (mention if it happens; don't build).
- No migration story. Single schema version is fine.

---

## §13. Required repo layout

```
memory-service/
├── README.md              # architecture, store, recall, tradeoffs
├── CHANGELOG.md           # iteration history (§6)
├── docker-compose.yml     # service + deps
├── Dockerfile             # the service container
├── src/                   # code (any language)
├── tests/                 # service-internal tests (§7)
├── fixtures/              # self-eval data
└── .env.example           # optional env vars
```

(Our current layout uses `app/` instead of `src/` per Python
convention — the brief is permissive on language layout, but every
other element above is required.)

---

## §14. README.md must contain

1. **Architecture** — diagram (ASCII / Mermaid / image) + 1–2
   paragraphs.
2. **Backing store choice** — what and why.
3. **Extraction pipeline** — how raw turns become structured
   memories. What is extracted, how, what is missed and why.
4. **Recall strategy** — `/recall` end-to-end. Ranking. Token-budget
   handling. Priority logic when budget is tight.
5. **Fact evolution** — contradictions, corrections, opinion changes.
6. **Tradeoffs** — what was optimized for, what was given up.
7. **Failure modes** — no data, slow disk, missing API keys.
8. **How to run the tests.**

---

## §15. Current build snapshot (mutable; supervisor sections above are immutable)

Current scope: **simple pgvector RAG** with **OpenAI
text-embedding-3-small** as the embedding source. Two tables, three
endpoints (`/health`, `/ingest`, `/query` internal naming → must be
wrapped to expose §3 contract before submission), no auth, no
extraction, no supersession, no rerank. We grow features only when an
A/B test demands it.

Embedding-source pivots so far: Qwen3-Embedding-0.6B in-process (slow
build, parked) → Groq (no embeddings endpoint exists, blocked) →
**OpenAI text-embedding-3-small** (current). Groq is reserved for any
future LLM step (extraction, generation) since it has no embedding
model.

> **⚠️ Gap vs. spec:** the current scaffold exposes `/ingest` +
> `/query`, not the §3 contract (`/turns`, `/recall`, `/search`,
> `/users/{user_id}/memories`, `DELETE /sessions/{id}`,
> `DELETE /users/{id}`). Closing this gap is on the critical path
> before any extraction work is judged.

The earlier 9-epic memory-service plan (in `../research/SYNTHESIS.md`,
parked) is useful context for the upgrade path but is not the current
build.

### CHANGELOG entry format we use

```
YYYY-MM-DD HH:MM-HH:MM
Title — §<spec section>
One or two short, direct lines about what was decided/built and why.
Result/metric if measured.
```

One block per work session. The §-citation is mandatory; metric is
mandatory when a quality-affecting change is made (per §10).

---

## §16. Questions

If a question materially blocks progress, **ask the user**. Otherwise
derive from §3 and design freely.
