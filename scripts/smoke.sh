#!/usr/bin/env bash
# §3 contract smoke. Runs the exact flow from CLAUDE.md §7:
#   /health → POST /turns → POST /recall → GET /users/{id}/memories → DELETE /users/{id}.
# Run `docker compose up -d` first; this script must be runnable on a
# clean machine per §8 (eval boots the stack and points at it).
#
# Override the target via HOST=... if the service runs elsewhere.
set -euo pipefail

HOST="${HOST:-http://localhost:8080}"
USER_ID="smoke-user-1"
SESSION_ID="smoke-1"

echo "==> GET /health"
curl -fsS "$HOST/health" | python3 -m json.tool

echo
echo "==> POST /turns — Berlin/NYC turn (§7 reference payload)"
curl -fsS -X POST "$HOST/turns" \
    -H 'content-type: application/json' \
    -d "$(cat <<JSON
{
    "session_id": "$SESSION_ID",
    "user_id": "$USER_ID",
    "messages": [
        {"role": "user", "content": "I just moved to Berlin from NYC last month. Loving it so far."},
        {"role": "assistant", "content": "That sounds exciting! Berlin is a great city. How are you settling in?"}
    ],
    "timestamp": "2026-04-01T10:30:00Z",
    "metadata": {}
}
JSON
)" | python3 -m json.tool

echo
echo "==> POST /recall — 'Where does this user live?'"
echo "    Should mention Berlin via the ## Known facts block (§3 reference shape)."
curl -fsS -X POST "$HOST/recall" \
    -H 'content-type: application/json' \
    -d "$(cat <<JSON
{
    "query": "Where does this user live?",
    "session_id": "smoke-2",
    "user_id": "$USER_ID",
    "max_tokens": 512
}
JSON
)" | python3 -m json.tool

echo
echo "==> GET /users/$USER_ID/memories"
echo "    Reviewer expects typed structured rows (key/value/confidence/active),"
echo "    not raw message chunks (§4.2 red flag)."
curl -fsS "$HOST/users/$USER_ID/memories" | python3 -m json.tool

echo
echo "==> DELETE /users/$USER_ID — eval cleanup (§3 204)"
curl -fsS -X DELETE -o /dev/null -w '%{http_code}\n' "$HOST/users/$USER_ID"

echo
echo "smoke OK"
