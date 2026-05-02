#!/usr/bin/env bash
# Seed the running stack with fixtures/conversations.json.
#
# Why this script exists: the eventual contract is /turns with
# {session_id, user_id, messages[]}. The current scaffold only has
# /ingest with {documents:[{id,text,metadata}]}. This script bridges
# the two by flattening each turn's messages into one doc per message.
#
# - id        := "<session_id>:<turn_idx>:<msg_idx>"
# - text      := "<role>: <content>"   (role kept inline so retrieval
#                                       sees who said what)
# - metadata  := { user_id, session_id, role, timestamp, ...turn.metadata }
#
# Run:
#   docker compose up -d --build && bash scripts/seed.sh
set -euo pipefail

HOST="${HOST:-http://localhost:8080}"
FIX="${FIX:-fixtures/conversations.json}"

if ! command -v jq >/dev/null; then
    echo "jq is required (apt-get install jq)" >&2
    exit 1
fi

echo "==> waiting for $HOST/health"
until curl -fsS "$HOST/health" >/dev/null 2>&1; do
    sleep 1
done
echo "    ready"

# Build the /ingest payload with jq. One document per message.
PAYLOAD="$(jq '
  {
    documents: [
      .turns[]
      | . as $t
      | ($t.messages | to_entries[]) as $m
      | {
          id: ( $t.session_id + ":" + ($t.timestamp|tostring) + ":" + ($m.key|tostring) ),
          text: ( $m.value.role + ": " + $m.value.content ),
          metadata: ( ($t.metadata // {}) + {
            user_id: $t.user_id,
            session_id: $t.session_id,
            role: $m.value.role,
            timestamp: $t.timestamp
          })
        }
    ]
  }
' "$FIX")"

N="$(echo "$PAYLOAD" | jq '.documents | length')"
echo "==> POST /ingest ($N documents)"

curl -fsS -X POST "$HOST/ingest" \
    -H 'content-type: application/json' \
    -d "$PAYLOAD" | jq .

echo "seed OK"
