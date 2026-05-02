#!/usr/bin/env bash
# Manual A/B smoke for the simple RAG. Run `docker compose up -d` first,
# then this. It ingests three sentences and queries one.
set -euo pipefail

HOST="${HOST:-http://localhost:8080}"

echo "==> /health"
curl -fsS "$HOST/health" | python3 -m json.tool

echo "==> /ingest 3 docs"
curl -fsS -X POST "$HOST/ingest" \
    -H 'content-type: application/json' \
    -d '{
        "documents": [
            {"id":"d1","text":"The capital of France is Paris."},
            {"id":"d2","text":"Bananas are a yellow fruit."},
            {"id":"d3","text":"Mount Everest is the tallest mountain on Earth."}
        ]
    }' | python3 -m json.tool

echo "==> /query 'Where is Paris?'"
curl -fsS -X POST "$HOST/query" \
    -H 'content-type: application/json' \
    -d '{"query":"Where is Paris located?","top_k":3}' | python3 -m json.tool

echo "smoke OK"
