#!/usr/bin/env bash
# Thin wrapper around scripts/mock.py — runs the 200-test parametric
# validation harness against `docker compose up`. See scripts/mock.py
# for the full test matrix and per-test predicted labels.
#
# Pass through any flags: `bash scripts/mock.sh --category §3 --filter noise`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/mock.py" "$@"
