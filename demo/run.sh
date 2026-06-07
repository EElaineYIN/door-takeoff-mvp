#!/usr/bin/env bash
# Launch the Door Takeoff demo.
#   ./demo/run.sh            -> http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8000}"
echo "Door Takeoff demo  ->  http://127.0.0.1:${PORT}"
exec .venv/bin/uvicorn demo.backend.main:app --reload --port "${PORT}"
