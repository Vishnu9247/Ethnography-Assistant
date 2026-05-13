#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run: bash scripts/setup_linux.sh"
  exit 1
fi

source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
