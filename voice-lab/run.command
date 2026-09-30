#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Сначала выполните установку из README.md'
  exit 1
fi
export HF_HOME="$PWD/data/models"
export HF_HUB_DISABLE_TELEMETRY=1
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8765
