#!/usr/bin/env bash
#
# One command to run the whole thing:
#
#     ./run.sh
#
# Creates the virtualenv (first run), installs dependencies (first run),
# then launches the dashboard. Synthetic data and the risk model are generated
# automatically on first load — no separate setup steps.
#
set -euo pipefail
cd "$(dirname "$0")"

PY=python3.11
command -v "$PY" >/dev/null 2>&1 || PY=python3

if [ ! -d .venv ]; then
  echo "▸ Creating virtual environment…"
  "$PY" -m venv .venv
fi

if ! .venv/bin/python -c "import streamlit" >/dev/null 2>&1; then
  echo "▸ Installing dependencies (first run only, ~1–2 min)…"
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
fi

echo "▸ Launching TenantEase AI Intelligence Engine…"
exec .venv/bin/streamlit run app.py
