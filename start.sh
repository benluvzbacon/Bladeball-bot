#!/usr/bin/env sh
# BladeBot launcher for macOS / Linux - PRACTICE USE ONLY (see README.md)
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -x .venv/bin/python ]; then
  echo "Creating a virtual environment and installing requirements..."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m bladebot "$@"
