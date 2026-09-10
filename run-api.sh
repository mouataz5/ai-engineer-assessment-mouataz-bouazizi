#!/usr/bin/env bash
# Start the API. Run this in its own terminal tab and leave it running.
set -e
cd "$(dirname "$0")"
.venv/bin/python -m scripts.build_index
exec .venv/bin/python -m uvicorn app.main:app --reload
