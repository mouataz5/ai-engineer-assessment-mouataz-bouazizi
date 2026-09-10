#!/usr/bin/env bash
# Start the Streamlit chat UI. Needs run-api.sh already running in another tab.
set -e
cd "$(dirname "$0")"
exec .venv/bin/python -m streamlit run frontend/streamlit_app.py
