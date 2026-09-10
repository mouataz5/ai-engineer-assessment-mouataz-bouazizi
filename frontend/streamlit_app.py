"""Minimal Streamlit UI for the /ask endpoint.

Run the API first (uvicorn app.main:app), then:  streamlit run frontend/streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
ASK_TIMEOUT = 60

ROUTE_LABELS = {
    "dataset": "🎬 Movie dataset",
    "superhero": "🦸 Superhero API",
    "both": "🎬 + 🦸 Both sources",
    "neither": "🚫 Neither",
}

st.set_page_config(page_title="Ask - movies & superheroes", page_icon="🦸")
st.title("🦸 Ask — movies & superheroes")
st.caption(f"Talks to `POST {API_URL}/ask`. Start the API before asking.")

with st.sidebar:
    st.subheader("Try these")
    st.markdown(
        "- What happens in the movie Inception?\n"
        "- What are Wonder Woman's power stats?\n"
        "- How does the movie portray Batman, and what are his real power stats?\n"
        "- What's the capital of France?"
    )
    if st.button("Clear conversation"):
        st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No sources — the model did not ground its answer in retrieved context.")
        return
    st.markdown("**Sources**")
    for s in sources:
        if s["type"] == "dataset":
            st.markdown(f"- 🎬 **{s.get('title') or s['ref']}** · dataset `{s['ref']}`")
        else:
            st.markdown(f"- 🦸 **{s['ref']}** · Superhero API id `{s.get('id')}`")


# Replay history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            st.error(turn["error"])
        else:
            st.write(turn["answer"])
            cols = st.columns(2)
            cols[0].caption(f"Route: {ROUTE_LABELS.get(turn['route'], turn['route'])}")
            if turn.get("router_degraded"):
                cols[1].caption("⚠️ router degraded (LLM unavailable, queried everything)")
            render_sources(turn.get("sources", []))

question = st.chat_input("Ask about a movie or a superhero…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API_URL}/ask", json={"question": question}, timeout=ASK_TIMEOUT
                )
            except requests.RequestException as exc:
                err = f"Could not reach the API at {API_URL}: {exc}"
                st.error(err)
                st.session_state.history.append({"question": question, "error": err})
                st.stop()

        if resp.status_code == 200:
            data = resp.json()
            st.write(data["answer"])
            cols = st.columns(2)
            cols[0].caption(f"Route: {ROUTE_LABELS.get(data['route'], data['route'])}")
            if data.get("router_degraded"):
                cols[1].caption("⚠️ router degraded (LLM unavailable, queried everything)")
            render_sources(data.get("sources", []))
            st.session_state.history.append({"question": question, **data})
        else:
            detail = resp.json().get("detail", resp.text)
            err = f"API returned {resp.status_code}: {detail}"
            st.error(err)
            st.session_state.history.append({"question": question, "error": err})
