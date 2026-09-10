import pytest
from fastapi.testclient import TestClient

from app.deps import get_dataset, get_llm, get_router, get_superhero
from app.errors import LLMError
from app.main import app
from app.router import Router
from tests.conftest import FakeLLM, FakeSuperheroClient

ROUTER_DATASET = {"route": "dataset", "hero_name": None, "reasoning": "movie q"}
SYNTH_OK = {"answer": "It is about a heist.", "used_sources": [0], "grounded": True}


@pytest.fixture
def make_client(dataset_search):
    def _make(llm_responses=None, llm_exc=None, heroes=None):
        llm = FakeLLM(llm_responses, raise_exc=llm_exc)
        superhero = FakeSuperheroClient(heroes=heroes)
        app.dependency_overrides[get_llm] = lambda: llm
        app.dependency_overrides[get_router] = lambda: Router(llm)
        app.dependency_overrides[get_dataset] = lambda: dataset_search
        app.dependency_overrides[get_superhero] = lambda: superhero
        return TestClient(app), llm, superhero

    yield _make
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "ab"},
        {"question": "x" * 1001},
        {},
        {"question": 123},
    ],
)
def test_invalid_input_returns_422(make_client, payload):
    client, _, _ = make_client()
    resp = client.post("/ask", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert isinstance(body["detail"], list)


def test_happy_path_dataset_question(make_client):
    client, llm, _ = make_client(llm_responses=[ROUTER_DATASET, SYNTH_OK])
    resp = client.post("/ask", json={"question": "What is Inception about?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "dataset"
    assert body["answer"] == "It is about a heist."
    assert body["sources"] and body["sources"][0]["type"] == "dataset"
    assert body["router_degraded"] is False
    assert len(llm.calls) == 2  # router + synthesis


def test_llm_failure_surfaces_as_503(make_client):
    # Every LLM call raises: the router degrades to "both" on its own, then the
    # synthesis call raises LLMError, which the exception handler turns into 503.
    client, _, _ = make_client(llm_exc=LLMError("down"))
    resp = client.post("/ask", json={"question": "Tell me about Gotham"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "LLM service unavailable"


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "ok"}
