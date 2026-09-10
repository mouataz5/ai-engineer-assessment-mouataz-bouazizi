"""A failed source must not look like an empty one, and a superhero route with
no identifiable hero must say so."""
from app.errors import SuperheroTimeout
from app.orchestrator import answer_question
from app.router import Router
from app.superhero import Hero
from app.synthesis import NO_CONTEXT_MESSAGE, NO_HERO_MESSAGE, PARTIAL_FAILURE_MESSAGE
from tests.conftest import FakeLLM, FakeSuperheroClient

BATMAN = Hero(id="70", name="Batman", full_name="Bruce Wayne", powerstats={"strength": "26"})


async def _answer(llm_responses, dataset_search, question, *, heroes=None, hero_exc=None):
    llm = FakeLLM(llm_responses)
    return await answer_question(
        question,
        router=Router(llm),
        dataset=dataset_search,
        superhero=FakeSuperheroClient(heroes=heroes, raise_exc=hero_exc),
        llm=llm,
        top_k=3,
    )


async def test_superhero_failure_is_not_reported_as_empty(dataset_search):
    resp = await _answer(
        [{"route": "superhero", "hero_name": "Batman", "reasoning": "x"}],
        dataset_search,
        "What are Batman's stats?",
        hero_exc=SuperheroTimeout("timed out"),
    )
    assert resp.answer == PARTIAL_FAILURE_MESSAGE
    assert resp.answer != NO_CONTEXT_MESSAGE
    assert any("superhero" in w for w in resp.warnings)
    assert resp.sources == []


async def test_partial_answer_when_one_source_fails(dataset_search):
    # Dataset has hits for a Batman question; superhero lookup errors out.
    resp = await _answer(
        [
            {"route": "both", "hero_name": "Batman", "reasoning": "x"},
            {"answer": "Batman fights the Joker in Gotham.", "used_sources": [0], "grounded": True},
        ],
        dataset_search,
        "How is Batman shown in the movies and what are his stats?",
        hero_exc=SuperheroTimeout("timed out"),
    )
    assert resp.answer == "Batman fights the Joker in Gotham."
    assert resp.sources and resp.sources[0].type == "dataset"
    assert any("superhero" in w for w in resp.warnings)


async def test_superhero_route_with_unidentifiable_hero(dataset_search):
    resp = await _answer(
        [{"route": "superhero", "hero_name": None, "reasoning": "x"}],
        dataset_search,
        "who is the strongest one",  # no name the fallback can latch onto
    )
    assert resp.answer == NO_HERO_MESSAGE
    assert any("identify" in w for w in resp.warnings)


async def test_genuine_empty_result_still_says_not_found(dataset_search):
    resp = await _answer(
        [{"route": "superhero", "hero_name": "Nobodyman", "reasoning": "x"}],
        dataset_search,
        "What are Nobodyman's powers?",
        heroes=[],  # clean miss, no error
    )
    assert resp.answer == NO_CONTEXT_MESSAGE
    assert resp.warnings == []
