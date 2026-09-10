import pytest

from app.router import Router, guess_hero_name
from tests.conftest import FakeLLM


async def _decide(response=None, raise_exc=None, question="some question"):
    llm = FakeLLM([response] if response else None, raise_exc=raise_exc)
    return await Router(llm).decide(question)


@pytest.mark.parametrize(
    "response, expected_route, expected_hero",
    [
        ({"route": "dataset", "hero_name": None, "reasoning": "x"}, "dataset", None),
        ({"route": "superhero", "hero_name": "Wonder Woman", "reasoning": "x"}, "superhero", "Wonder Woman"),
        ({"route": "both", "hero_name": "Batman", "reasoning": "x"}, "both", "Batman"),
        ({"route": "neither", "hero_name": None, "reasoning": "x"}, "neither", None),
    ],
)
async def test_router_maps_response_to_decision(response, expected_route, expected_hero):
    decision = await _decide(response)
    assert decision.route == expected_route
    assert decision.hero_name == expected_hero
    assert decision.degraded is False


async def test_router_strips_hero_name_when_route_is_dataset():
    decision = await _decide({"route": "dataset", "hero_name": "Batman"})
    assert decision.hero_name is None


async def test_router_clamps_unknown_route_to_both():
    decision = await _decide({"route": "banana", "hero_name": None})
    assert decision.route == "both"


async def test_router_degrades_to_both_on_llm_failure(llm_error):
    decision = await _decide(raise_exc=llm_error)
    assert decision.route == "both"
    assert decision.hero_name is None
    assert decision.degraded is True


async def test_degraded_router_recovers_hero_name_from_question(llm_error):
    decision = await _decide(
        raise_exc=llm_error, question="What are Wonder Woman's power stats?"
    )
    assert decision.route == "both"
    assert decision.degraded is True
    assert decision.hero_name == "Wonder Woman"  # so "both" really queries both


async def test_router_recovers_hero_name_when_model_returns_null():
    decision = await _decide(
        {"route": "superhero", "hero_name": None, "reasoning": "x"},
        question="how tough is the Hulk really",
    )
    assert decision.route == "superhero"
    assert decision.hero_name == "Hulk"


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What are Wonder Woman's power stats?", "Wonder Woman"),
        ("Tell me about Iron Man", "Iron Man"),
        ("how strong is the Hulk?", "Hulk"),
        ("what is the capital of France", "France"),  # false positive: acceptable, best-effort
        ("who is stronger", None),
    ],
)
def test_guess_hero_name(question, expected):
    assert guess_hero_name(question) == expected
