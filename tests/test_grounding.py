"""The core anti-hallucination guarantees: no context -> refuse; provenance
only lists blocks the model actually used."""
from app.orchestrator import answer_question
from app.router import Router
from app.superhero import Hero
from app.synthesis import NEITHER_MESSAGE, NO_CONTEXT_MESSAGE, ContextBlock, synthesize
from tests.conftest import FakeLLM, FakeSuperheroClient

BATMAN = Hero(id="70", name="Batman", full_name="Bruce Wayne", powerstats={"strength": "26"})


async def _answer(llm_responses, dataset_search, question, heroes=None):
    llm = FakeLLM(llm_responses)
    return (
        await answer_question(
            question,
            router=Router(llm),
            dataset=dataset_search,
            superhero=FakeSuperheroClient(heroes=heroes),
            llm=llm,
            top_k=3,
        ),
        llm,
    )


async def test_no_retrieval_hits_refuses_without_calling_llm_again(dataset_search):
    resp, llm = await _answer(
        [{"route": "superhero", "hero_name": "Nobody", "reasoning": "x"}],
        dataset_search,
        "What are Nobody's powers?",
        heroes=[],
    )
    assert resp.answer == NO_CONTEXT_MESSAGE
    assert resp.sources == []
    assert len(llm.calls) == 1  # router only; synthesis short-circuited


async def test_neither_route_returns_canned_message(dataset_search):
    resp, llm = await _answer(
        [{"route": "neither", "hero_name": None, "reasoning": "off topic"}],
        dataset_search,
        "What's the weather tomorrow?",
    )
    assert resp.answer == NEITHER_MESSAGE
    assert resp.route == "neither"
    assert resp.sources == []
    assert len(llm.calls) == 1


async def test_ungrounded_answer_carries_no_sources(dataset_search):
    resp, _ = await _answer(
        [
            {"route": "dataset", "hero_name": None, "reasoning": "x"},
            {"answer": "The context doesn't say.", "used_sources": [], "grounded": False},
        ],
        dataset_search,
        "Who directed The Dark Knight?",
    )
    assert resp.answer == "The context doesn't say."
    assert resp.sources == []


async def test_sources_reflect_only_blocks_the_model_used(dataset_search):
    resp, _ = await _answer(
        [
            {"route": "both", "hero_name": "Batman", "reasoning": "x"},
            {"answer": "Batman fights the Joker.", "used_sources": [0], "grounded": True},
        ],
        dataset_search,
        "How does the movie portray Batman versus his real strength?",
        heroes=[BATMAN],
    )
    assert len(resp.sources) == 1
    assert resp.sources[0].type == "dataset"  # block 0 is a dataset hit


async def test_synthesize_empty_blocks_makes_no_llm_call():
    llm = FakeLLM([])
    result = await synthesize(llm, "anything", [])
    assert result.grounded is False
    assert result.used_sources == []
    assert llm.calls == []


async def test_synthesize_backfills_sources_when_model_claims_grounding():
    llm = FakeLLM([{"answer": "Yes.", "used_sources": [], "grounded": True}])
    from app.models import Source

    blocks = [
        ContextBlock(source=Source(type="dataset", ref="doc_001", title="T"), text="a"),
        ContextBlock(source=Source(type="dataset", ref="doc_002", title="U"), text="b"),
    ]
    result = await synthesize(llm, "q", blocks)
    assert result.used_sources == [0, 1]
