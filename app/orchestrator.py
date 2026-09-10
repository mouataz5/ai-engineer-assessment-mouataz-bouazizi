import asyncio
import logging

from .dataset import DatasetSearch
from .models import AskResponse, Source
from .router import Router
from .superhero import SuperheroClient, format_hero
from .synthesis import NEITHER_MESSAGE, ContextBlock, synthesize
from .llm import LLMClient

logger = logging.getLogger(__name__)


def _dedupe(sources: list[Source]) -> list[Source]:
    seen: set[tuple] = set()
    out: list[Source] = []
    for s in sources:
        key = (s.type, s.ref, s.id)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


async def answer_question(
    question: str,
    *,
    router: Router,
    dataset: DatasetSearch,
    superhero: SuperheroClient,
    llm: LLMClient,
    top_k: int = 3,
) -> AskResponse:
    decision = await router.decide(question)
    logger.info("route=%s hero=%r degraded=%s", decision.route, decision.hero_name, decision.degraded)

    if decision.route == "neither":
        return AskResponse(
            answer=NEITHER_MESSAGE, route="neither", sources=[],
            router_degraded=decision.degraded,
        )

    want_dataset = decision.route in ("dataset", "both")
    want_hero = decision.route in ("superhero", "both")

    async def _run_dataset() -> list[ContextBlock]:
        if not want_dataset:
            return []
        try:
            docs = await asyncio.to_thread(dataset.search, question, top_k)
        except Exception as exc:  # noqa: BLE001 - one retriever failing must not kill the request
            logger.warning("dataset search failed: %s", exc)
            return []
        return [
            ContextBlock(
                source=Source(type="dataset", ref=d.doc_id, title=d.title),
                text=f"{d.title} ({d.year}). {d.plot}",
            )
            for d in docs
        ]

    async def _run_hero() -> list[ContextBlock]:
        if not want_hero or not decision.hero_name:
            return []
        try:
            heroes = await superhero.search(decision.hero_name)
        except Exception as exc:  # noqa: BLE001 - same: degrade, don't crash
            logger.warning("superhero search failed: %s", exc)
            return []
        return [
            ContextBlock(
                source=Source(type="superhero_api", ref=h.name, id=h.id or None),
                text=format_hero(h),
            )
            for h in heroes
        ]

    dataset_blocks, hero_blocks = await asyncio.gather(_run_dataset(), _run_hero())
    blocks = dataset_blocks + hero_blocks

    result = await synthesize(llm, question, blocks)
    sources = _dedupe([blocks[i].source for i in result.used_sources])

    return AskResponse(
        answer=result.answer,
        route=decision.route,
        sources=sources,
        router_degraded=decision.degraded,
    )
