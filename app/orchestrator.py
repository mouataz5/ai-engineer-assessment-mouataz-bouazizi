import asyncio
import logging
from dataclasses import dataclass, field

from .dataset import DatasetSearch
from .llm import LLMClient
from .models import AskResponse, Source
from .router import Router
from .superhero import SuperheroClient, format_hero
from .synthesis import (
    NEITHER_MESSAGE,
    NO_CONTEXT_MESSAGE,
    NO_HERO_MESSAGE,
    PARTIAL_FAILURE_MESSAGE,
    ContextBlock,
    synthesize,
)

logger = logging.getLogger(__name__)


@dataclass
class _Retrieval:
    """Outcome of one retriever. `failed` is True only on an error, never on an
    honest empty result - that distinction is the whole point."""

    blocks: list[ContextBlock] = field(default_factory=list)
    failed: bool = False


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
    logger.info(
        "route=%s hero=%r degraded=%s", decision.route, decision.hero_name, decision.degraded
    )

    if decision.route == "neither":
        return AskResponse(
            answer=NEITHER_MESSAGE, route="neither", sources=[],
            router_degraded=decision.degraded,
        )

    want_dataset = decision.route in ("dataset", "both")
    want_hero = decision.route in ("superhero", "both")
    # Wanted a superhero lookup but neither the router nor the fallback found a name.
    hero_unidentified = want_hero and not decision.hero_name

    async def _run_dataset() -> _Retrieval:
        if not want_dataset:
            return _Retrieval()
        try:
            docs = await asyncio.to_thread(dataset.search, question, top_k)
        except Exception as exc:  # noqa: BLE001 - one retriever failing must not kill the request
            logger.warning("dataset search failed: %s", exc)
            return _Retrieval(failed=True)
        return _Retrieval([
            ContextBlock(
                source=Source(type="dataset", ref=d.doc_id, title=d.title),
                text=f"{d.title} ({d.year}). {d.plot}",
            )
            for d in docs
        ])

    async def _run_hero() -> _Retrieval:
        if not want_hero or not decision.hero_name:
            return _Retrieval()
        try:
            heroes = await superhero.search(decision.hero_name)
        except Exception as exc:  # noqa: BLE001 - same: degrade, don't crash
            logger.warning("superhero search failed: %s", exc)
            return _Retrieval(failed=True)
        return _Retrieval([
            ContextBlock(
                source=Source(type="superhero_api", ref=h.name, id=h.id or None),
                text=format_hero(h),
            )
            for h in heroes
        ])

    ds, hs = await asyncio.gather(_run_dataset(), _run_hero())
    blocks = ds.blocks + hs.blocks

    failed_sources = [
        name for name, r in (("the movie dataset", ds), ("the superhero API", hs)) if r.failed
    ]
    warnings: list[str] = []
    notes: list[str] = []
    if failed_sources:
        joined = " and ".join(failed_sources)
        warnings.append(f"{joined} could not be reached; the answer may be incomplete")
        notes.append(f"{joined.capitalize()} could not be reached for this question.")
    if hero_unidentified:
        warnings.append("could not identify which superhero the question refers to")

    if not blocks:
        if failed_sources:
            answer = PARTIAL_FAILURE_MESSAGE
        elif hero_unidentified and decision.route == "superhero":
            answer = NO_HERO_MESSAGE
        else:
            answer = NO_CONTEXT_MESSAGE
        return AskResponse(
            answer=answer, route=decision.route, sources=[],
            router_degraded=decision.degraded, warnings=warnings,
        )

    result = await synthesize(llm, question, blocks, notes=notes)
    sources = _dedupe([blocks[i].source for i in result.used_sources])

    return AskResponse(
        answer=result.answer,
        route=decision.route,
        sources=sources,
        router_degraded=decision.degraded,
        warnings=warnings,
    )
