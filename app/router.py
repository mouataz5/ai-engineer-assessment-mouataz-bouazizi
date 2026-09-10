import logging
import re
from dataclasses import dataclass

from .errors import LLMError
from .llm import LLMClient
from .models import Route

logger = logging.getLogger(__name__)

_VALID_ROUTES: set[Route] = {"dataset", "superhero", "both", "neither"}

_SYSTEM = """You route a user's question to the data sources needed to answer it.

Sources:
- "dataset": a collection of MOVIE PLOT SUMMARIES. Use for questions about films,
  their plots, characters as portrayed on screen, directors, cast, or release year.
- "superhero": a STRUCTURED SUPERHERO DATABASE. Use for questions about a
  superhero's power stats, real/full name, publisher, alignment, race, height,
  weight, occupation, or first comic appearance.

Choose "both" when answering needs film context AND superhero-database facts
(e.g. "how does the movie portray Thor compared to his actual strength?").
Choose "neither" when the question is about something else entirely.

Respond with a JSON object:
{
  "route": "dataset" | "superhero" | "both" | "neither",
  "hero_name": "<name to look up in the superhero database, or null>",
  "reasoning": "<one short sentence>"
}
Set "hero_name" only when route is "superhero" or "both"; otherwise use null."""

_TITLECASE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:[ -][A-Z][A-Za-z0-9]+)*)\b")
_NAME_STOPWORDS = {
    "what", "who", "how", "why", "where", "when", "the", "a", "an", "is", "are",
    "was", "does", "do", "did", "tell", "give", "compare", "can", "could", "i",
}


def guess_hero_name(question: str) -> str | None:
    """Best-effort hero-name extraction, used only when the router LLM did not
    supply one (outage, or it returned null). Picks the longest Title-Case span,
    after stripping a leading possessive. Not a replacement for the LLM."""
    q = re.sub(r"['’]s\b", "", question)
    candidates = [c for c in _TITLECASE.findall(q) if c.lower() not in _NAME_STOPWORDS]
    if not candidates:
        return None
    return max(candidates, key=len)


@dataclass
class RouteDecision:
    route: Route
    hero_name: str | None
    reasoning: str
    degraded: bool = False


class Router:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def decide(self, question: str) -> RouteDecision:
        try:
            raw = await self._llm.complete_json(_SYSTEM, question)
        except LLMError as exc:
            # Fail open: query everything rather than 500 the endpoint. Still try
            # to recover a hero name so "both" actually means both.
            logger.warning("router LLM call failed, degrading to 'both': %s", exc)
            return RouteDecision(
                route="both",
                hero_name=guess_hero_name(question),
                reasoning="router unavailable; querying all sources",
                degraded=True,
            )
        return self._parse(raw, question)

    @staticmethod
    def _parse(raw: dict, question: str) -> RouteDecision:
        route = str(raw.get("route", "")).strip().lower()
        if route not in _VALID_ROUTES:
            logger.warning("router returned unknown route %r, clamping to 'both'", route)
            route = "both"

        hero_name = raw.get("hero_name")
        if isinstance(hero_name, str):
            hero_name = hero_name.strip() or None
        else:
            hero_name = None

        if route in ("dataset", "neither"):
            hero_name = None
        elif hero_name is None:
            # Model chose a superhero route but gave no name - recover one.
            hero_name = guess_hero_name(question)
            if hero_name:
                logger.info("router gave no hero_name; guessed %r from question", hero_name)

        reasoning = str(raw.get("reasoning", "")).strip()
        return RouteDecision(route=route, hero_name=hero_name, reasoning=reasoning)
