import logging
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
            # Fail open: query everything rather than 500 the endpoint.
            logger.warning("router LLM call failed, degrading to 'both': %s", exc)
            return RouteDecision(
                route="both",
                hero_name=None,
                reasoning="router unavailable; querying all sources",
                degraded=True,
            )
        return self._parse(raw)

    @staticmethod
    def _parse(raw: dict) -> RouteDecision:
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

        reasoning = str(raw.get("reasoning", "")).strip()
        return RouteDecision(route=route, hero_name=hero_name, reasoning=reasoning)
