import logging
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

from .errors import SuperheroError, SuperheroTimeout

logger = logging.getLogger(__name__)

# The Superhero API returns a large nested blob per character. We forward only
# this subset to the LLM: cheaper tokens, less noise, deliberate context.
MAX_RESULTS = 3


@dataclass
class Hero:
    id: str
    name: str
    full_name: str | None = None
    publisher: str | None = None
    alignment: str | None = None
    race: str | None = None
    height: str | None = None
    weight: str | None = None
    occupation: str | None = None
    first_appearance: str | None = None
    powerstats: dict[str, str] = field(default_factory=dict)


def _last(value) -> str | None:
    """Appearance fields come back as ["imperial", "metric"]; keep the metric one."""
    if isinstance(value, list) and value:
        v = str(value[-1]).strip()
        return v or None
    if isinstance(value, str):
        return value.strip() or None
    return None


def _clean(value) -> str | None:
    if isinstance(value, str):
        v = value.strip()
        return v if v and v.lower() != "null" else None
    return None


def trim(result: dict) -> Hero:
    bio = result.get("biography") or {}
    app = result.get("appearance") or {}
    work = result.get("work") or {}
    stats = result.get("powerstats") or {}
    return Hero(
        id=str(result.get("id", "")),
        name=str(result.get("name", "")).strip(),
        full_name=_clean(bio.get("full-name")),
        publisher=_clean(bio.get("publisher")),
        alignment=_clean(bio.get("alignment")),
        race=_clean(app.get("race")),
        height=_last(app.get("height")),
        weight=_last(app.get("weight")),
        occupation=_clean(work.get("occupation")),
        first_appearance=_clean(bio.get("first-appearance")),
        powerstats={k: str(v) for k, v in stats.items() if _clean(str(v))},
    )


def format_hero(h: Hero) -> str:
    lines = [f"Name: {h.name}"]
    if h.full_name:
        lines.append(f"Full name: {h.full_name}")
    if h.publisher:
        lines.append(f"Publisher: {h.publisher}")
    if h.alignment:
        lines.append(f"Alignment: {h.alignment}")
    if h.race:
        lines.append(f"Race: {h.race}")
    if h.height or h.weight:
        lines.append(f"Height/Weight: {h.height or '?'} / {h.weight or '?'}")
    if h.occupation:
        lines.append(f"Occupation: {h.occupation}")
    if h.first_appearance:
        lines.append(f"First appearance: {h.first_appearance}")
    if h.powerstats:
        stats = ", ".join(f"{k}={v}" for k, v in h.powerstats.items())
        lines.append(f"Power stats: {stats}")
    return "\n".join(lines)


class SuperheroClient:
    def __init__(self, token: str | None, timeout: float, base_url: str) -> None:
        self._token = token or "missing-token"
        self._base = base_url.rstrip("/")
        # superheroapi.com 302-redirects to its www.../api.php host.
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def search(self, name: str) -> list[Hero]:
        url = f"{self._base}/{self._token}/search/{quote(name.strip())}"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise SuperheroTimeout(f"Superhero API timed out for {name!r}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise SuperheroError(f"Superhero API request failed for {name!r}: {exc}") from exc

        if not isinstance(data, dict) or data.get("response") != "success":
            # "error" response = character not found; treat as an empty result.
            logger.info("superhero search for %r returned no results", name)
            return []

        results = data.get("results") or []
        return [trim(r) for r in results[:MAX_RESULTS] if isinstance(r, dict)]

    async def aclose(self) -> None:
        await self._client.aclose()
