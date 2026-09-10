import httpx
import pytest
import respx

from app.errors import SuperheroError, SuperheroTimeout
from app.superhero import SuperheroClient, trim

BASE = "https://superheroapi.com/api"
TOKEN = "testtoken"

SUCCESS = {
    "response": "success",
    "results-for": "batman",
    "results": [
        {
            "id": "70",
            "name": "Batman",
            "powerstats": {"intelligence": "100", "strength": "26", "speed": "27"},
            "biography": {
                "full-name": "Bruce Wayne",
                "publisher": "DC Comics",
                "alignment": "good",
                "first-appearance": "Detective Comics #27",
            },
            "appearance": {"race": "Human", "height": ["6'2", "188 cm"], "weight": ["210 lb", "95 kg"]},
            "work": {"occupation": "Businessman"},
        }
    ],
}


def _client() -> SuperheroClient:
    return SuperheroClient(TOKEN, timeout=1.0, base_url=BASE)


@respx.mock
async def test_search_hit_returns_trimmed_hero():
    respx.get(f"{BASE}/{TOKEN}/search/batman").mock(
        return_value=httpx.Response(200, json=SUCCESS)
    )
    heroes = await _client().search("batman")
    assert len(heroes) == 1
    h = heroes[0]
    assert h.id == "70"
    assert h.full_name == "Bruce Wayne"
    assert h.height == "188 cm"  # metric side of the pair
    assert h.powerstats["intelligence"] == "100"


@respx.mock
async def test_search_miss_returns_empty_list():
    respx.get(url__regex=rf"{BASE}/.*").mock(
        return_value=httpx.Response(200, json={"response": "error", "error": "not found"})
    )
    assert await _client().search("nobody") == []


@respx.mock
async def test_search_timeout_raises():
    respx.get(url__regex=rf"{BASE}/.*").mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(SuperheroTimeout):
        await _client().search("batman")


@respx.mock
async def test_search_malformed_payload_raises():
    respx.get(url__regex=rf"{BASE}/.*").mock(
        return_value=httpx.Response(200, content=b"<html>not json</html>")
    )
    with pytest.raises(SuperheroError):
        await _client().search("batman")


@respx.mock
async def test_search_http_500_raises():
    respx.get(url__regex=rf"{BASE}/.*").mock(return_value=httpx.Response(500))
    with pytest.raises(SuperheroError):
        await _client().search("batman")


def test_trim_tolerates_missing_sections():
    h = trim({"id": 1, "name": "Ghost"})
    assert h.name == "Ghost"
    assert h.full_name is None
    assert h.powerstats == {}
