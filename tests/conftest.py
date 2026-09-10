import pytest

from app.dataset import DatasetSearch, build_index
from app.errors import LLMError


class FakeLLM:
    """Substitute for GroqLLMClient.

    `json_responses` is a list consumed one call at a time; or set `raise_exc`
    to simulate provider failure. `calls` records every (system, user) pair.
    """

    def __init__(self, json_responses=None, raise_exc: Exception | None = None):
        self._responses = list(json_responses or [])
        self._raise = raise_exc
        self.calls: list[tuple[str, str]] = []

    async def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        if self._raise is not None:
            raise self._raise
        if not self._responses:
            raise AssertionError("FakeLLM ran out of queued responses")
        return self._responses.pop(0)

    async def aclose(self) -> None:  # pragma: no cover
        pass


class FakeSuperheroClient:
    def __init__(self, heroes=None, raise_exc: Exception | None = None):
        self._heroes = heroes or []
        self._raise = raise_exc
        self.calls: list[str] = []

    async def search(self, name: str):
        self.calls.append(name)
        if self._raise is not None:
            raise self._raise
        return list(self._heroes)

    async def aclose(self) -> None:  # pragma: no cover
        pass


@pytest.fixture(scope="session")
def dataset_db(tmp_path_factory) -> str:
    db = tmp_path_factory.mktemp("data") / "movies.db"
    build_index("data/movies.csv", db)
    return str(db)


@pytest.fixture
def dataset_search(dataset_db) -> DatasetSearch:
    return DatasetSearch(dataset_db)


@pytest.fixture
def llm_error() -> LLMError:
    return LLMError("boom")
