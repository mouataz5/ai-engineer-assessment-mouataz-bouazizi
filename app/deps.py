from fastapi import Request

from .dataset import DatasetSearch
from .llm import LLMClient
from .router import Router
from .superhero import SuperheroClient

# These read singletons created in the lifespan handler (app.main). Keeping them
# as tiny functions is what lets the tests override each dependency in isolation.


def get_router(request: Request) -> Router:
    return request.app.state.router


def get_dataset(request: Request) -> DatasetSearch:
    return request.app.state.dataset


def get_superhero(request: Request) -> SuperheroClient:
    return request.app.state.superhero


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm
