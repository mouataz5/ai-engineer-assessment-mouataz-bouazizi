from typing import Literal

from pydantic import BaseModel, Field, field_validator

Route = Literal["dataset", "superhero", "both", "neither"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return v


class Source(BaseModel):
    """One piece of provenance. `type` says which retriever produced it."""

    type: Literal["dataset", "superhero_api"]
    ref: str
    id: str | None = None
    title: str | None = None


class AskResponse(BaseModel):
    answer: str
    route: Route
    sources: list[Source]
    # True when the router LLM call failed and we fell back to querying everything.
    router_degraded: bool = False
