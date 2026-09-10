import json
from typing import Protocol

from openai import AsyncOpenAI

from .errors import LLMError

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMClient(Protocol):
    """The seam every other module depends on. Tests substitute a fake."""

    async def complete_json(self, system: str, user: str) -> dict: ...

    async def aclose(self) -> None: ...


class GroqLLMClient:
    """Groq via its OpenAI-compatible endpoint.

    Swapping providers (Gemini, Cerebras, NIM) is a base_url + model change,
    because everything downstream only knows about `complete_json`.
    """

    def __init__(self, api_key: str | None, model: str, timeout: float) -> None:
        self._model = model
        # A placeholder key lets the client construct; a real call will 401.
        self._client = AsyncOpenAI(
            api_key=api_key or "missing-key",
            base_url=GROQ_BASE_URL,
            timeout=timeout,
            max_retries=1,
        )

    async def complete_json(self, system: str, user: str) -> dict:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001 - normalise every SDK failure
            raise LLMError(f"LLM request failed: {exc}") from exc

        content = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned non-JSON content: {content!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"LLM returned JSON that is not an object: {content!r}")
        return parsed

    async def aclose(self) -> None:
        await self._client.close()
