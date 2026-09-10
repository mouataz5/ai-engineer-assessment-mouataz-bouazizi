import logging
from dataclasses import dataclass

from .errors import LLMError
from .llm import LLMClient
from .models import Source

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in the movie dataset or the superhero database "
    "to answer that."
)
NEITHER_MESSAGE = (
    "I can only answer questions about movies (from a plot-summary dataset) or "
    "about superheroes (from a superhero database). That question falls outside both."
)

_SYSTEM = """You answer the user's question using ONLY the numbered context blocks provided.

Rules:
- Use only facts stated in the context. Do not add outside knowledge or guesses.
- If the context does not contain enough information, say so plainly and set
  "grounded" to false.
- Be concise.

Respond with a JSON object:
{
  "answer": "<your answer>",
  "used_sources": [<indices of the blocks you actually used, e.g. 0, 2>],
  "grounded": true | false
}"""


@dataclass
class ContextBlock:
    source: Source
    text: str


@dataclass
class Synthesis:
    answer: str
    used_sources: list[int]
    grounded: bool


def _render(blocks: list[ContextBlock]) -> str:
    parts = []
    for i, b in enumerate(blocks):
        label = f"[{i}] source_type={b.source.type} ref={b.source.ref!r}"
        parts.append(f"{label}\n{b.text}")
    return "\n\n".join(parts)


async def synthesize(
    llm: LLMClient, question: str, blocks: list[ContextBlock]
) -> Synthesis:
    if not blocks:
        # No retrieval hits -> deterministic refusal, no LLM call.
        return Synthesis(answer=NO_CONTEXT_MESSAGE, used_sources=[], grounded=False)

    user = f"Question: {question}\n\nContext blocks:\n{_render(blocks)}"
    raw = await llm.complete_json(_SYSTEM, user)

    answer = str(raw.get("answer", "")).strip()
    if not answer:
        raise LLMError(f"synthesis returned no answer: {raw!r}")

    grounded = bool(raw.get("grounded", True))

    used = raw.get("used_sources")
    if isinstance(used, list):
        used_idx = sorted({i for i in used if isinstance(i, int) and 0 <= i < len(blocks)})
    else:
        used_idx = []
    # If the model claims grounding but names no sources, fall back to all blocks
    # we passed in rather than returning an answer with empty provenance.
    if grounded and not used_idx:
        used_idx = list(range(len(blocks)))
    if not grounded:
        used_idx = []

    return Synthesis(answer=answer, used_sources=used_idx, grounded=grounded)
