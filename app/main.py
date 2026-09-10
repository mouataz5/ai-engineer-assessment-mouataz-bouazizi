import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse

from .config import get_settings
from .dataset import DatasetSearch, ensure_index
from .deps import get_dataset, get_llm, get_router, get_superhero
from .errors import LLMError
from .llm import GroqLLMClient, LLMClient
from .models import AskRequest, AskResponse
from .orchestrator import answer_question
from .router import Router
from .superhero import SuperheroClient

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ensure_index(settings.dataset_csv, settings.db_path)

    app.state.settings = settings
    app.state.llm = GroqLLMClient(
        settings.groq_api_key, settings.groq_model, settings.llm_timeout
    )
    app.state.router = Router(app.state.llm)
    app.state.dataset = DatasetSearch(settings.db_path)
    app.state.superhero = SuperheroClient(
        settings.superhero_api_token, settings.http_timeout, settings.superhero_base_url
    )
    try:
        yield
    finally:
        await app.state.llm.aclose()
        await app.state.superhero.aclose()


app = FastAPI(title="AI Engineer Assessment - Ask", lifespan=lifespan)


@app.exception_handler(LLMError)
async def _llm_error_handler(_request, exc: LLMError):
    logging.getLogger(__name__).error("LLM unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "LLM service unavailable"})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    router: Router = Depends(get_router),
    dataset: DatasetSearch = Depends(get_dataset),
    superhero: SuperheroClient = Depends(get_superhero),
    llm: LLMClient = Depends(get_llm),
) -> AskResponse:
    return await answer_question(
        req.question,
        router=router,
        dataset=dataset,
        superhero=superhero,
        llm=llm,
        top_k=get_settings().top_k,
    )
