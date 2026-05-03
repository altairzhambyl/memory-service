"""FastAPI app — open DB pool, init OpenAI client, serve /health /ingest /query."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import RequestContextMiddleware
from app.api.routes import router
from app.config import get_settings
from app.db.pool import close_pool, init_pool
from app.embeddings.openai import get_client
from app.obs.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("app.lifespan")

    log.info("startup.begin", embedding_model=settings.embedding_model)
    get_client()                                 # construct once; API is exercised on ingest/query
    await init_pool(settings)
    log.info("startup.ready")

    try:
        yield
    finally:
        log.info("shutdown.begin")
        await close_pool()


def create_app() -> FastAPI:
    app = FastAPI(title="Simple RAG", version="0.1.0", lifespan=lifespan)
    # Permissive CORS for local dev: lets a localhost client (e.g. the
    # `with-ui` branch's inspection panel served from a separate port,
    # or a curl/jq smoke harness) hit the API. Trim before any public
    # deployment — `allow_origin_regex` covers any localhost port only.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app


app = create_app()
