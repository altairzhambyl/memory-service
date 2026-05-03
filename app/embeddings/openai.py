"""
OpenAI embeddings — thin async wrapper around the official SDK.

Why this file is its own thing: keeping the embedding call in one named
module means swapping providers later (Voyage, Cohere, local fastembed)
is a one-file change. The HTTP routes import `embed_documents` and
`embed_query` and don't care where the vectors came from.

text-embedding-3-small returns 1536-dim float vectors. We do NOT pass
`dimensions=` — the schema is `vector(1536)`, so we let the model use
its native size. Cosine similarity on these vectors lives roughly in
[0, 1] for related text, near 0 for unrelated.

Note: text-embedding-3 doesn't distinguish query vs document inputs the
way Qwen3 does. Same `embed()` call for both — much simpler.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config import get_settings
from app.obs.logging import get_logger

log = get_logger(__name__)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """
    Process-singleton AsyncOpenAI. Explicit `timeout=30` because the SDK
    default is 600s (per https://github.com/openai/openai-python/issues/762),
    which is far above our §3 60s /turns budget. `max_retries=2` lets the
    SDK transparently retry transient 429/503 with exponential backoff so
    the application code only sees terminal failures.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=30.0,
            max_retries=2,
        )
        log.info("openai.client.ready", model=settings.embedding_model)
    return _client


async def embed_documents(
    texts: list[str],
    *,
    max_retries: int | None = None,
) -> list[list[float]]:
    """
    Embed N texts in one API call. Returns N lists of 1536 floats.

    `max_retries` overrides the client default (2). Hot paths inside
    POST /turns must pass `max_retries=0` because the §3 60s budget
    can't accommodate three sequential 30s retry windows on a 429
    or transient 5xx — see openai-python issue #762 for the default
    timeout reasoning. Best-effort callers (/ingest, /recall) keep
    the default so transient blips don't bubble up.
    """
    settings = get_settings()
    client = get_client()
    if max_retries is not None:
        client = client.with_options(max_retries=max_retries)
    resp = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in resp.data]


async def embed_query(query: str, *, max_retries: int | None = None) -> list[float]:
    """Embed a single query string. Returns 1536 floats."""
    vecs = await embed_documents([query], max_retries=max_retries)
    return vecs[0]
