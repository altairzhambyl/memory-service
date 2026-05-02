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
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
        log.info("openai.client.ready", model=settings.embedding_model)
    return _client


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed N texts in one API call. Returns N lists of 1536 floats."""
    settings = get_settings()
    client = get_client()
    resp = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in resp.data]


async def embed_query(query: str) -> list[float]:
    """Embed a single query string. Returns 1536 floats."""
    vecs = await embed_documents([query])
    return vecs[0]
