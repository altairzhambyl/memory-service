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

Both functions follow the (vectors, error_or_none) tuple convention
established by `app/extraction.py` so the §3 callers can branch on
upstream failures without try/except wrappers everywhere. §5 mandates
the service degrades gracefully on transient OpenAI blips
(`APIConnectionError`, `APITimeoutError`, `RateLimitError`,
`BadRequestError`); `AuthenticationError` is re-raised because that
is a config bug and silently producing zero vectors would hide it.

Note: text-embedding-3 doesn't distinguish query vs document inputs the
way Qwen3 does. Same `embed()` call for both — much simpler.
"""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from app.config import get_settings
from app.obs.logging import get_logger

log = get_logger(__name__)

_client: AsyncOpenAI | None = None

# Test-only failure injection. The DB is not exposed to the host so HGT-24
# tests can't run in-process; instead they POST /admin/inject_embed_failure
# to load up a counter, then exercise /turns or /recall via the live HTTP
# stack. The counter is decremented on each embed call until exhausted, at
# which point normal behaviour resumes. NOT part of the §3 contract.
_force_fail_remaining: int = 0
_force_fail_error: str = "connection"


def inject_embed_failure(count: int, error: str = "connection") -> None:
    global _force_fail_remaining, _force_fail_error
    _force_fail_remaining = max(0, int(count))
    _force_fail_error = error or "connection"


def _consume_forced_failure() -> str | None:
    global _force_fail_remaining
    if _force_fail_remaining > 0:
        _force_fail_remaining -= 1
        return _force_fail_error
    return None


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
) -> tuple[list[list[float]], str | None]:
    """
    Embed N texts in one API call. Returns `(vectors, error_or_none)`.
    On success: vectors is a list of N lists of 1536 floats, error is None.
    On transient API failure: vectors is `[]`, error is a short string
    tag (`rate_limit`, `timeout`, `connection`, `bad_request:{status}`,
    `unknown:{ExcType}`). Caller decides whether to persist partial
    state, return 503, or fall back to cold output.

    `max_retries` overrides the client default (2). Hot paths inside
    POST /turns must pass `max_retries=0` because the §3 60s budget
    can't accommodate three sequential 30s retry windows on a 429
    or transient 5xx — see openai-python issue #762 for the default
    timeout reasoning. Best-effort callers (/ingest, /recall) keep
    the default so transient blips don't bubble up.

    `AuthenticationError` is deliberately re-raised — that is a config
    bug, silently degrading would hide it from the operator.
    """
    forced = _consume_forced_failure()
    if forced is not None:
        log.warning("embed.test_injected_failure", n=len(texts), reason=forced)
        return [], forced
    settings = get_settings()
    client = get_client()
    if max_retries is not None:
        client = client.with_options(max_retries=max_retries)
    try:
        resp = await client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
        )
    except AuthenticationError:
        raise
    except RateLimitError:
        log.warning("embed.rate_limit", n=len(texts))
        return [], "rate_limit"
    except APITimeoutError:
        log.warning("embed.timeout", n=len(texts))
        return [], "timeout"
    except APIConnectionError as exc:
        log.warning("embed.connection_error", n=len(texts), error=str(exc)[:200])
        return [], "connection"
    except BadRequestError as exc:
        log.warning(
            "embed.bad_request",
            n=len(texts), status=exc.status_code, error=str(exc)[:200],
        )
        return [], f"bad_request:{exc.status_code}"
    except Exception as exc:  # noqa: BLE001 — last-resort net for unmapped SDK errors
        log.warning(
            "embed.unknown_error",
            n=len(texts),
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return [], f"unknown:{type(exc).__name__}"
    return [item.embedding for item in resp.data], None


async def embed_query(
    query: str, *, max_retries: int | None = None,
) -> tuple[list[float], str | None]:
    """Embed a single query string. Returns `(vector, error_or_none)`."""
    vecs, err = await embed_documents([query], max_retries=max_retries)
    if err is not None:
        return [], err
    return vecs[0], None
