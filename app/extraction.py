"""
LLM-based memory extraction.

Pulls typed structured memories from a /turns body. Uses OpenAI Structured
Outputs (`client.chat.completions.parse` + Pydantic schema) so the model is
constrained server-side to emit a valid `ExtractionResponse` — the
`json_object`-mode failure modes (truncated JSON with `finish_reason=stop`,
schema drift between identical calls) are eliminated by construction.

`extract_memories_from_turn` never raises on transient API failure: it
returns `(memories, error_string_or_none)` so the caller can persist the
error in `turns.metadata.extraction_error` and still commit the turn.
`AuthenticationError` IS re-raised — that is a config bug and silently
degrading would hide it from the operator.
"""

from __future__ import annotations

from typing import Literal

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from app.api.schemas import TurnMessage
from app.config import get_settings
from app.embeddings.openai import get_client
from app.obs.logging import get_logger

log = get_logger(__name__)


MemoryType = Literal["fact", "preference", "opinion", "event"]


class ExtractedMemory(BaseModel):
    """One typed memory atom. Strict JSON schema is built from this."""

    type: MemoryType
    key: str
    value: str
    confidence: float


class ExtractionResponse(BaseModel):
    """Top-level response — list of zero or more atoms."""

    memories: list[ExtractedMemory]


# Prompt is written from the reference shape only; not lifted from
# mem0 / honcho / hindsight (originality). Examples come from the v1
# fixture content so the model sees the kind of data we actually have.
_SYSTEM_PROMPT = """You extract typed structured memories from a single conversation turn between a user and an assistant. Emit ONLY memories about the user that they explicitly stated or clearly implied — never about the assistant, never speculation, never general knowledge.

CATEGORIES (cover these when present):
- employment: current employer, role, previous companies → keys like `employment.company`, `employment.role`, `employment.previous_company`.
- location: current and previous cities → `location.city`, `location.previous_city`.
- family / pets: names, species, ages → `pet.name`, `pet.species`, `pet.age_years`, `family.spouse_name`.
- preferences: dietary style, beverages, communication style → `diet.style`, `beverage.coffee`, `communication.tone`.
- opinions: stated views → `opinion.<topic>`.
- implicit facts: "walking my dog Biscuit" implies the user has a pet named Biscuit.
- corrections: "actually, I meant X" — emit the corrected fact only, NOT both.

TYPES:
- `fact` — objective state (employment, location, ownership, allergies).
- `preference` — choices the user makes (vegetarian, no coffee).
- `opinion` — stated views on a topic ("I love TypeScript").
- `event` — a specific time-bounded occurrence ("got married last year").

KEY CONVENTIONS:
- Lowercase dotted-path: `pet.name`, NOT `Pet Name` or `pet_name`.
- One memory per atomic fact. "Vegetarian and allergic to shellfish" is TWO memories: `diet.style=vegetarian` (preference) AND `allergy=shellfish` (fact).
- Confidence: 0.95 explicit ("I work at Notion"), 0.85 clear implication ("walking Biscuit"), 0.65 hedged ("I think I might be lactose intolerant").
- If the turn says nothing memory-worthy about the user, return `{"memories": []}`.

EXAMPLES:

Input:
user: I just moved to Berlin from NYC last month. Loving it so far.
assistant: That sounds exciting! Berlin is a great city.

Output:
{"memories":[
  {"type":"fact","key":"location.city","value":"Berlin","confidence":0.95},
  {"type":"fact","key":"location.previous_city","value":"NYC","confidence":0.9}
]}

Input:
user: I'm vegetarian and allergic to shellfish. Don't drink coffee — caffeine wrecks my sleep.

Output:
{"memories":[
  {"type":"preference","key":"diet.style","value":"vegetarian","confidence":0.95},
  {"type":"fact","key":"allergy","value":"shellfish","confidence":0.95},
  {"type":"preference","key":"beverage.coffee","value":"no","confidence":0.85}
]}

Input:
user: Just got back from walking Biscuit in the park. Beautiful morning.

Output:
{"memories":[
  {"type":"fact","key":"pet.name","value":"Biscuit","confidence":0.85}
]}
"""


def _format_turn(messages: list[TurnMessage]) -> str:
    """Inline the turn for the user message slot — one role:content per line."""
    lines: list[str] = []
    for m in messages:
        prefix = m.role
        if m.role == "tool" and m.name:
            prefix = f"tool:{m.name}"
        lines.append(f"{prefix}: {m.content}")
    return "\n".join(lines)


async def extract_memories_from_turn(
    messages: list[TurnMessage],
) -> tuple[list[dict], str | None]:
    """
    Extract typed memories from a /turns body. Returns
    `(memories, error_or_none)`. Never raises on transient API failure.

    `AuthenticationError` IS re-raised so a missing/invalid API key fails
    loudly at request time rather than silently producing zero memories
    forever.
    """
    settings = get_settings()
    client = get_client()
    try:
        # max_retries=0: same hot-path reasoning as embed in /turns —
        # the SDK default of 2 retries × 30s would blow the 60s
        # /turns budget on a 429 burst. Single-attempt-or-fail is the
        # right primitive here; the caller persists the error and
        # commits the turn anyway.
        completion = await client.with_options(max_retries=0).chat.completions.parse(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _format_turn(messages)},
            ],
            response_format=ExtractionResponse,
            timeout=30.0,
        )
    except AuthenticationError:
        # Config bug, not a runtime data issue — surface it.
        raise
    except RateLimitError:
        log.warning("extraction.rate_limit")
        return [], "rate_limit"
    except APITimeoutError:
        log.warning("extraction.timeout")
        return [], "timeout"
    except APIConnectionError as exc:
        log.warning("extraction.connection_error", error=str(exc)[:200])
        return [], "connection"
    except BadRequestError as exc:
        log.warning("extraction.bad_request", status=exc.status_code, error=str(exc)[:200])
        return [], f"bad_request:{exc.status_code}"
    except LengthFinishReasonError:
        # Model output got cut off by max_tokens before the JSON closed.
        log.warning("extraction.length_finish")
        return [], "length_finish"
    except ContentFilterFinishReasonError:
        # OpenAI content filter killed the response mid-stream.
        log.warning("extraction.content_filter")
        return [], "content_filter"
    except ValidationError as exc:
        # Strict structured outputs should make this unreachable; defence in depth.
        log.warning("extraction.validation_error", error=str(exc)[:200])
        return [], "validation"
    except Exception as exc:  # noqa: BLE001 — last-resort net for unmapped transient SDK errors
        log.warning(
            "extraction.unknown_error",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return [], f"unknown:{type(exc).__name__}"

    msg = completion.choices[0].message
    if msg.parsed is None:
        # Refusal (model declined) — treat as soft failure with the refusal text.
        log.warning("extraction.refusal", refusal=msg.refusal)
        return [], "refusal"

    out: list[dict] = []
    for m in msg.parsed.memories:
        # Defensive post-processing: clamp confidence, drop empty key/value.
        # Strict schema accepts any string but we don't want blanks polluting
        # the memories table.
        key = m.key.strip()
        value = m.value.strip()
        if not key or not value:
            continue
        confidence = max(0.0, min(1.0, float(m.confidence)))
        out.append({
            "type": m.type,
            "key": key,
            "value": value,
            "confidence": confidence,
        })
    return out, None
