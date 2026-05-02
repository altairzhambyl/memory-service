# syntax=docker/dockerfile:1.6
# Single-stage image. No model pre-download — embeddings run in OpenAI's cloud.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

RUN pip install --no-cache-dir \
        "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" \
        "asyncpg>=0.29.0" "pgvector>=0.3.6" \
        "pydantic>=2.9.0" "pydantic-settings>=2.5.0" \
        "openai>=1.54.0" "structlog>=24.4.0"

COPY --chown=app:app app ./app
COPY --chown=app:app migrations ./migrations

USER app

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
