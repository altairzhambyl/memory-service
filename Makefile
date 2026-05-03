.PHONY: help up down logs build smoke test test-e2e fmt lint clean

help:
	@echo "Targets:"
	@echo "  up        - docker compose up -d"
	@echo "  down      - docker compose down"
	@echo "  logs      - tail app logs"
	@echo "  build     - docker compose build"
	@echo "  smoke     - manual smoke against http://localhost:8080"
	@echo "  test      - uv-run pytest (skips e2e by default)"
	@echo "  test-e2e  - uv-run pytest including live-stack tests (requires up)"
	@echo "  fmt       - ruff format src tests"
	@echo "  lint      - ruff check src tests"
	@echo "  clean     - remove postgres volume (DESTROYS DATA)"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f app

build:
	docker compose build

smoke:
	@bash scripts/smoke.sh

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev pytest -p pytest_asyncio.plugin tests

test-e2e:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RAG_E2E=1 uv run --extra dev pytest -p pytest_asyncio.plugin tests

fmt:
	uv run --extra dev ruff format src tests

lint:
	uv run --extra dev ruff check src tests

clean:
	docker compose down -v
