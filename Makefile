.PHONY: help up down logs build smoke test test-e2e fmt lint clean

help:
	@echo "Targets:"
	@echo "  up        - docker compose up -d"
	@echo "  down      - docker compose down"
	@echo "  logs      - tail app logs"
	@echo "  build     - docker compose build"
	@echo "  smoke     - manual smoke against http://localhost:8080"
	@echo "  test      - pytest (skips e2e by default)"
	@echo "  test-e2e  - pytest including live-stack tests (requires up)"
	@echo "  fmt       - ruff format"
	@echo "  lint      - ruff check"
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
	pytest tests

test-e2e:
	RAG_E2E=1 pytest tests

fmt:
	ruff format app tests

lint:
	ruff check app tests

clean:
	docker compose down -v
