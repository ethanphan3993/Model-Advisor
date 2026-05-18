.PHONY: dev dev-fresh dev-backend dev-frontend stop build test clean refresh fresh package run

BPORT := 8000
FPORT := 5173
PY := PYTHONPATH=. .venv/bin/python

# Interactive: prompts whether to wipe cache + refresh data before starting.
# Press Enter (default = no) to start fast with existing cache.
# Type `y` to do a clean refresh first (~25s for all sources).
dev:
	@printf "\033[33m? \033[0mReset cache and refresh data sources first? [y/N] "; \
	read yn; \
	if [ "$$yn" = "y" ] || [ "$$yn" = "Y" ]; then \
		echo "→ Wiping .cache/ and refreshing all sources..."; \
		rm -rf .cache; \
		$(MAKE) --no-print-directory refresh; \
		echo ""; \
	fi
	@echo "Starting both servers (Ctrl-C to stop both)..."
	@(make dev-backend &); make dev-frontend

# Non-interactive: always wipes + refreshes + starts. Useful in scripts.
dev-fresh:
	@rm -rf .cache && $(MAKE) --no-print-directory refresh && $(MAKE) --no-print-directory dev-backend &
	@$(MAKE) --no-print-directory dev-frontend

# One-shot: wipe + refresh, no dev server.
fresh:
	@rm -rf .cache && $(MAKE) --no-print-directory refresh

dev-backend:
	@echo "Starting backend on http://localhost:$(BPORT)..."
	@$(PY) -m uvicorn backend.main:app --host 0.0.0.0 --port $(BPORT) --reload

dev-frontend:
	@echo "Starting frontend on http://localhost:$(FPORT)..."
	@cd frontend && npm run dev

# One-time data refresh — fetches all 8 sources into the local SQLite cache.
refresh:
	@echo "Refreshing data sources..."
	@$(PY) -c "import asyncio; from backend.db import init_db; from backend.services.refresh import refresh_all; \
	init_db(); \
	results = asyncio.run(refresh_all()); \
	[print(f'{r.source:25s} {r.status:8s} rows={r.rows_written:5d} ({r.duration_ms}ms) {r.error}') for r in results]"

stop:
	@-lsof -ti:$(BPORT) | xargs kill 2>/dev/null || true
	@-lsof -ti:$(FPORT) | xargs kill 2>/dev/null || true
	@echo "Stopped all servers"

build:
	@echo "Building frontend..."
	@cd frontend && npm run build

# Build frontend into backend/static and serve everything from one uvicorn on :8000.
package: build
	@rm -rf backend/static
	@cp -r frontend/dist backend/static
	@echo "Bundle copied to backend/static. Run 'make run' to start the single-port server."

run:
	@echo "Starting single-port server on http://localhost:$(BPORT) (frontend + API)..."
	@$(PY) -m uvicorn backend.main:app --host 0.0.0.0 --port $(BPORT)

test:
	@echo "Running backend tests..."
	@$(PY) -m pytest -xvs backend/tests/

clean: stop
	@-rm -rf .venv backend/__pycache__ backend/services/__pycache__ backend/routers/__pycache__ backend/models/__pycache__ backend/services/sources/__pycache__ backend/tests/__pycache__
	@-rm -rf frontend/node_modules frontend/dist backend/static
	@-rm -rf .cache
	@echo "Done"
