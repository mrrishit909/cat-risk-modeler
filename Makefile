export DATABASE_URL ?= postgresql://app:app-dev-only@127.0.0.1:55330/app
PY = ./venv/bin/python
BASE = http://127.0.0.1:8330

bootstrap:      ## local virtualenv + database container
	python3 -m venv venv && ./venv/bin/pip install -q -r requirements-dev.txt
	docker compose up -d db
seed:           ## migrate and load the synthetic world (no-op if already seeded)
	$(PY) -m ct.seed --if-empty
dev:            ## API with reload on :8330, jobs run by a local worker
	($(PY) -m core.jobs ct.api &) && $(PY) -m uvicorn ct.api:app --reload --port 8330
test:
	$(PY) -m pytest --cov=ct --cov=core --cov-report=term-missing
demo:           ## the full stack in Docker, then the demo scenario against it
	docker compose up --build -d --wait && $(PY) -m core.scenario $(BASE)
load-test:      ## run `make demo` first
	$(PY) -m core.loadtest $(BASE) coastal-viewer-demo 32 10 "GET /v1/portfolios/a7854d50-edca-5ffd-b7fb-535901062e11/ep-curve" "GET /v1/portfolios/a7854d50-edca-5ffd-b7fb-535901062e11" 
evaluate:       ## held-out coasts -> docs/evaluation.md
	$(PY) -m ct.evaluate > docs/evaluation.md
reset:          ## drop all data (volume included)
	docker compose down -v
.PHONY: bootstrap seed dev test demo load-test evaluate reset
