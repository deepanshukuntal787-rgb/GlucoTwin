# GlucoTwin Makefile

PYTHON ?= python3

.PHONY: help data train baseline evaluate test test-api explain whatif serve serve-prod docker-build docker-up docker-down

help:
	@echo "GlucoTwin Pipeline & Service Commands:"
	@echo "  make serve        - Start FastAPI backend development server on :8000"
	@echo "  make serve-prod   - Start production ASGI server with multi-worker concurrency"
	@echo "  make test-api     - Run automated API endpoint & WebSocket integration tests"
	@echo "  make test         - Run TRD compliance and data leakage tests"
	@echo "  make docker-build - Build backend and frontend Docker container images"
	@echo "  make docker-up    - Launch full-stack container services with Docker Compose"
	@echo "  make docker-down  - Stop and tear down running containers"
	@echo "  make data         - Preprocess CSV datasets into sliding window tensors"
	@echo "  make train        - Train FusionTwinNet multimodal neural digital twin"
	@echo "  make baseline     - Train LightGBM feature-engineered baseline"
	@echo "  make evaluate     - Evaluate trained models on held-out test cohort"
	@echo "  make explain      - Run clinical explainability factor attribution"
	@echo "  make whatif       - Run counterfactual What-If scenario simulation"

serve:
	PYTHONPATH=. $(PYTHON) -m uvicorn gluco_twin.service.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	PYTHONPATH=. $(PYTHON) -m uvicorn gluco_twin.service.main:app --host 0.0.0.0 --port 8000 --workers 2

test-api:
	PYTHONPATH=. $(PYTHON) tests/test_api.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

data:
	PYTHONPATH=. $(PYTHON) -m gluco_twin.cli data

train:
	PYTHONPATH=. $(PYTHON) -m gluco_twin.models.train --epochs 15 --batch-size 256

baseline:
	PYTHONPATH=. $(PYTHON) -m gluco_twin.models.baseline_lgbm

evaluate:
	PYTHONPATH=. $(PYTHON) -m gluco_twin.eval.evaluate

test:
	PYTHONPATH=. $(PYTHON) tests/test_model_pipeline.py

explain:
	PYTHONPATH=. $(PYTHON) gluco_twin/cli.py explain --patient-id P086

whatif:
	PYTHONPATH=. $(PYTHON) gluco_twin/cli.py whatif --patient-id P086 --carbs 45 --steps 2500
