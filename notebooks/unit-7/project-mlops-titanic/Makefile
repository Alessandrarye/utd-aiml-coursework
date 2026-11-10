.PHONY: setup pipeline trigger watch test serve monitor mlflow repro

setup:            ## install dependencies and initialise Git + DVC tracking
	pip install -r requirements.txt
	git init -q 2>/dev/null || true
	dvc init -q 2>/dev/null || true
	dvc add data/raw/titanic.csv

pipeline:         ## run the full CI/CD pipeline
	python -m pipeline.run_pipeline

trigger:          ## run the pipeline only if data, code, or params changed
	python -m pipeline.trigger

watch:            ## poll for changes every 30 seconds
	python -m pipeline.trigger --watch 30

test:
	python -m pytest -q

serve:            ## requires: export API_KEY=...
	uvicorn api.main:app --host 0.0.0.0 --port 8000

monitor:
	python -m src.monitor

mlflow:           ## open the MLflow UI at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db

repro:            ## DVC: re-run only the stages whose inputs changed
	dvc repro
