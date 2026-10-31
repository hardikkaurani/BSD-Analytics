.PHONY: help install lint test pipeline train dashboard clean

help:
	@echo "Available commands:"
	@echo "  make install    : Install production and dev dependencies"
	@echo "  make lint       : Run ruff linter and format check"
	@echo "  make test       : Run test suite with pytest"
	@echo "  make pipeline   : Execute complete data ingestion and preprocessing"
	@echo "  make train      : Train regression models and save best model"
	@echo "  make dashboard  : Launch Streamlit analytical dashboard"
	@echo "  make clean      : Clean temporary caches and builds"

install:
	pip install -e .[dev]

lint:
	ruff check .
	ruff format --check .

test:
	pytest tests/

pipeline:
	python scripts/run_pipeline.py

train:
	python scripts/train_model.py

dashboard:
	streamlit run dashboard/app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache __pycache__
