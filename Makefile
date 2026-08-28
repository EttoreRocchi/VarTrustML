.PHONY: help install dev test lint format pre-commit lock docs docs-serve \
       docker-build docker-test dist dist-check dist-test-install \
       publish-test publish release clean

DOCKER_IMAGE := vartrustml
DOCKER_TAG   := $(shell python -c "from vartrustml._version import __version__; print(__version__)")

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Python development

install: ## Install package
	@pip install .

dev: ## Install package in editable mode with dev dependencies
	@pip install -e ".[dev]"

test: ## Run tests
	@pytest tests/ -v --tb=short

lint: ## Check linting and formatting (matches CI)
	@ruff check .
	@ruff format --check .

format: ## Auto-fix linting issues and format code
	@ruff check --fix .
	@ruff format .

pre-commit: ## Run all pre-commit hooks
	@pre-commit run --all-files

# Dependencies

lock: ## Regenerate requirements.lock.txt
	@pip-compile --output-file=requirements.lock.txt requirements.txt

# Documentation

docs: ## Build HTML documentation
	@sphinx-build -b html docs docs/_build/html

docs-serve: docs ## Build and serve docs locally on port 8000
	@python -m http.server 8000 --directory docs/_build/html

# Docker

docker-build: ## Build Docker image
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-test: ## Smoke-test Docker image
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) version
	@docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) smoke-test

# PyPI distribution

dist: clean ## Build sdist and wheel
	@python -m build

dist-check: dist ## Validate distribution packages
	@twine check dist/*

publish-test: dist-check ## Upload to TestPyPI
	@twine upload --repository testpypi dist/*

publish: dist-check ## Upload to PyPI
	@twine upload dist/*


# Cleanup

clean: ## Remove build artifacts
	@rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache docs/_build
