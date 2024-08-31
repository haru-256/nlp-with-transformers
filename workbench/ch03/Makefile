.PHONY: lint fmt help
.DEFAULT_GOAL := help

lint: ## Run Linter
	uvx ruff@latest check .

fmt: ## Run formatter
	uvx ruff@latest check --fix .
	uvx ruff@latest format .

lock: ## Lock dependencies
	uv lock

setup: ## Setup the project
	uv sync

help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
