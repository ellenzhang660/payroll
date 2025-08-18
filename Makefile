# Define your target folders
TARGETS = src

.PHONY: format lint

format:
	poetry run black $(TARGETS)
	poetry run isort $(TARGETS)
	poetry run ruff check $(TARGETS) --fix

lint:
	poetry run ruff $(TARGETS)

test:
	poetry run pytest -s