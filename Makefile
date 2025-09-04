format:
	uv run ruff format .

lint:
	uv run ruff check . --fix --preview

type-check:
	uv run mypy .

all: format lint type-check
