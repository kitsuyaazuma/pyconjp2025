format:
	uv run ruff format .

lint:
	uv run ruff check . --fix --preview

type-check:
	uv run mypy .

check: format lint type-check

gil:
	uv python pin 3.14
	uv venv -p 3.14 --clear
	uv run python -VV

nogil:
	uv python pin 3.14t
	uv venv -p 3.14t --clear
	uv run python -VV

