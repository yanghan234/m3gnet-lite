# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/m3gnet/`, grouped by concern (`graph/`, `layers/`, `datasets/`, `utils/`, and the Lightning entry point `lightning.py`). Tests mirror the package layout under `tests/m3gnet/graph/`. Use `scripts/` for runnable utilities (training entry points, experiment helpers), and keep datasets or LMDB shards in `data/`. Notebooks belong in `notebooks/`; avoid checking in generated artifacts outside `wandb/` runs.

## Build, Test, and Development Commands
Run `uv sync` once to install all runtime and dev dependencies. Use `uv run pytest` to execute the full test suite. Apply formatting and linting with `uv run ruff check --fix` and `uv run ruff format` before opening a PR. During model work, prefer `uv run python -m scripts.training.train` (see module docstrings for arguments).

## Coding Style & Naming Conventions
Python modules follow 4-space indentation and 88-character lines (enforced by Ruff). Import order is managed automatically; keep first-party modules under the `m3gnet` namespace. Use descriptive, lowercase_with_underscore names for functions and tests, PascalCase for classes, and annotate public functions with type hints. Include short Google-style docstrings for new modules and complex functions.

## Testing Guidelines
Author tests with PyTest; place new suites alongside implementation modules (e.g., `tests/m3gnet/<module>/test_<feature>.py`). Name tests `test_<action>_<case>` for clarity, and rely on fixtures for reusable tensors or converters. Ensure every new model path or graph utility has at least one assertion covering edge cases such as periodic boundary conditions. Run `uv run pytest` locally and avoid skipping tests unless a clear TODO documents the follow-up.

## Commit & Pull Request Guidelines
Commit messages should be concise, present-tense summaries (e.g., "add wandb logger"). Group related changes together and keep commits focused. For PRs, provide a short problem statement, outline the solution, list verification steps (tests, training runs), and link any tracking issues or experiment dashboards. Include screenshots or metric tables when modifying notebooks or training flows.

## Experiment & Data Notes
Store training configuration files under `scripts/` or `notebooks/` and reference external datasets by relative paths. Keep sensitive API keys out of the repo; configure `wandb` credentials via environment variables or local profiles. Document non-trivial data preprocessing steps in the README or in-line within notebooks.
