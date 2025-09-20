This repository is a personal re-implementation of the [M3GNet](https://www.nature.com/articles/s43588-022-00349-3) graph neural network for materials property prediction. It began as a cleanup of an older private codebase I wrote while studying the original model. The current version emphasizes clarity and faithfulness to the equations in the paper, serving as a reference for **experimentation and learning** rather than large-scale production use.

## Repository Layout
- `src/m3gnet/`: Core library modules (`graph/`, `layers/`, `datasets/`, `utils/`, and the PyTorch Lightning entry point `lightning.py`).
- `scripts/`: Helper scripts, including training entry points such as `scripts/training/training_lightning.py`.
- `tests/`: PyTest suites organized to mirror the source package layout.
- `data/`: Placeholder for local datasets or LMDB shards (ignored by Git).
- `notebooks/`: Exploratory notebooks for analysis and training notes.

## Getting Started
### Prerequisites
- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) for dependency management

### Installation
```bash
uv sync --dev
```

This command installs both runtime and development dependencies into a project-local environment managed by `uv`.

### Running Tests
```bash
uv run pytest
```

Please run all tests before committing changes or opening pull requests.

## Training
Launch the default PyTorch Lightning training loop against the MPF2021 dataset:

```bash
uv run python -m scripts.training.training_lightning
```

On first use, the dataset wrapper downloads the raw pickles from Figshare, shards them into LMDB format, and caches the processed results in `~/.cache/datasets/mpf2021`. The script automatically splits the data into train/validation/test sets, enables force and stress targets, and logs metrics to Weights & Biases (`project="m3gnet"`). You can adjust batch sizes, epochs, and logging configurations directly in the script or by extending it.


## Disclaimer
This repository is designed for learning and experimentation. It is not optimized for large-scale production training or deployment. Treat it as a reference starting point for your own materials modeling projects. Expect ongoing improvements as the legacy codebase continues to be refined.
