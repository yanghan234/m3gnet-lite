# Gemini Code Assistant Context

This document provides context for the Gemini code assistant to understand the M3GNet project.

## Project Overview

This project is a Python-based re-implementation of the M3GNet (Materials Graph Network) architecture, a deep learning model for materials science. It is built using PyTorch and `torch-geometric` for graph-based deep learning. The primary purpose of this project is for learning and research.

The core of the project is the `M3GNet` model, which is a `torch.nn.Module`. The model takes graph representations of atomic structures as input and predicts properties like energy and forces.

### Key Technologies

*   **Programming Language:** Python 3.13
*   **Deep Learning Framework:** PyTorch
*   **Graph Neural Networks:** `torch-geometric`
*   **Materials Science Libraries:** `ase`, `pymatgen`
*   **Dependency Management:** `uv`
*   **Linting:** `ruff`

### Project Structure

*   `src/m3gnet`: Contains the core source code for the M3GNet model and its components.
    *   `m3gnet.py`: The main M3GNet model implementation.
    *   `layers/`: Different layers used in the M3GNet model.
    *   `graph/`: Code for converting atomic structures to graphs.
*   `scripts/`: Contains scripts for running the model and performing tests.
    *   `m3gnet/quick_run.py`: A script demonstrating how to load data, create a model, and run it.
*   `data_samples/`: Contains sample data for testing the model.
*   `pyproject.toml`: Defines project dependencies, metadata, and tool configurations.
*   `tests/`: Contains unit tests for the project.

## Building and Running

### Dependencies

To install the required dependencies, use `uv`:

```bash
uv pip install -e .
```

### Running the Model

The `scripts/m3gnet/quick_run.py` script provides a simple example of how to run the M3GNet model.

```bash
python scripts/m3gnet/quick_run.py
```

This script will:

1.  Load atomic structures from `data_samples/mpf-TP.xyz`.
2.  Convert the structures to graphs.
3.  Create a `M3GNet` model instance.
4.  Run the model on the data.
5.  Compute and print the forces.

### Testing

The project uses `pytest` for testing. To run the tests, execute the following command:

```bash
pytest
```

## Development Conventions

### Coding Style

The project uses `ruff` for linting and formatting. The configuration can be found in the `pyproject.toml` file. Key conventions include:

*   **Line Length:** 88 characters
*   **Indentation:** 4 spaces
*   **Quote Style:** Double quotes
*   **Docstring Convention:** Google style

### Contribution Guidelines

While there are no explicit contribution guidelines, the presence of a `.pre-commit-config.yaml` file suggests that pre-commit hooks are used to enforce code quality. It is recommended to install and use the pre-commit hooks for any new contributions.

```bash
pre-commit install
```
