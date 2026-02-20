# GEMINI.md

This project, **openpi**, is Physical Intelligence's open-source robotics library containing Vision-Language-Action (VLA) models: **π₀**, **π₀-FAST**, and **π₀.₅**. It supports both JAX (primary) and PyTorch implementations.

## Project Overview

- **Purpose:** Open-source Vision-Language-Action (VLA) models for robotics.
- **Key Models:**
  - **π₀**: Flow-based VLA.
  - **π₀-FAST**: Autoregressive VLA using the FAST action tokenizer.
  - **π₀.₅**: Improved generalization with knowledge insulation.
- **Technologies:** JAX (Flax NNX), PyTorch, Python 3.11+, `uv` for dependency management.
- **Architecture:** 
  - **Core:** `src/openpi/` contains JAX models, policies, training infrastructure, and data transforms.
  - **PyTorch:** `src/openpi/models_pytorch/` contains PyTorch implementations.
  - **Client:** `packages/openpi-client/` is a lightweight Python client for robot-side inference.
  - **Scripts:** `scripts/` contains entry points for training, serving, and utility tasks.

## Building and Running

### Environment Setup
Requires Ubuntu 22.04 and NVIDIA GPU.

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Install dependencies (using uv)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Key Commands

- **Training (JAX):** `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <CONFIG_NAME> --exp-name <RUN_NAME>`
- **Training (PyTorch):** `uv run scripts/train_pytorch.py <CONFIG_NAME> --exp-name <RUN_NAME>`
- **Serve Policy:** `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<CONFIG> --policy.dir=<CHECKPOINT_DIR>`
- **Compute Norm Stats:** `uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>` (Required before fine-tuning)
- **Running Tests:** `uv run pytest`
- **Linting:** `uv run ruff check src/`
- **Formatting:** `uv run ruff format src/`

## Development Conventions

- **Python Standards:** Python >= 3.11. Use `uv` for all package management.
- **Linting & Formatting:** 
  - `ruff` is used with a line length of 120.
  - Single-line imports are enforced (`tool.ruff.lint.isort.force-single-line = true`).
  - Strict typing using `jaxtyping` and `beartype`.
- **Testing:** 
  - Tests are co-located with source files (e.g., `src/openpi/models/pi0_test.py`).
  - Use `pytest`. GPU is auto-detected; tests fall back to CPU if no GPU is available.
- **Configuration:** 
  - Centralized in `src/openpi/training/config.py`.
  - CLI uses `tyro` for argument parsing.
- **Data Transforms:** Modular pipeline using `DataTransformFn` protocol.
- **Checkpointing:** JAX uses Orbax; PyTorch uses native PyTorch format. Checkpoints are cached in `~/.cache/openpi`.

## Important Directories

- `src/openpi/models/`: JAX model definitions (Flax NNX).
- `src/openpi/models_pytorch/`: PyTorch model definitions.
- `src/openpi/policies/`: Environment-specific policy wrappers (ALOHA, DROID, LIBERO).
- `src/openpi/training/`: Training logic, config registry, and data loaders.
- `scripts/`: Training, serving, and data utility scripts.
- `packages/openpi-client/`: Lightweight client for remote inference.
- `examples/`: Example scripts and notebooks for different robot platforms.
