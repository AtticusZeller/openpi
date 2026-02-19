# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**openpi** is Physical Intelligence's open-source robotics library containing Vision-Language-Action (VLA) models: **π₀** (flow-based), **π₀-FAST** (autoregressive with FAST tokenizer), and **π₀.₅** (improved generalization with knowledge insulation). Pre-trained on 10k+ hours of robot data, supporting fine-tuning and zero-shot inference.

## Environment Requirements

- **OS**: Ubuntu 22.04 only (Windows/macOS not supported; use WSL2 or Docker on Windows)
- **Python**: >=3.11
- **Package Manager**: `uv` (not pip)
- **GPU**: NVIDIA required
  - Inference: >8GB (RTX 4090)
  - Fine-tuning LoRA: >22.5GB (RTX 4090)
  - Fine-tuning Full: >70GB (A100 80GB / H100)
- **Simulation**: MuJoCo-based (no Isaac Lab/Isaac Sim support)
  - ALOHA Sim: `gym-aloha`, MuJoCo, Python 3.10
  - LIBERO: RoboSuite + MuJoCo, Python 3.8
  - Headless rendering: `MUJOCO_GL=egl`

## Common Commands

```bash
# Installation
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Linting (ruff, line-length=120, target py311)
uv run ruff check src/                    # lint
uv run ruff check --fix src/              # lint with auto-fix
uv run ruff format src/                   # format

# Tests
uv run pytest                             # run all tests
uv run pytest src/openpi/models/pi0_test.py              # single test file
uv run pytest src/openpi/models/pi0_test.py::test_name   # single test
uv run pytest -m "not manual"             # skip manual-only tests

# Training (JAX - primary)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <CONFIG_NAME> --exp-name <RUN_NAME>

# Training (PyTorch)
uv run scripts/train_pytorch.py <CONFIG_NAME> --exp-name <RUN_NAME>

# Compute normalization stats (required before fine-tuning)
uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>

# Serve policy (WebSocket server on port 8000)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<CONFIG> --policy.dir=<CHECKPOINT_DIR>
# Shorthand with env defaults:
uv run scripts/serve_policy.py --env ALOHA_SIM

# Simulation without robot
uv run examples/simple_client/main.py --env ALOHA_SIM   # minimal test client
MUJOCO_GL=egl python examples/aloha_sim/main.py          # ALOHA sim
python examples/libero/main.py                            # LIBERO sim
```

## Architecture

### Core Source (`src/openpi/`)

- **`models/`** — JAX model implementations (Flax NNX)
  - `pi0.py`: π₀ flow-based model, `pi0_fast.py`: π₀-FAST autoregressive model
  - `gemma.py`: Gemma LLM backbone, `siglip.py`: SigLIP vision encoder
  - `tokenizer.py`: PaligemmaTokenizer + FASTTokenizer
  - `lora.py`: LoRA adapter support
- **`models_pytorch/`** — PyTorch implementations (mirrors JAX models)
  - `transformers_replace/`: Custom patches for AdaRMS normalization, precision control
- **`policies/`** — Environment-specific policy wrappers
  - `policy.py`: Base Policy class (applies transforms, calls model, returns action chunks)
  - `policy_config.py`: Policy instantiation from checkpoints (auto-detects JAX vs PyTorch)
  - Per-robot: `aloha_policy.py`, `droid_policy.py`, `libero_policy.py`
- **`training/`** — Training infrastructure
  - `config.py`: Central config registry (`_CONFIGS` dict) — all training/model configs live here
  - `data_loader.py`: LeRobot dataset integration
  - `sharding.py`: FSDP multi-GPU support
  - `checkpoints.py`: Orbax checkpoint save/load
- **`transforms.py`** — Modular data transform pipeline (`DataTransformFn` protocol, `CompositeTransform`)
- **`serving/websocket_policy_server.py`** — WebSocket inference server
- **`shared/`** — Utilities (array typing, normalization, image tools, GCS download)

### Client Package (`packages/openpi-client/`)

Standalone lightweight client (Python 3.7+) for robot-side inference. Includes WebSocket client, image utilities, msgpack serialization, and `Runtime`/`Environment` abstractions.

### Key Patterns

- **Config system**: All configs registered in `src/openpi/training/config.py` as a `_CONFIGS` dict. CLI uses `tyro` for typed argument parsing with config override.
- **Data pipeline**: Raw data → LeRobot format → Transform chain (repack → robot-specific → model-specific → normalization → tokenization) → Model input.
- **Client-server architecture**: Policy server (GPU machine) ↔ WebSocket ↔ Robot client. Decouples heavy inference from robot control.
- **Dual framework**: JAX (primary, Flax NNX) and PyTorch implementations coexist. Checkpoints can be converted via `examples/convert_jax_model_to_pytorch.py`.
- **Normalization**: z-score or quantile-based, computed via `compute_norm_stats.py` and bundled with checkpoints.
- **Array typing**: `jaxtyping` + custom `openpi.shared.array_typing` for runtime shape/type checking.
- **Checkpoints cached**: Auto-downloaded to `~/.cache/openpi` (configurable via `OPENPI_DATA_HOME`).

### Test Structure

Tests are co-located with source files using `_test.py` suffix. GPU auto-detected in `conftest.py` (falls back to CPU). Marker `manual` for tests requiring manual invocation.

## Code Style

- Ruff linter/formatter, line length 120, Python 3.11 target
- Force single-line imports (isort)
- `flax.nnx` for new JAX model definitions
- `DataTransformFn` protocol for data transforms
- Pre-commit hooks: `uv-lock`, `ruff --fix`, `ruff format`
- Excluded from linting: `docker/`, `third_party/`, `transformers_replace/`
