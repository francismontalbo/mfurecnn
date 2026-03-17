# Architecture and Codebase Layout

This repository supports two complementary workflows while staying aligned to the BSPC paper:

1. **Notebook-first research workflow** (paper-faithful):
   - `000` to `008` notebooks for training/evaluation/Grad-CAM.
2. **Package-first production workflow**:
   - `src/mfurecnn/` modules for reusable layers, model builders, inference, and utilities.
   - Includes both TensorFlow and PyTorch-style code paths.

## Package modules

### TensorFlow path (compatible with current notebook ecosystem)
- `config.py`: strongly typed dataclass configs.
- `layers.py`: custom `FusionResidualBlock` implementation.
- `models.py`: reusable MFuRe-style model assembly.
- `inference.py`: stable prediction pipeline.

### PyTorch path (professional deployment-friendly option)
- `torch_layers.py`: `FusionResidualBlockTorch`.
- `torch_models.py`: `MFuReTorch` and model factory.
- `torch_inference.py`: checkpoint loading + image inference.

### Shared utilities
- `utils.py`: reproducibility and run naming helpers.

## Why this layout

- Separates experimental notebooks from reusable production code.
- Enables clean imports for scripts, APIs, and test harnesses.
- Preserves paper alignment while allowing framework flexibility.
- Makes CI/CD integration easier (`pip install -e .`, lint, tests).

## Suggested next upgrades

- Add continuous integration for linting and smoke tests.
- Add model artifact registry integration.
- Add API service wrapper (FastAPI/Flask) for both CLIs.
