# Architecture and Codebase Layout

This repository now supports two complementary workflows:

1. **Notebook-first research workflow** (paper-faithful):
   - `000` to `008` notebooks for training/evaluation/Grad-CAM.
2. **Package-first production workflow**:
   - `src/mfurecnn/` modules for reusable layers, model builders, inference, and utilities.

## Package modules

- `config.py`: strongly typed dataclass configs.
- `layers.py`: custom `FusionResidualBlock` implementation.
- `models.py`: reusable MFuRe-style model assembly.
- `inference.py`: stable prediction pipeline.
- `utils.py`: reproducibility and run naming helpers.

## Why this layout

- Separates experimental notebooks from reusable production code.
- Enables clean imports for scripts, APIs, and test harnesses.
- Makes future CI/CD integration easier (`pip install -e .`, lint, tests).

## Suggested next upgrades

- Add continuous integration for linting and smoke tests.
- Add model artifact registry integration.
- Add API service wrapper (FastAPI/Flask) around `scripts/predict.py`.
