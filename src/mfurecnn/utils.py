"""General utility helpers for reproducible MFuRe-CNN workflows."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int, *, deterministic_tf: bool = False) -> None:
    """Set random seed across Python/Numpy/TensorFlow (if installed)."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if importlib.util.find_spec("tensorflow") is None:
        return

    tf = importlib.import_module("tensorflow")
    tf.random.set_seed(seed)
    if deterministic_tf:
        tf.config.experimental.enable_op_determinism()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_run_name(architecture: str, condition: str, suffix: Optional[str] = None) -> str:
    run_name = f"{architecture.lower()}_{condition.lower()}"
    return run_name if not suffix else f"{run_name}_{suffix}"
