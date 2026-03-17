"""Typed configuration objects for model assembly and inference."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Core model hyperparameters shared across MFuRe-style builds."""

    image_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    num_classes: int = 4
    dropout_rate: float = 0.1
    use_alpha_dropout: bool = True
    dense_units: int = 256
    backbone_trainable: bool = False


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime paths and thresholds used for image-level inference."""

    model_path: Path
    image_path: Path
    class_names_path: Path | None = None
    top_k: int = 4
