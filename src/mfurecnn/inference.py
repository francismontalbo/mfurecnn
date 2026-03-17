"""Inference helpers for running pretrained MFuRe/MFNR models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from .config import InferenceConfig
from .layers import FusionResidualBlock


def load_class_names(path: Path | None) -> list[str]:
    if path is None:
        return ["normal", "ulcer", "polyp", "esophagitis"]
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        return [payload[str(i)] for i in sorted(map(int, payload.keys()))]
    return list(payload)


def _preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = keras.preprocessing.image.load_img(image_path, target_size=image_size)
    arr = keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def predict_image(config: InferenceConfig, image_size: tuple[int, int] = (224, 224)) -> dict:
    class_names = load_class_names(config.class_names_path)

    model = keras.models.load_model(
        config.model_path,
        custom_objects={"FusionResidualBlock": FusionResidualBlock},
        compile=False,
    )
    x = _preprocess_image(config.image_path, image_size=image_size)
    probs = model.predict(x, verbose=0)[0]

    rank = np.argsort(probs)[::-1][: config.top_k]
    predictions = [
        {"class": class_names[idx], "index": int(idx), "probability": float(probs[idx])} for idx in rank
    ]
    return {"image": str(config.image_path), "predictions": predictions}
