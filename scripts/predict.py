#!/usr/bin/env python3
"""CLI for quick inference on a single endoscopy image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mfurecnn.config import InferenceConfig
from mfurecnn.inference import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MFuRe-CNN inference on one image.")
    parser.add_argument("--model", required=True, type=Path, help="Path to model file (.h5 or SavedModel)")
    parser.add_argument("--image", required=True, type=Path, help="Path to test image")
    parser.add_argument("--classes", type=Path, default=None, help="Optional JSON class label mapping")
    parser.add_argument("--top-k", type=int, default=4, help="Number of predictions to return")
    args = parser.parse_args()

    cfg = InferenceConfig(
        model_path=args.model,
        image_path=args.image,
        class_names_path=args.classes,
        top_k=args.top_k,
    )
    output = predict_image(cfg)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
