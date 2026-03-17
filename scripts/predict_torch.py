#!/usr/bin/env python3
"""CLI for PyTorch single-image inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mfurecnn.torch_inference import predict_image_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PyTorch MFuRe-style inference on one image.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to .pt/.pth checkpoint")
    parser.add_argument("--image", required=True, type=Path, help="Path to test image")
    parser.add_argument("--classes", type=Path, default=None, help="Optional JSON class label mapping")
    parser.add_argument("--top-k", type=int, default=4, help="Number of predictions to return")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    args = parser.parse_args()

    output = predict_image_torch(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        classes_path=args.classes,
        top_k=args.top_k,
        device=args.device,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
