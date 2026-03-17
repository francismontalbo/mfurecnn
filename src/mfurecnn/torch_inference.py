"""PyTorch inference helpers for MFuRe-style models."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .torch_models import MFuReTorch, TorchModelConfig, build_mfure_torch_classifier


DEFAULT_CLASSES = ["normal", "ulcer", "polyp", "esophagitis"]


def load_class_names(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_CLASSES
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        return [payload[str(i)] for i in sorted(map(int, payload.keys()))]
    return list(payload)


def load_torch_checkpoint(checkpoint_path: Path, device: torch.device, config: TorchModelConfig) -> MFuReTorch:
    model = build_mfure_torch_classifier(config)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_image_torch(
    checkpoint_path: Path,
    image_path: Path,
    classes_path: Path | None = None,
    top_k: int = 4,
    image_size: int = 224,
    device: str = "cpu",
) -> dict:
    class_names = load_class_names(classes_path)
    torch_device = torch.device(device)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(torch_device)

    model = load_torch_checkpoint(checkpoint_path, torch_device, TorchModelConfig(num_classes=len(class_names)))

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    values, indices = torch.topk(probs, k=min(top_k, len(class_names)))
    predictions = [
        {"class": class_names[idx.item()], "index": idx.item(), "probability": float(prob.item())}
        for prob, idx in zip(values, indices)
    ]
    return {"image": str(image_path), "predictions": predictions, "framework": "pytorch"}
