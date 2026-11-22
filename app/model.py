"""The ResNet18 classifier: preprocessing, loading weights, and prediction.

Preprocessing here is shared with scripts/train.py, so the model sees images
prepared the same way when it is trained and when it is served.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models

from . import config


def build_model(num_classes: int = len(config.CLASS_NAMES), pretrained: bool = False) -> nn.Module:
    """ResNet18 with a new final layer for our classes.

    pretrained=True downloads the ImageNet weights (used for training). Serving
    loads a full saved state dict instead, so it needs no download.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB image, honouring phone EXIF rotation."""
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def preprocess(image: Image.Image) -> torch.Tensor:
    """RGB image -> normalised 3x224x224 tensor, as in the assignment notebook."""
    image = image.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


class WasteClassifier:
    """Loads the model once and classifies single images."""

    def __init__(self, model_path: Path = config.MODEL_PATH, allow_untrained: bool = config.ALLOW_UNTRAINED):
        self.model = build_model()
        self.trained = False
        if model_path.exists():
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.trained = True
        elif not allow_untrained:
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run scripts/train.py first, "
                "or set ALLOW_UNTRAINED=1 to start with random weights."
            )
        self.model.eval()
        self.model_path = model_path

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        start = time.perf_counter()
        batch = preprocess(image).unsqueeze(0)
        probabilities = torch.softmax(self.model(batch), dim=1)[0].tolist()
        latency_ms = (time.perf_counter() - start) * 1000

        best = int(np.argmax(probabilities))
        label = config.CLASS_NAMES[best]
        confidence = probabilities[best]
        return {
            "label": label,
            "display_name": config.CLASS_LABELS[label],
            "confidence": round(confidence, 4),
            "low_confidence": confidence < config.LOW_CONFIDENCE,
            "probabilities": {
                name: round(p, 4) for name, p in zip(config.CLASS_NAMES, probabilities)
            },
            "latency_ms": round(latency_ms, 1),
        }
