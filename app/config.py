"""Settings, read from environment variables so Docker and tests can change them."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Order matters: index i of the model output is CLASS_NAMES[i].
CLASS_NAMES = ["AluCan", "Glass", "HDPEM", "PET"]

# Friendly names shown on the page.
CLASS_LABELS = {
    "AluCan": "Aluminium can",
    "Glass": "Glass bottle",
    "HDPEM": "HDPE milk bottle",
    "PET": "PET plastic bottle",
}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "waste_resnet18.pth"))

# With no trained weights, the app refuses to start unless this is set. Tests and
# UI work use it; the predictions are then random, and /api/health says so.
ALLOW_UNTRAINED = os.getenv("ALLOW_UNTRAINED", "0") == "1"

# Every prediction is appended here as one JSON line, for the monitoring page.
PREDICTION_LOG = Path(os.getenv("PREDICTION_LOG", ROOT / "logs" / "predictions.jsonl"))

# Below this confidence a prediction is flagged as uncertain.
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.6"))

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
