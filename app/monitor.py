"""A small prediction log, so the monitoring page can show how the model behaves.

Each prediction is one JSON line. Images are never stored, only the result.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import config


class PredictionLog:
    def __init__(self, path: Path = config.PREDICTION_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, result: dict, source: str) -> None:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "label": result["label"],
            "confidence": result["confidence"],
            "low_confidence": result["low_confidence"],
            "latency_ms": result["latency_ms"],
        }
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock, self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self, recent: int = 20) -> dict:
        entries = self._read()
        total = len(entries)
        if total == 0:
            return {
                "total": 0,
                "by_class": {name: 0 for name in config.CLASS_NAMES},
                "average_confidence": None,
                "low_confidence_rate": None,
                "average_latency_ms": None,
                "recent": [],
            }
        counts = Counter(e["label"] for e in entries)
        return {
            "total": total,
            "by_class": {name: counts.get(name, 0) for name in config.CLASS_NAMES},
            "average_confidence": round(sum(e["confidence"] for e in entries) / total, 4),
            "low_confidence_rate": round(sum(e["low_confidence"] for e in entries) / total, 4),
            "average_latency_ms": round(sum(e["latency_ms"] for e in entries) / total, 1),
            "recent": entries[-recent:][::-1],
        }
