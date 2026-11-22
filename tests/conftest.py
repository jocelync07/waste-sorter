"""Test setup: run the app with random weights and a temporary prediction log.

The settings are read when app.config is imported, so they are set here first.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = Path(tempfile.mkdtemp(prefix="waste-sorter-tests-"))
os.environ["ALLOW_UNTRAINED"] = "1"
os.environ["MODEL_PATH"] = str(_tmp / "no-model.pth")
os.environ["PREDICTION_LOG"] = str(_tmp / "predictions.jsonl")
