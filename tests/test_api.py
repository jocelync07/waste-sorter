import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs startup, so the model is loaded
        yield c


def jpeg_bytes(color=(30, 120, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 300), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health_reports_untrained_model(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["classes"] == config.CLASS_NAMES


def test_pages_are_served(client):
    assert "Live Waste Sorter" in client.get("/").text
    assert "Model monitoring" in client.get("/monitor").text


def test_predict_returns_a_class(client):
    response = client.post(
        "/api/predict",
        files={"file": ("can.jpg", jpeg_bytes(), "image/jpeg")},
        data={"source": "webcam"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in config.CLASS_NAMES
    assert body["display_name"] == config.CLASS_LABELS[body["label"]]
    assert body["latency_ms"] >= 0


def test_predict_rejects_non_images(client):
    response = client.post("/api/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_predict_rejects_empty_files(client):
    response = client.post("/api/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_predict_rejects_large_files(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/api/predict", files={"file": ("big.jpg", jpeg_bytes(), "image/jpeg")})
    assert response.status_code == 413


def test_stats_count_predictions(client):
    before = client.get("/api/stats").json()["total"]
    client.post("/api/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
    client.post("/api/predict", files={"file": ("b.jpg", jpeg_bytes((10, 10, 10)), "image/jpeg")})
    stats = client.get("/api/stats").json()
    assert stats["total"] == before + 2
    assert sum(stats["by_class"].values()) == stats["total"]
    assert stats["recent"][0]["source"] == "upload"
    assert 0.0 <= stats["average_confidence"] <= 1.0
