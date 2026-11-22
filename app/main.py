"""FastAPI app: the webcam page, the prediction API and the monitoring API."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from . import config
from .model import WasteClassifier, load_image
from .monitor import PredictionLog

STATIC_DIR = config.ROOT / "app" / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request.
    state["classifier"] = WasteClassifier()
    state["log"] = PredictionLog()
    yield
    state.clear()


app = FastAPI(title="Live Waste Sorter", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/monitor", include_in_schema=False)
def monitor_page():
    return FileResponse(STATIC_DIR / "monitor.html")


@app.get("/api/health")
def health():
    classifier: WasteClassifier = state["classifier"]
    return {
        "status": "ok",
        "model_loaded": classifier.trained,
        "model_path": str(classifier.model_path.name),
        "classes": config.CLASS_NAMES,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), source: str = Form("upload")):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large.")
    try:
        image = load_image(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file is not a readable image.")

    result = state["classifier"].predict(image)
    state["log"].record(result, source="webcam" if source == "webcam" else "upload")
    return result


@app.get("/api/stats")
def stats():
    return state["log"].summary()
