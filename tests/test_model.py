import io

import pytest
import torch
from PIL import Image

from app import config
from app.model import WasteClassifier, build_model, load_image, preprocess


def make_image(size=(320, 240), color=(200, 30, 30)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_preprocess_shape_and_normalisation():
    tensor = preprocess(make_image())
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    # A mid-grey pixel should land near zero after ImageNet normalisation.
    grey = preprocess(make_image(color=(124, 116, 104)))
    assert grey.abs().max() < 0.05


def test_preprocess_converts_greyscale_to_rgb():
    tensor = preprocess(Image.new("L", (100, 100), 128))
    assert tensor.shape[0] == 3


def test_load_image_reads_png_bytes():
    buffer = io.BytesIO()
    make_image().save(buffer, format="PNG")
    image = load_image(buffer.getvalue())
    assert image.mode == "RGB"
    assert image.size == (320, 240)


def test_build_model_has_one_output_per_class():
    model = build_model().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    assert out.shape == (1, len(config.CLASS_NAMES))


def test_predict_returns_valid_probabilities():
    result = WasteClassifier().predict(make_image())
    assert result["label"] in config.CLASS_NAMES
    assert set(result["probabilities"]) == set(config.CLASS_NAMES)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["low_confidence"] == (result["confidence"] < config.LOW_CONFIDENCE)


def test_missing_model_fails_without_flag(tmp_path):
    with pytest.raises(FileNotFoundError):
        WasteClassifier(model_path=tmp_path / "missing.pth", allow_untrained=False)


def test_saved_weights_are_loaded(tmp_path):
    path = tmp_path / "model.pth"
    torch.save(build_model().state_dict(), path)
    classifier = WasteClassifier(model_path=path, allow_untrained=False)
    assert classifier.trained
