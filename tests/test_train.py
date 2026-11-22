"""Tests for the training helpers: a reproducible split and correct metrics."""
import importlib.util
import random
from pathlib import Path

import torch

from app import config

_spec = importlib.util.spec_from_file_location("train", Path(__file__).resolve().parent.parent / "scripts" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def fake_rows():
    sizes = {"AluCan": 100, "Glass": 100, "HDPEM": 10, "PET": 100}
    return [(f"{name}/{i}.jpg", name) for name, n in sizes.items() for i in range(n)]


def test_split_is_reproducible_with_the_same_seed():
    a = train.stratified_split(fake_rows(), random.Random(42))
    b = train.stratified_split(list(reversed(fake_rows())), random.Random(42))
    assert a == b  # same seed gives the same split, whatever the CSV row order


def test_split_changes_with_a_different_seed():
    a = train.stratified_split(fake_rows(), random.Random(42))
    b = train.stratified_split(fake_rows(), random.Random(7))
    assert a["train"] != b["train"]


def test_split_is_stratified_and_complete():
    rows = fake_rows()
    splits = train.stratified_split(rows, random.Random(0))
    assert sorted(splits["train"] + splits["devtest"] + splits["test"]) == sorted(rows)
    for name, total in {"AluCan": 100, "HDPEM": 10}.items():
        counts = {k: sum(1 for _, label in v if label == name) for k, v in splits.items()}
        assert counts == {"train": round(0.6 * total), "devtest": round(0.2 * total),
                          "test": total - round(0.6 * total) - round(0.2 * total)}


def test_class_counts_accepts_a_generator():
    labels = ["AluCan", "Glass", "Glass", "PET"]
    assert train.class_counts(label for label in labels) == {"AluCan": 1, "Glass": 2, "HDPEM": 0, "PET": 1}


def test_metrics_on_a_known_example():
    # Actual:    AluCan AluCan Glass HDPEM PET
    # Predicted: AluCan Glass  Glass HDPEM PET
    targets = torch.tensor([0, 0, 1, 2, 3])
    predictions = torch.tensor([0, 1, 1, 2, 3])
    m = train.compute_metrics(predictions, targets)
    assert m["accuracy"] == 0.8
    assert m["per_class"]["AluCan"] == {"precision": 1.0, "recall": 0.5, "f1": round(2 / 3, 4)}
    assert m["per_class"]["Glass"]["precision"] == 0.5
    assert m["confusion_matrix"][0] == [1, 1, 0, 0]
    assert len(m["confusion_matrix"]) == len(config.CLASS_NAMES)


def test_split_resnet_matches_the_full_model():
    from app.model import build_model
    model = build_model().eval()
    stem, top = train.split_resnet(model)
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        assert torch.allclose(top(stem(x)), model(x), atol=1e-5)
