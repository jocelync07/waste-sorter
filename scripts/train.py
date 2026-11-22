"""Train the waste classifier and save it for the web app.

This follows the final model from the COMP3420 assignment notebook:

1. A stratified 60/20/20 train/devtest/test split of waste_dataset.csv.
2. Optional extra HDPEM webcam images, added to the training set only.
3. Augmented HDPEM images, so every class has as many training images as the largest.
4. ResNet18 pre-trained on ImageNet with a frozen backbone; only the final layer
   is trained, and the epoch with the lowest devtest loss is kept.

Because the backbone is frozen, each image's 512 backbone features never change.
They are computed once, and the final layer is trained on them, which gives the
same model as the notebook in minutes on a CPU.

Example:
    python scripts/train.py --data-root "path/to/rawimgs" --csv "path/to/waste_dataset.csv" \
        --extra-hdpem "path/to/new_HDPEM"
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.model import build_model, preprocess  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# The augmentation used in the notebook (task 3.2), applied to HDPEM images only.
AUGMENT = transforms.Compose([
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)),
    transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=(128, 128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Folder holding the AluCan, Glass, HDPEM and PET image folders")
    parser.add_argument("--csv", type=Path, required=True,
                        help="waste_dataset.csv, with path and label columns")
    parser.add_argument("--extra-hdpem", type=Path, default=None,
                        help="Optional folder of extra HDPEM images (e.g. your webcam captures)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=config.MODEL_PATH)
    return parser.parse_args()


def read_rows(csv_path: Path, data_root: Path) -> list[tuple[Path, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = [(data_root / r["path"], r["label"]) for r in csv.DictReader(f)]
    missing = [p for p, _ in rows if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} images listed in the CSV are missing, e.g. {missing[0]}")
    unknown = {label for _, label in rows} - set(config.CLASS_NAMES)
    if unknown:
        sys.exit(f"Unknown labels in the CSV: {sorted(unknown)}")
    return rows


def stratified_split(rows, rng: random.Random):
    """Split each class 60/20/20, so every partition keeps the class proportions."""
    train, devtest, test = [], [], []
    for name in config.CLASS_NAMES:
        class_rows = [r for r in rows if r[1] == name]
        rng.shuffle(class_rows)
        n_train = round(0.6 * len(class_rows))
        n_devtest = round(0.2 * len(class_rows))
        train += class_rows[:n_train]
        devtest += class_rows[n_train:n_train + n_devtest]
        test += class_rows[n_train + n_devtest:]
    return train, devtest, test


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_training_images(train_rows, extra_hdpem: Path | None):
    """Training images as (PIL image, label), balanced with augmented HDPEM."""
    items = [(open_rgb(p), label) for p, label in train_rows]

    if extra_hdpem:
        extra = sorted(p for p in extra_hdpem.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        items += [(open_rgb(p), "HDPEM") for p in extra]
        print(f"Added {len(extra)} extra HDPEM images from {extra_hdpem}")

    counts = {name: sum(1 for _, label in items if label == name) for name in config.CLASS_NAMES}
    target = max(n for name, n in counts.items() if name != "HDPEM")
    hdpem = [image for image, label in items if label == "HDPEM"]
    needed = max(0, target - len(hdpem))
    items += [(AUGMENT(hdpem[i % len(hdpem)]), "HDPEM") for i in range(needed)]
    print(f"Training counts before augmentation: {counts}; added {needed} augmented HDPEM images")
    return items


@torch.inference_mode()
def extract_features(backbone: nn.Module, images, batch_size: int = 32, name: str = "") -> torch.Tensor:
    features = []
    start = time.time()
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(img) for img in images[i:i + batch_size]])
        features.append(backbone(batch))
        done = min(i + batch_size, len(images))
        print(f"\r  {name}: {done}/{len(images)} images ({time.time() - start:.0f}s)", end="", flush=True)
    print()
    return torch.cat(features)


def labels_tensor(labels) -> torch.Tensor:
    return torch.tensor([config.CLASS_NAMES.index(label) for label in labels])


def evaluate(head: nn.Linear, features, targets):
    with torch.no_grad():
        logits = head(features)
        loss = nn.functional.cross_entropy(logits, targets).item()
        predictions = logits.argmax(1)
    return loss, predictions


def metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    n = len(config.CLASS_NAMES)
    confusion = [[0] * n for _ in range(n)]
    for t, p in zip(targets.tolist(), predictions.tolist()):
        confusion[t][p] += 1
    per_class = {}
    f1s = []
    for i, name in enumerate(config.CLASS_NAMES):
        tp = confusion[i][i]
        actual = sum(confusion[i])
        predicted = sum(row[i] for row in confusion)
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    accuracy = (predictions == targets).float().mean().item()
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1s) / n, 4),
        "per_class": per_class,
        "confusion_matrix": {"classes": config.CLASS_NAMES, "rows_actual_cols_predicted": confusion},
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    rows = read_rows(args.csv, args.data_root)
    train_rows, devtest_rows, test_rows = stratified_split(rows, rng)
    print(f"Split: {len(train_rows)} train, {len(devtest_rows)} devtest, {len(test_rows)} test")

    print("Loading images...")
    train_items = build_training_images(train_rows, args.extra_hdpem)
    devtest_images = [open_rgb(p) for p, _ in devtest_rows]
    test_images = [open_rgb(p) for p, _ in test_rows]

    model = build_model(pretrained=True)
    backbone = model
    head_in = model.fc.in_features
    backbone.fc = nn.Identity()  # the backbone now outputs the 512 features
    backbone.eval()

    print("Computing backbone features (done once, as the backbone is frozen)...")
    x_train = extract_features(backbone, [img for img, _ in train_items], name="train")
    x_dev = extract_features(backbone, devtest_images, name="devtest")
    x_test = extract_features(backbone, test_images, name="test")
    y_train = labels_tensor([label for _, label in train_items])
    y_dev = labels_tensor([label for _, label in devtest_rows])
    y_test = labels_tensor([label for _, label in test_rows])

    head = nn.Linear(head_in, len(config.CLASS_NAMES))
    optimizer = torch.optim.Adam(head.parameters())
    best_loss, best_epoch, best_state = float("inf"), 0, None

    print("Training the final layer...")
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(x_train))
        for i in range(0, len(order), 32):
            batch = order[i:i + 32]
            loss = nn.functional.cross_entropy(head(x_train[batch]), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        dev_loss, dev_pred = evaluate(head, x_dev, y_dev)
        dev_acc = (dev_pred == y_dev).float().mean().item()
        marker = ""
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            marker = "  <- best"
        print(f"  epoch {epoch:>2}: devtest loss {dev_loss:.4f}, accuracy {dev_acc:.4f}{marker}")

    head.load_state_dict(best_state)
    test_loss, test_pred = evaluate(head, x_test, y_test)
    results = metrics(test_pred, y_test)
    results.update({
        "test_loss": round(test_loss, 4),
        "best_epoch": best_epoch,
        "devtest_loss": round(best_loss, 4),
        "train_images": len(train_items),
        "devtest_images": len(devtest_rows),
        "test_images": len(test_rows),
        "seed": args.seed,
    })

    # Put the trained layer back on the full model and save everything, so the
    # app can load it without downloading the ImageNet weights.
    final = build_model()
    final_state = {f"{k}": v for k, v in backbone.state_dict().items()}
    final_state.update({f"fc.{k}": v for k, v in head.state_dict().items()})
    final.load_state_dict(final_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final.state_dict(), args.out)
    metrics_path = args.out.with_name("metrics.json")
    metrics_path.write_text(json.dumps(results, indent=2))

    print(f"\nTest accuracy {results['accuracy']:.4f}, macro-F1 {results['macro_f1']:.4f}")
    for name, m in results["per_class"].items():
        print(f"  {name:<7} recall {m['recall']:.4f}  precision {m['precision']:.4f}")
    print(f"Saved model to {args.out} and metrics to {metrics_path}")


if __name__ == "__main__":
    main()
