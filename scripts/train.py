"""Train the waste classifier, compare training settings, and save the best model.

Steps:

1. Split waste_dataset.csv 60/20/20 into train/devtest/test, separately for each
   class, with a fixed seed. The split is saved to models/splits/ so it can be
   checked and reused.
2. Add extra HDPEM photos to the training set only.
3. Augment HDPEM training images until every class has as many as the largest.
4. Train several configurations of ResNet18 (pre-trained on ImageNet) and compare
   them on the devtest set:
     - head-lr1e-3   frozen backbone, only the final layer trained, learning rate 1e-3
     - head-lr1e-4   the same with learning rate 1e-4
     - finetune-l4   fine-tuning: the last residual block (layer4) is also trained,
                     with a smaller learning rate (1e-4) than the final layer (1e-3)
5. Pick the configuration with the best devtest macro-F1, then evaluate it once
   on the test set and save it for the app.

Speed: conv1 to layer3 are frozen in every configuration, so their output for
each image never changes. It is computed once and cached; only layer4 and the
final layer run during training. This makes training practical on a laptop CPU.

Example:
    python scripts/train.py --data-root path/to/rawimgs --csv path/to/waste_dataset.csv \
        --extra-hdpem data/new_HDPEM
"""
from __future__ import annotations

import argparse
import copy
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
SPLIT_RATIOS = {"train": 0.6, "devtest": 0.2}  # test takes the remainder

# Augmentation for the HDPEM training images. Each transform imitates a change
# that happens in real use, without making an HDPE bottle look like another class:
# - ColorJitter: different room lighting and camera white balance.
# - RandomAffine: the bottle slightly rotated, off-centre, nearer or further away.
#   Mid-grey fill is closer to real backgrounds than black corners.
# - Horizontal flip: bottles have no meaningful left/right side.
# - Vertical flip: in a bin or on a sorting belt, containers lie upside down as
#   often as upright, and the label, cap and shape cues stay the same.
# - GaussianBlur: webcam frames are softer than the dataset's product photos.
AUGMENT = transforms.Compose([
    transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)),
    transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=(128, 128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
])

CONFIGS = {
    "head-lr1e-3": {"finetune_layer4": False, "head_lr": 1e-3, "layer4_lr": None},
    "head-lr1e-4": {"finetune_layer4": False, "head_lr": 1e-4, "layer4_lr": None},
    "finetune-l4": {"finetune_layer4": True, "head_lr": 1e-3, "layer4_lr": 1e-4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path,
                        help="Folder holding the AluCan, Glass, HDPEM and PET image folders")
    parser.add_argument("--csv", type=Path,
                        help="waste_dataset.csv, with path and label columns")
    parser.add_argument("--extra-hdpem", type=Path, default=None,
                        help="Optional folder of extra HDPEM photos, added to training only")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--head-epochs", type=int, default=30, help="Epochs for frozen-backbone runs")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="Epochs for fine-tuning runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--docs", type=Path, default=ROOT / "docs", help="Where to save the charts")
    parser.add_argument("--charts-only", action="store_true",
                        help="Redraw the charts from models/metrics.json without training")
    args = parser.parse_args()
    if not args.charts_only and (args.data_root is None or args.csv is None):
        parser.error("--data-root and --csv are required unless --charts-only is given")
    return args


def seed_everything(seed: int) -> random.Random:
    torch.manual_seed(seed)
    return random.Random(seed)


# ---------------------------------------------------------------- data

def read_rows(csv_path: Path, data_root: Path) -> list[tuple[str, str]]:
    """Rows of (relative path, label) from the dataset CSV, checked against disk."""
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = [(r["path"], r["label"]) for r in csv.DictReader(f)]
    missing = [p for p, _ in rows if not (data_root / p).exists()]
    if missing:
        sys.exit(f"{len(missing)} images listed in the CSV are missing, e.g. {missing[0]}")
    unknown = {label for _, label in rows} - set(config.CLASS_NAMES)
    if unknown:
        sys.exit(f"Unknown labels in the CSV: {sorted(unknown)}")
    return rows


def stratified_split(rows, rng: random.Random) -> dict[str, list]:
    """Split every class 60/20/20 on its own, so each partition keeps the class mix."""
    splits = {"train": [], "devtest": [], "test": []}
    for class_name in config.CLASS_NAMES:
        class_rows = sorted(r for r in rows if r[1] == class_name)  # sorted: same start order every run
        rng.shuffle(class_rows)

        n_train = round(SPLIT_RATIOS["train"] * len(class_rows))
        n_devtest = round(SPLIT_RATIOS["devtest"] * len(class_rows))

        splits["train"] += class_rows[:n_train]
        splits["devtest"] += class_rows[n_train:n_train + n_devtest]
        splits["test"] += class_rows[n_train + n_devtest:]
    return splits


def save_splits(splits: dict[str, list], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows(rows)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def class_counts(labels) -> dict[str, int]:
    labels = list(labels)  # may be a generator; it is read once per class
    return {name: labels.count(name) for name in config.CLASS_NAMES}


def build_training_set(train_rows, data_root: Path, extra_hdpem: Path | None):
    """Training images as (PIL image, label), plus the class counts at each stage."""
    items = [(open_rgb(data_root / p), label) for p, label in train_rows]
    stages = {"Original split": class_counts(label for _, label in items)}

    if extra_hdpem:
        extra = sorted(p for p in extra_hdpem.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        items += [(open_rgb(p), "HDPEM") for p in extra]
        stages["+ extra HDPEM photos"] = class_counts(label for _, label in items)
        print(f"Added {len(extra)} extra HDPEM photos from {extra_hdpem}")

    target = max(n for name, n in class_counts(l for _, l in items).items() if name != "HDPEM")
    hdpem = [image for image, label in items if label == "HDPEM"]
    needed = max(0, target - len(hdpem))
    items += [(AUGMENT(hdpem[i % len(hdpem)]), "HDPEM") for i in range(needed)]
    stages["+ augmentation"] = class_counts(label for _, label in items)
    print(f"Added {needed} augmented HDPEM images. Training counts: {stages['+ augmentation']}")
    return items, stages


# ---------------------------------------------------------------- model parts

def split_resnet(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """(frozen stem: conv1..layer3, trainable top: layer4 + pooling + final layer)."""
    stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                         model.layer1, model.layer2, model.layer3)
    top = nn.Sequential(model.layer4, model.avgpool, nn.Flatten(), model.fc)
    return stem, top


@torch.inference_mode()
def cache_stem_outputs(stem: nn.Module, images, name: str, batch_size: int = 32) -> torch.Tensor:
    """layer3 output for every image (256x14x14), stored as float16 to save memory."""
    outputs, start = [], time.time()
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(img) for img in images[i:i + batch_size]])
        outputs.append(stem(batch).half())
        print(f"\r  {name}: {min(i + batch_size, len(images))}/{len(images)} images "
              f"({time.time() - start:.0f}s)", end="", flush=True)
    print()
    return torch.cat(outputs)


def labels_tensor(labels) -> torch.Tensor:
    return torch.tensor([config.CLASS_NAMES.index(label) for label in labels])


@torch.inference_mode()
def predict_logits(top: nn.Module, cached: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    top.eval()
    return torch.cat([top(cached[i:i + batch_size].float()) for i in range(0, len(cached), batch_size)])


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict:
    n = len(config.CLASS_NAMES)
    confusion = [[0] * n for _ in range(n)]
    for t, p in zip(targets.tolist(), predictions.tolist()):
        confusion[t][p] += 1
    per_class, f1s = {}, []
    for i, name in enumerate(config.CLASS_NAMES):
        tp = confusion[i][i]
        actual, predicted = sum(confusion[i]), sum(row[i] for row in confusion)
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    return {
        "accuracy": round((predictions == targets).float().mean().item(), 4),
        "macro_f1": round(sum(f1s) / n, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def train_config(name: str, settings: dict, base_top: nn.Module, x_train, y_train, x_dev, y_dev,
                 epochs: int, seed: int) -> dict:
    """Train one configuration; keep the epoch with the lowest devtest loss."""
    torch.manual_seed(seed)  # same shuffling for every configuration
    top = copy.deepcopy(base_top)
    layer4, fc = top[0], top[3]

    for p in layer4.parameters():
        p.requires_grad = settings["finetune_layer4"]
    groups = [{"params": fc.parameters(), "lr": settings["head_lr"]}]
    if settings["finetune_layer4"]:
        groups.append({"params": layer4.parameters(), "lr": settings["layer4_lr"]})
    optimizer = torch.optim.Adam(groups)

    if settings["finetune_layer4"]:
        # layer4 changes every step, so it runs on the cached layer3 output.
        inputs_train, inputs_dev, trainable = x_train, x_dev, top
    else:
        # layer4 is frozen too, so its pooled output is also fixed: compute it
        # once and train only the final layer on it.
        frozen = top[:3].eval()
        with torch.inference_mode():
            inputs_train = torch.cat([frozen(x_train[i:i + 128].float()) for i in range(0, len(x_train), 128)])
            inputs_dev = torch.cat([frozen(x_dev[i:i + 128].float()) for i in range(0, len(x_dev), 128)])
        trainable = fc

    best = {"loss": float("inf"), "epoch": 0, "state": None}
    history = []
    print(f"\n[{name}] {epochs} epochs")
    for epoch in range(1, epochs + 1):
        start = time.time()
        trainable.train()
        layer4.eval()  # keep BatchNorm statistics from ImageNet; its weights still train
        order = torch.randperm(len(inputs_train))
        for i in range(0, len(order), 32):
            batch = order[i:i + 32]
            loss = nn.functional.cross_entropy(trainable(inputs_train[batch].float()), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trainable.eval()
        with torch.inference_mode():
            logits = trainable(inputs_dev.float()) if trainable is fc else predict_logits(top, x_dev)
        dev_loss = nn.functional.cross_entropy(logits, y_dev).item()
        dev = compute_metrics(logits.argmax(1), y_dev)
        history.append({"epoch": epoch, "devtest_loss": round(dev_loss, 4), **{k: dev[k] for k in ("accuracy", "macro_f1")}})
        marker = ""
        if dev_loss < best["loss"]:
            best = {"loss": dev_loss, "epoch": epoch, "state": copy.deepcopy(top.state_dict())}
            marker = "  <- best"
        print(f"  epoch {epoch:>2}: devtest loss {dev_loss:.4f}, accuracy {dev['accuracy']:.4f}, "
              f"macro-F1 {dev['macro_f1']:.4f} ({time.time() - start:.0f}s){marker}")

    top.load_state_dict(best["state"])
    logits = predict_logits(top, x_dev)
    dev = compute_metrics(logits.argmax(1), y_dev)
    return {
        "name": name,
        "settings": settings,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "devtest_loss": round(best["loss"], 4),
        "devtest": dev,
        "history": history,
        "top": top,
    }


# ---------------------------------------------------------------- charts

def save_charts(stages: dict, results: list[dict], chosen_name: str, test: dict, docs: Path) -> list[Path]:
    """Draw the three report charts from the numbers saved in metrics.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Training-set class counts before and after each balancing step.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(stages)
    xs = range(len(config.CLASS_NAMES))
    for i, (stage, counts) in enumerate(stages.items()):
        values = [counts[c] for c in config.CLASS_NAMES]
        bars = ax.bar([x + i * width for x in xs], values, width, label=stage)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks([x + width * (len(stages) - 1) / 2 for x in xs], config.CLASS_NAMES)
    ax.set_ylabel("Training images")
    ax.set_ylim(0, max(max(c.values()) for c in stages.values()) * 1.25)
    ax.set_title("Training-set class distribution before and after balancing")
    ax.legend(loc="upper center", ncol=len(stages), frameon=False)
    fig.tight_layout()
    path = docs / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # 2. Configurations compared on devtest.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["name"] for r in results]
    metrics = {
        "Accuracy": [r["devtest"]["accuracy"] for r in results],
        "Macro-F1": [r["devtest"]["macro_f1"] for r in results],
        "HDPEM recall": [r["devtest"]["per_class"]["HDPEM"]["recall"] for r in results],
    }
    width = 0.8 / len(metrics)
    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar([x + i * width for x in range(len(names))], values, width, label=metric)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks([x + width for x in range(len(names))], names)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Devtest score")
    ax.set_title(f"Configurations compared on devtest (chosen: {chosen_name})")
    ax.legend(loc="upper center", ncol=len(metrics), frameon=False)
    fig.tight_layout()
    path = docs / "config_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # 3. Test-set confusion matrix of the chosen model.
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    matrix = test["confusion_matrix"]
    ax.imshow(matrix, cmap="Greens")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center",
                    color="white" if value > max(map(max, matrix)) / 2 else "black")
    ax.set_xticks(range(len(config.CLASS_NAMES)), config.CLASS_NAMES)
    ax.set_yticks(range(len(config.CLASS_NAMES)), config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Test confusion matrix ({chosen_name})")
    fig.tight_layout()
    path = docs / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    return saved


# ---------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    metrics_path = args.out.parent / "metrics.json"
    if args.charts_only:
        report = json.loads(metrics_path.read_text())
        charts = save_charts(report["training_counts"], report["configurations"], report["chosen"],
                             report["test"], args.docs)
        print("Redrew " + ", ".join(str(c) for c in charts))
        return

    rng = seed_everything(args.seed)

    rows = read_rows(args.csv, args.data_root)
    splits = stratified_split(rows, rng)
    save_splits(splits, args.out.parent / "splits")
    print("Split: " + ", ".join(f"{len(v)} {k}" for k, v in splits.items())
          + f" (seed {args.seed}, saved to {args.out.parent / 'splits'})")

    print("Loading images...")
    train_items, stages = build_training_set(splits["train"], args.data_root, args.extra_hdpem)
    dev_images = [open_rgb(args.data_root / p) for p, _ in splits["devtest"]]
    test_images = [open_rgb(args.data_root / p) for p, _ in splits["test"]]

    model = build_model(pretrained=True)
    stem, base_top = split_resnet(model)
    stem.eval()

    print("Caching the frozen layers' output (conv1 to layer3), once per image...")
    x_train = cache_stem_outputs(stem, [img for img, _ in train_items], "train")
    x_dev = cache_stem_outputs(stem, dev_images, "devtest")
    x_test = cache_stem_outputs(stem, test_images, "test")
    y_train = labels_tensor([label for _, label in train_items])
    y_dev = labels_tensor([label for _, label in splits["devtest"]])
    y_test = labels_tensor([label for _, label in splits["test"]])

    results = []
    for name in args.configs:
        settings = CONFIGS[name]
        epochs = args.finetune_epochs if settings["finetune_layer4"] else args.head_epochs
        results.append(train_config(name, settings, base_top, x_train, y_train, x_dev, y_dev, epochs, args.seed))

    # Choose on devtest only: macro-F1 first (it weighs the rare HDPEM class
    # equally), devtest loss as the tie-breaker. The test set is used once, below.
    chosen = max(results, key=lambda r: (r["devtest"]["macro_f1"], -r["devtest_loss"]))
    print("\nDevtest comparison:")
    for r in results:
        d = r["devtest"]
        mark = "  <- chosen" if r is chosen else ""
        print(f"  {r['name']:<12} accuracy {d['accuracy']:.4f}  macro-F1 {d['macro_f1']:.4f}  "
              f"HDPEM recall {d['per_class']['HDPEM']['recall']:.4f}  (best epoch {r['best_epoch']}){mark}")

    test_logits = predict_logits(chosen["top"], x_test)
    test = compute_metrics(test_logits.argmax(1), y_test)
    test["loss"] = round(nn.functional.cross_entropy(test_logits, y_test).item(), 4)

    # Save the whole network (stem + trained top) so the app needs no download.
    final = build_model()
    final_stem, final_top = split_resnet(final)
    final_stem.load_state_dict(stem.state_dict())
    final_top.load_state_dict(chosen["top"].state_dict())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final.state_dict(), args.out)

    charts = save_charts(stages, results, chosen["name"], test, args.docs)

    report = {
        "seed": args.seed,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "training_counts": stages,
        "selection_rule": "highest devtest macro-F1, then lowest devtest loss",
        "configurations": [{k: v for k, v in r.items() if k != "top"} for r in results],
        "chosen": chosen["name"],
        "test": test,
    }
    metrics_path.write_text(json.dumps(report, indent=2))

    print(f"\nChosen: {chosen['name']}. Test accuracy {test['accuracy']:.4f}, macro-F1 {test['macro_f1']:.4f}")
    for name, m in test["per_class"].items():
        print(f"  {name:<7} recall {m['recall']:.4f}  precision {m['precision']:.4f}")
    print(f"Saved model to {args.out}, metrics to {args.out.parent / 'metrics.json'}, "
          f"charts to {', '.join(str(c) for c in charts)}")


if __name__ == "__main__":
    main()
