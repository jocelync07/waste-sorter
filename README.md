# Live Waste Sorter

[![CI](https://github.com/jocelync07/waste-sorter/actions/workflows/ci.yml/badge.svg)](https://github.com/jocelync07/waste-sorter/actions/workflows/ci.yml)

Point a webcam at a drink container and the app says what it is (**aluminium can, glass bottle, HDPE milk bottle or PET plastic bottle**) in real time, so it can go in the right recycling bin.

It takes an image classifier I built for a university computer-vision assignment and turns it into a deployed web app, with an API, tests, a Docker image, CI and a monitoring page.

<!-- Add a screenshot or GIF here: docs/demo.gif -->

## Results

Served model: ResNet18 transfer learning, with 150 extra webcam images of HDPE bottles and augmentation to balance the classes. Scores are on a held-out test set of 620 images the model never saw (seed 42, from `models/metrics.json`).

| Class | Recall | Precision |
|---|---|---|
| Aluminium can | 0.965 | 0.955 |
| Glass bottle | 0.930 | 0.916 |
| HDPE milk bottle | 0.750 | 0.938 |
| PET plastic bottle | 0.965 | 0.970 |
| **Overall** | **Accuracy 0.947** | **Macro-F1 0.921** |

HDPE is still the hardest class: the test set has only 20 HDPE images, so its recall moves a lot with each mistake. More real HDPE photos would help more than more augmentation.

## How it works

```
 Browser (webcam or upload)
        │  JPEG frame, once a second
        ▼
 FastAPI  POST /api/predict ──▶ preprocess (224×224, ImageNet normalisation)
        │                         ▼
        │                    ResNet18 (PyTorch, CPU) ──▶ class + probabilities
        │                         │
        ▼                         ▼
 GET /api/stats  ◀── prediction log (JSON lines: class, confidence, latency; no images)
        │
        ▼
 /monitor page: prediction counts, average confidence, low-confidence rate, latency
```

- **Model:** ResNet18 pre-trained on ImageNet, backbone frozen, final layer trained on the [Drinking Waste Classification](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification) dataset. The rare HDPE class was boosted with my own webcam photos and data augmentation.
- **Training:** `scripts/train.py` caches the frozen backbone's features once, so the whole model trains in minutes on a laptop CPU.
- **Serving:** FastAPI loads the model once at startup. Uploads are checked for size and type; unreadable files get a 400, not a crash.
- **Monitoring:** every prediction is logged (without the image) and summarised on `/monitor`, so you can spot when the model is often unsure.

| Endpoint | What it does |
|---|---|
| `GET /` | Webcam and upload page |
| `GET /monitor` | Monitoring page |
| `POST /api/predict` | Classify one image (multipart field `file`) |
| `GET /api/stats` | Prediction statistics |
| `GET /api/health` | Status and whether a trained model is loaded |
| `GET /docs` | Interactive API docs (Swagger) |

## Run it locally

Needs Python 3.9 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
```

### 1. Train the model

Download the dataset and `waste_dataset.csv`, then:

```bash
python scripts/train.py --data-root path/to/rawimgs --csv path/to/waste_dataset.csv --extra-hdpem data/new_HDPEM
```

This writes `models/waste_resnet18.pth` (about 45 MB) and `models/metrics.json`.

### 2. Start the app

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 and allow camera access. Browsers only allow the camera on `localhost` or HTTPS.

To work on the page without a trained model, start it with `ALLOW_UNTRAINED=1` (predictions are then random and the page says so).

### 3. Run the tests

```bash
pytest -v
```

The tests use random weights, so they need no dataset and run in CI.

## Docker

```bash
docker build -t waste-sorter .
docker run -p 8000:8000 waste-sorter
```

## Deploy

The image reads the `PORT` variable, so it runs on most container hosts. For example, on **Render**: create a *Web Service* from this GitHub repo, choose *Docker*, and deploy. On **AWS**, push the image to ECR and run it on App Runner.

## Project structure

```
app/
  main.py        FastAPI routes
  model.py       ResNet18, preprocessing, prediction
  monitor.py     prediction log and statistics
  config.py      settings from environment variables
  static/        webcam page and monitoring page
scripts/train.py training, evaluation and export
tests/           pytest tests for the model and the API
.github/workflows/ci.yml   tests and Docker build on every push
```

## Limitations

- The model only knows four classes, so anything else (a hand, a cup, an empty desk) is still given one of them, sometimes with high confidence. A fifth "not a container" class, or rejecting low-confidence frames, would fix this.
- The training photos are mostly clean product shots, so busy backgrounds and poor light lower the confidence. The monitoring page shows this as a higher low-confidence rate.

## What I learned

- Taking a model from a notebook to a service: one shared preprocessing function for training and serving, loading the model once, and validating inputs.
- Why monitoring matters: a lab test-set score does not show how the model does on a real webcam, but confidence and low-confidence rates do.
- Packaging with Docker and checking every change with CI.
