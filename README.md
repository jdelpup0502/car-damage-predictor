<div align="center">

# 🚗 Car Damage Severity Predictor

**Deep learning model that classifies car damage as minor, moderate, or severe from a single photo.**

Built with EfficientNet-B0 transfer learning · Served via FastAPI · Next.js frontend · Deployable on Railway

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## How It Works

Upload a photo of a damaged car → get back a severity classification with confidence scores, uncertainty, and an optional Grad-CAM heatmap showing which regions influenced the prediction.

```
POST /predict  →  { "prediction": "03-severe", "confidence": 0.92, "uncertainty": 0.26 }
```

The model uses **EfficientNet-B0** pretrained on ImageNet, fine-tuned on ~1,400 labeled car damage images using a two-phase transfer learning strategy. A second **curriculum-trained** model is also included and can be used standalone or in an ensemble.

---

## Example Predictions

<table>
<tr>
<td align="center" width="33%">
<img src="examples/minor_example.jpg" width="280"/>
<br><br>
<b>✅ Minor</b><br>
Confidence: <code>93.4%</code>
<br>
<sub>minor: 93.4% · moderate: 6.5% · severe: 0.1%</sub>
</td>
<td align="center" width="33%">
<img src="examples/moderate_example.jpg" width="280"/>
<br><br>
<b>✅ Moderate</b><br>
Confidence: <code>62.7%</code>
<br>
<sub>minor: 13.5% · moderate: 62.7% · severe: 23.8%</sub>
</td>
<td align="center" width="33%">
<img src="examples/severe_example_1.jpg" width="280"/>
<br><br>
<b>✅ Severe</b><br>
Confidence: <code>91.9%</code>
<br>
<sub>minor: 0.1% · moderate: 8.1% · severe: 91.9%</sub>
</td>
</tr>
</table>

---

## Model Performance

Trained on the [Car Damage Severity Dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset) with class-weighted loss to handle the ambiguity of moderate damage.

| Metric | Minor | Moderate | Severe |
|--------|-------|----------|--------|
| Precision | 0.78 | 0.55 | 0.84 |
| Recall | 0.84 | 0.61 | 0.71 |
| F1-Score | 0.81 | 0.58 | 0.77 |

**Overall Accuracy: 73%**

> The model excels at distinguishing minor and severe damage. Moderate is inherently ambiguous (a scratched fender vs. a dented panel is subjective even for humans), which is reflected in the lower moderate scores.

The Kaggle notebook for how the model was built can be found [here](https://www.kaggle.com/code/justindelpup/car-damage-pre).

---

## Architecture

```
Input (224×224×3)
  → EfficientNet-B0 backbone (pretrained on ImageNet)
  → Global Average Pooling
  → Dense(256) + Dropout(0.5)
  → Dense(128) + Dropout(0.3)
  → Dense(3) + Softmax → [minor, moderate, severe]
```

### Standard Training

1. **Phase 1** — Freeze backbone, train classification head (lr=1e-3, 10 epochs)
2. **Phase 2** — Unfreeze top 50 backbone layers, fine-tune (lr=5e-5, 30 epochs with early stopping)

### Curriculum Learning Training

For improved moderate class accuracy, use multi-stage curriculum learning:

1. **Stage 1** — Binary: Minor vs Non-minor (learns clear patterns first)
2. **Stage 2** — Binary: Severe vs Non-severe
3. **Stage 3A** — 3-class frozen backbone, curriculum sampling (gradual introduction of moderate)
4. **Stage 3B** — Unfreeze top 50 EfficientNet layers, full end-to-end fine-tune

Results are saved to `models_curriculum/`. Both models are available at runtime via the model registry.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/jdelpup0502/car-damage-predictor.git
cd car-damage-predictor
pip install -r requirements.txt
```

### 2. Start the API

```bash
python api_server.py
```

Server runs at `http://localhost:8000`. Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 3. Make a prediction

```bash
curl -X POST http://localhost:8000/predict -F "file=@your_car_image.jpg"
```

### Example response

```json
{
  "model_version": "v1",
  "prediction": "03-severe",
  "confidence": 0.9187,
  "all_probabilities": {
    "01-minor": 0.0005,
    "02-moderate": 0.0807,
    "03-severe": 0.9187
  },
  "inference_time_ms": 111.05
}
```

### 4. Start the frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` to use the drag-and-drop web UI.

---

## API Endpoints

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict/batch` | Multiple images in one request |
| `POST` | `/predict/ensemble` | Weighted ensemble across model versions |
| `GET` | `/health` | Health check (`{"status":"ok"}`) |
| `GET` | `/docs` | Interactive Swagger UI |

**`/predict` query params:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `heatmap` | bool | false | Include Grad-CAM heatmap overlay (base64 PNG) |
| `uncertainty` | bool | false | Include predictive entropy score [0–1] |
| `tta` | bool | false | Test-time augmentation (8 views averaged, ~8× slower) |
| `version` | string | active | Model version to use (`v1`, `v2-curriculum`) |

```bash
# Prediction with heatmap + uncertainty
curl -X POST "http://localhost:8000/predict?heatmap=true&uncertainty=true" \
  -F "file=@image.jpg"

# Prediction with test-time augmentation
curl -X POST "http://localhost:8000/predict?tta=true" -F "file=@image.jpg"

# Ensemble prediction
curl -X POST "http://localhost:8000/predict/ensemble?ensemble_name=real-ensemble" \
  -F "file=@image.jpg"
```

### Model Registry

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/models` | List all versions and ensembles |
| `POST` | `/models/register` | Register a new model version |
| `POST` | `/models/{version}/activate` | Hot-swap the active model |
| `POST` | `/models/ensembles/{name}` | Create or update a named ensemble |
| `DELETE` | `/models/ensembles/{name}` | Delete an ensemble |

### Hard Example Mining

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/hard-examples` | List uncertain/wrong predictions |
| `GET` | `/hard-examples/stats` | Summary statistics |
| `POST` | `/hard-examples/{id}/label` | Add a ground-truth label |
| `POST` | `/hard-examples/export` | Export a labelled dataset for retraining |

```bash
# View the most difficult predictions
curl "http://localhost:8000/hard-examples?limit=20&only_wrong=true"

# Label a prediction for retraining
curl -X POST "http://localhost:8000/hard-examples/<id>/label" \
  -H "Content-Type: application/json" -d '"02-moderate"'

# Export as a retraining dataset
curl -X POST "http://localhost:8000/hard-examples/export?strategy=uncertainty&output_dir=hard_examples/dataset"
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Ensemble inference** | Weighted average across multiple model versions |
| **Uncertainty estimation** | Predictive entropy — 0 = certain, 1 = maximally uncertain |
| **Grad-CAM heatmaps** | Visualise which image regions drove the prediction |
| **Test-time augmentation** | Average 8 augmented views for more robust predictions |
| **Hard example mining** | Auto-log uncertain/wrong predictions to SQLite for targeted retraining |
| **Model hot-swap** | Switch active model version without restarting the server |
| **Curriculum learning** | Multi-stage training strategy to improve the ambiguous moderate class |
| **Web frontend** | Next.js drag-and-drop UI with probability bars and heatmap viewer |

---

## Deployment (Docker + Railway)

### Backend

```bash
# Build
docker build -t car-damage-api .

# Run locally
docker run -p 8000:8000 -e CORS_ORIGINS="http://localhost:3000" car-damage-api
```

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port to listen on |
| `ENV` | — | Set to `production` to disable hot-reload |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `HARD_EXAMPLES_DB` | `hard_examples.db` | SQLite database path |
| `HARD_EXAMPLES_IMAGE_DIR` | `hard_examples/images` | Saved image directory |

### Frontend

```bash
# Build (bakes API URL at build time)
cd frontend
docker build --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 -t car-damage-frontend .

# Run
docker run -p 3000:3000 car-damage-frontend
```

### Railway

Deploy as two services in one Railway project:

1. **Backend** — root dir `/`, Dockerfile at root. Set `ENV=production`, `CORS_ORIGINS=<frontend-url>`.
2. **Frontend** — root dir `/frontend`, Dockerfile at `frontend/Dockerfile`. Set build arg `NEXT_PUBLIC_API_URL=<backend-url>`.

Deploy backend first to get its URL, then deploy frontend with that URL, then update `CORS_ORIGINS` on the backend.

---

## Project Structure

```
car-damage-predictor/
├── api_server.py                        # FastAPI server (prediction, registry, mining)
├── hard_example_miner.py                # SQLite-backed uncertain prediction logger
├── train_curriculum.py                  # Multi-stage curriculum learning training
├── evaluate_model.py                    # Model evaluation and comparison
├── model_registry.json                  # Versioned model registry with ensemble definitions
├── class_mapping.json                   # Class index → label mapping
├── car_damage_model.keras               # Base EfficientNet-B0 model (v1)
├── models_curriculum/
│   ├── final_curriculum_model.keras     # Curriculum-trained model (v2-curriculum)
│   └── class_mapping.json
├── Dockerfile                           # Backend container (tensorflow-cpu)
├── .dockerignore
├── requirements.txt                     # Dev dependencies (tensorflow GPU)
├── requirements-deploy.txt              # Prod dependencies (tensorflow-cpu)
├── railway.toml                         # Railway health check config
├── examples/                            # Sample prediction images
├── frontend/
│   ├── app/                             # Next.js App Router pages
│   ├── components/                      # UI components (uploader, results, heatmap)
│   ├── lib/                             # API client and TypeScript types
│   ├── Dockerfile                       # Frontend container (multi-stage Node 20)
│   └── package.json
└── README.md
```

---

## Training

```bash
# Curriculum learning (recommended)
python train_curriculum.py --data-dir path/to/training_data

# With a separate validation set
python train_curriculum.py --data-dir path/to/train --val-dir path/to/val

# Evaluate a model
python evaluate_model.py --model models_curriculum/final_curriculum_model.keras \
  --data-dir path/to/test_data

# Compare two models
python evaluate_model.py --model car_damage_model.keras --data-dir path/to/test \
  --compare models_curriculum/final_curriculum_model.keras
```

---

## Tech Stack

- **Model:** EfficientNet-B0 (TensorFlow/Keras)
- **API:** FastAPI + Uvicorn
- **Frontend:** Next.js 15 · TypeScript · Tailwind CSS
- **Deployment:** Docker · Railway
- **Training:** Kaggle (Tesla T4 GPU)
- **Dataset:** [prajwalbhamere/car-damage-severity-dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)

---

## Notes
This was built with the help of Claude Sonnet 4.6 to gain experience.

<div align="center">
<sub>Built by <a href="https://github.com/jdelpup0502">jdelpup0502</a></sub>
</div>
