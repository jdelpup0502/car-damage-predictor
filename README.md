<div align="center">

# 🚗 Car Damage Severity Predictor

**Deep learning model that classifies car damage as minor, moderate, or severe from a single photo.**

Built with EfficientNet-B0 transfer learning · Served via FastAPI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## How It Works

Upload a photo of a damaged car → get back a severity classification with confidence scores.

```
POST /predict  →  { "prediction": "03-severe", "confidence": 0.92 }
```

The model uses **EfficientNet-B0** pretrained on ImageNet, fine-tuned on ~1,400 labeled car damage images using a two-phase transfer learning strategy.

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
Confidence: <code>70.3%</code>
<br>
<sub>minor: 2.5% · moderate: 27.2% · severe: 70.3%</sub>
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

### Standard Training Strategy

1. **Phase 1** — Freeze backbone, train classification head (lr=1e-3, 10 epochs)
2. **Phase 2** — Unfreeze top 50 backbone layers, fine-tune (lr=5e-5, 30 epochs with early stopping)

Class weights boosted moderate by 1.5× to combat its lower recall.

### Curriculum Learning Training (Experimental)

For improved moderate class accuracy, use multi-stage curriculum learning:

```bash
python train_curriculum.py --data-dir path/to/training_data
```

**Curriculum strategy:**
1. **Stage 1** — Binary: Minor vs Non-minor (learns clear patterns first)
2. **Stage 2** — Binary: Severe vs Non-severe (learns clear patterns first)
3. **Stage 3** — 3-class fine-tuning with gradual introduction of ambiguous moderate samples

The `train_curriculum.py` script implements:
- **Warmup epochs**: Train on clear minor/severe samples only
- **Gradual curriculum**: Incrementally add moderate samples (20% → 100%)
- **Label smoothing**: Soft labels for ambiguous moderate samples
- **Class weighting**: 2× boost for moderate class

Results are saved to `models_curriculum/` directory.

The kaggle notebook including how the model was built can be found [here.](https://www.kaggle.com/code/justindelpup/car-damage-pre)

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

Server runs at `http://localhost:8000`

### 3. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@your_car_image.jpg"
```

Or open `http://localhost:8000/docs` for the interactive Swagger UI.

### Example response

```json
{
  "prediction": "03-severe",
  "confidence": 0.9187,
  "all_probabilities": {
    "01-minor": 0.0005,
    "02-moderate": 0.0807,
    "03-severe": 0.9187
  },
  "inference_time_ms": 248.43
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload an image, get damage severity prediction |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

## Training

### Standard Training

```bash
python api_server.py  # Uses pretrained model by default
```

### Curriculum Learning Training

```bash
python train_curriculum.py --data-dir path/to/training_data
```

Train using multi-stage curriculum learning to improve moderate class accuracy.

---

## Project Structure

## Project Structure

```
car-damage-predictor/
├── api_server.py              # FastAPI server
├── car_damage_model.keras     # Trained model weights
├── class_mapping.json         # Class index → label mapping
├── requirements.txt           # Python dependencies
├── train_curriculum.py        # Curriculum learning training script
├── evaluate_model.py          # Model evaluation script
├── examples/                  # Sample prediction images
│   ├── minor_example.jpg
│   ├── moderate_example.jpg
│   └── severe_example_2.jpg
└── README.md
```

---

## Future Improvements

- [ ] Add more training data (label images from other Kaggle datasets)
- [ ] Build a drag-and-drop web frontend
- [ ] Dockerize for cloud deployment
- [ ] Add damage localization with Grad-CAM heatmaps

---

## Tech Stack

- **Model:** EfficientNet-B0 (TensorFlow/Keras)
- **API:** FastAPI + Uvicorn
- **Training:** Kaggle (Tesla T4 GPU)
- **Dataset:** [prajwalbhamere/car-damage-severity-dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)

---

## Notes
This was built with the help of Claude Opus 4.6 and Qwen-3-coder to gain experience.

<div align="center">
<sub>Built by <a href="https://github.com/jdelpup0502">jdelpup0502</a></sub>
</div>
