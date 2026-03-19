import io
import json
import time
import base64
import os
import numpy as np
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from typing import List
from fastapi.responses import JSONResponse
import uvicorn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import cv2

# Configuration
REGISTRY_PATH = "model_registry.json"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6
GRADCAM_LAYER = "top_conv"  # Last conv layer inside EfficientNet-B0

app = FastAPI(title="Car Damage Severity API")


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Manages multiple model versions and the currently active one."""

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self._loaded: dict = {}      # version -> {"model", "eff_grad_model", "post_eff_layers", "class_names"}
        self._meta: dict = {}        # persisted registry JSON
        self.active_version: str = None

    # -- Persistence ---------------------------------------------------------

    def _read(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                return json.load(f)
        return {"active": None, "versions": {}}

    def _write(self, data: dict):
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    # -- Loading -------------------------------------------------------------

    def _load_model_artifacts(self, version: str) -> dict:
        """Load a model version into memory and return its artifacts."""
        meta = self._meta["versions"][version]
        model = keras.models.load_model(meta["path"])

        efficientnet = model.get_layer("efficientnetb0")
        last_conv = efficientnet.get_layer(GRADCAM_LAYER)
        eff_grad_model = keras.Model(
            inputs=efficientnet.inputs,
            outputs=[last_conv.output, efficientnet.output]
        )
        post_eff_layers = model.layers[2:]

        with open(meta["class_mapping"]) as f:
            mapping = json.load(f)
            class_names = [mapping[str(i)] for i in range(len(mapping))]

        return {
            "model": model,
            "eff_grad_model": eff_grad_model,
            "post_eff_layers": post_eff_layers,
            "class_names": class_names,
        }

    # -- Public API ----------------------------------------------------------

    def startup(self):
        """Load registry and activate the stored active version."""
        self._meta = self._read()
        if not self._meta["versions"]:
            raise RuntimeError("model_registry.json has no registered versions.")
        active = self._meta.get("active")
        if not active or active not in self._meta["versions"]:
            active = next(iter(self._meta["versions"]))
            self._meta["active"] = active
            self._write(self._meta)
        self.activate(active)

    def register(self, version: str, path: str, class_mapping: str, description: str = ""):
        """Add a new version to the registry (does not activate it)."""
        self._meta = self._read()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        if not os.path.exists(class_mapping):
            raise FileNotFoundError(f"Class mapping not found: {class_mapping}")
        self._meta["versions"][version] = {
            "path": path,
            "class_mapping": class_mapping,
            "description": description,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        if not self._meta.get("active"):
            self._meta["active"] = version
        self._write(self._meta)

    def activate(self, version: str):
        """Switch the active version, loading it into memory if needed."""
        self._meta = self._read()
        if version not in self._meta["versions"]:
            raise KeyError(f"Unknown version: {version}")
        if version not in self._loaded:
            print(f"Loading model version '{version}'...")
            self._loaded[version] = self._load_model_artifacts(version)
            print(f"Version '{version}' loaded. Classes: {self._loaded[version]['class_names']}")
        self.active_version = version
        self._meta["active"] = version
        self._write(self._meta)

    def get_active(self) -> dict:
        """Return artifacts for the currently active version."""
        return self._loaded[self.active_version]

    def list_versions(self) -> list:
        self._meta = self._read()
        result = []
        for v, info in self._meta["versions"].items():
            result.append({
                "version": v,
                "active": v == self.active_version,
                "loaded": v in self._loaded,
                "path": info["path"],
                "description": info.get("description", ""),
                "registered_at": info.get("registered_at", ""),
            })
        return result


registry = ModelRegistry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Startup — seed registry from legacy MODEL_PATH if needed
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    data = registry._read()
    if not data["versions"]:
        # First run: register the existing model as v1
        registry._meta = data
        registry._write(data)
        registry.register(
            version="v1",
            path="car_damage_model.keras",
            class_mapping="class_mapping.json",
            description="Base EfficientNet-B0 model",
        )
    registry.startup()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def compute_gradcam(img_array, class_idx, artifacts: dict):
    eff_grad_model = artifacts["eff_grad_model"]
    post_eff_layers = artifacts["post_eff_layers"]

    with tf.GradientTape() as tape:
        conv_outputs, eff_out = eff_grad_model(img_array)
        tape.watch(conv_outputs)
        x = eff_out
        for layer in post_eff_layers:
            x = layer(x)
        loss = x[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap(image_bytes, heatmap, alpha=0.4):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img)

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * img_np + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predictive_entropy(probabilities: np.ndarray) -> float:
    """Normalised predictive entropy as uncertainty estimate.

    Returns a value in [0, 1]:
      0 = perfectly confident (one class has prob 1.0)
      1 = maximally uncertain (uniform distribution)
    """
    num_classes = len(probabilities)
    p = np.clip(probabilities, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(num_classes))


def build_result(probabilities, class_names, confidence_threshold):
    predicted_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_idx])
    return {
        "predicted_idx": predicted_idx,
        "prediction": "uncertain" if confidence < confidence_threshold else class_names[predicted_idx],
        "confidence": round(confidence, 4),
        "all_probabilities": {
            class_names[i]: round(float(probabilities[i]), 4)
            for i in range(len(class_names))
        },
    }


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------

@app.get("/models")
async def list_models():
    """List all registered model versions."""
    return {"versions": registry.list_versions()}


@app.post("/models/register")
async def register_model(
    version: str = Query(..., description="Version name, e.g. 'v2'"),
    path: str = Query(..., description="Path to the .keras model file"),
    class_mapping: str = Query("class_mapping.json", description="Path to class_mapping.json"),
    description: str = Query("", description="Optional description"),
):
    """Register a new model version. Does not activate it."""
    try:
        registry.register(version, path, class_mapping, description)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"registered": version, "versions": registry.list_versions()}


@app.post("/models/{version}/activate")
async def activate_model(version: str):
    """Switch the active model version (hot-swap, no restart needed)."""
    try:
        registry.activate(version)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"active": registry.active_version, "versions": registry.list_versions()}


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    heatmap: bool = Query(False, description="Include Grad-CAM heatmap overlay in response"),
    uncertainty: bool = Query(False, description="Run MC dropout to estimate prediction uncertainty"),
    version: str = Query(None, description="Model version to use (defaults to active)"),
):
    if version:
        try:
            artifacts = registry._loaded.get(version) or registry._load_model_artifacts(version)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown version: {version}")
    else:
        artifacts = registry.get_active()

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    start_time = time.time()
    probabilities = artifacts["model"].predict(img_array, verbose=0)[0]
    inference_time = (time.time() - start_time) * 1000

    result = build_result(probabilities, artifacts["class_names"], CONFIDENCE_THRESHOLD)
    response = {
        "model_version": version or registry.active_version,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "inference_time_ms": round(inference_time, 2),
    }
    if uncertainty:
        response["uncertainty"] = round(predictive_entropy(probabilities), 4)

    if heatmap:
        cam = compute_gradcam(img_array, result["predicted_idx"], artifacts)
        response["heatmap_png_base64"] = overlay_heatmap(image_bytes, cam)

    return response


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    heatmap: bool = Query(False, description="Include Grad-CAM heatmap overlay for each image"),
    uncertainty: bool = Query(False, description="Run MC dropout to estimate prediction uncertainty"),
    version: str = Query(None, description="Model version to use (defaults to active)"),
):
    if version:
        try:
            artifacts = registry._loaded.get(version) or registry._load_model_artifacts(version)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown version: {version}")
    else:
        artifacts = registry.get_active()

    images_bytes = [await f.read() for f in files]
    batch = np.concatenate([preprocess_image(b) for b in images_bytes], axis=0)

    start_time = time.time()
    all_probs = list(artifacts["model"].predict(batch, verbose=0))
    total_inference_time = (time.time() - start_time) * 1000

    results = []
    for i, probabilities in enumerate(all_probs):
        result = build_result(probabilities, artifacts["class_names"], CONFIDENCE_THRESHOLD)
        item = {
            "filename": files[i].filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
        }
        if uncertainty:
            item["uncertainty"] = round(predictive_entropy(probabilities), 4)
        if heatmap:
            img_array = np.expand_dims(batch[i], axis=0)
            cam = compute_gradcam(img_array, result["predicted_idx"], artifacts)
            item["heatmap_png_base64"] = overlay_heatmap(images_bytes[i], cam)
        results.append(item)

    return {
        "model_version": version or registry.active_version,
        "results": results,
        "total_images": len(files),
        "total_inference_time_ms": round(total_inference_time, 2),
        "avg_inference_time_ms": round(total_inference_time / len(files), 2),
    }


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
