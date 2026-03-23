import io
import json
import time
import base64
import os
import random
import numpy as np
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from fastapi.responses import JSONResponse
import uvicorn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image, ImageEnhance
import cv2

from hard_example_miner import HardExampleMiner

# Configuration
REGISTRY_PATH = "model_registry.json"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6
GRADCAM_LAYER = "top_conv"  # Last conv layer inside EfficientNet-B0

app = FastAPI(title="Car Damage Severity API")

# CORS — allow frontend origin(s) set via CORS_ORIGINS env var
_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hard example mining — logs uncertain/wrong predictions for targeted retraining
miner = HardExampleMiner(
    db_path=os.environ.get("HARD_EXAMPLES_DB", "hard_examples.db"),
    image_store_dir=os.environ.get("HARD_EXAMPLES_IMAGE_DIR", "hard_examples/images"),
    uncertainty_threshold=0.5,
    confidence_threshold=0.7,
)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Manages multiple model versions, ensembles, and the currently active version."""

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self._loaded: dict = {}   # version -> {"model", "eff_grad_model", "post_eff_layers", "class_names"}
        self._meta: dict = {}
        self.active_version: str = None

    # -- Persistence ---------------------------------------------------------

    def _read(self) -> dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                return json.load(f)
        return {"active": None, "versions": {}, "ensembles": {}}

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

    def _ensure_loaded(self, version: str):
        """Load a version into memory if not already loaded."""
        if version not in self._loaded:
            self._meta = self._read()
            if version not in self._meta["versions"]:
                raise KeyError(f"Unknown version: {version}")
            print(f"Loading model version '{version}'...")
            self._loaded[version] = self._load_model_artifacts(version)
            print(f"Version '{version}' loaded. Classes: {self._loaded[version]['class_names']}")

    # -- Version management --------------------------------------------------

    def startup(self):
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
        self._meta = self._read()
        if version not in self._meta["versions"]:
            raise KeyError(f"Unknown version: {version}")
        self._ensure_loaded(version)
        self.active_version = version
        self._meta["active"] = version
        self._write(self._meta)

    def get_active(self) -> dict:
        return self._loaded[self.active_version]

    def get_version(self, version: str) -> dict:
        self._ensure_loaded(version)
        return self._loaded[version]

    def list_versions(self) -> list:
        self._meta = self._read()
        return [
            {
                "version": v,
                "active": v == self.active_version,
                "loaded": v in self._loaded,
                "path": info["path"],
                "description": info.get("description", ""),
                "registered_at": info.get("registered_at", ""),
            }
            for v, info in self._meta["versions"].items()
        ]

    # -- Ensemble management -------------------------------------------------

    def register_ensemble(self, name: str, versions: list, weights: list = None, description: str = ""):
        """Define a named ensemble. Weights are normalised to sum to 1."""
        self._meta = self._read()
        for v in versions:
            if v not in self._meta["versions"]:
                raise KeyError(f"Unknown version '{v}' — register it first.")
        if weights is None:
            weights = [1.0 / len(versions)] * len(versions)
        if len(weights) != len(versions):
            raise ValueError("len(weights) must equal len(versions)")
        total = sum(weights)
        weights = [w / total for w in weights]  # normalise
        self._meta.setdefault("ensembles", {})[name] = {
            "versions": versions,
            "weights": weights,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write(self._meta)

    def delete_ensemble(self, name: str):
        self._meta = self._read()
        if name not in self._meta.get("ensembles", {}):
            raise KeyError(f"Unknown ensemble: {name}")
        del self._meta["ensembles"][name]
        self._write(self._meta)

    def list_ensembles(self) -> list:
        self._meta = self._read()
        return [
            {"name": n, **info}
            for n, info in self._meta.get("ensembles", {}).items()
        ]

    def get_ensemble_members(self, name: str) -> list:
        """Return list of (artifacts, weight) for a named ensemble, loading models as needed."""
        self._meta = self._read()
        ensembles = self._meta.get("ensembles", {})
        if name not in ensembles:
            raise KeyError(f"Unknown ensemble: {name}")
        info = ensembles[name]
        members = []
        for version, weight in zip(info["versions"], info["weights"]):
            self._ensure_loaded(version)
            members.append((self._loaded[version], weight))
        return members

    # -- A/B experiment management -------------------------------------------

    def get_active_experiment(self) -> dict | None:
        """Return the active experiment dict (with its name), or None."""
        self._meta = self._read()
        for name, exp in self._meta.get("experiments", {}).items():
            if exp.get("status") == "active":
                return {"name": name, **exp}
        return None

    def create_experiment(self, name: str, variants: list, description: str = "") -> dict:
        """
        Create and activate a new A/B experiment.

        variants: list of {"version": str, "weight": float}
        Raises ValueError if another experiment is already active or a version is unknown.
        """
        self._meta = self._read()
        for v in variants:
            if v["version"] not in self._meta["versions"]:
                raise KeyError(f"Unknown version: {v['version']}")
        if self.get_active_experiment():
            raise ValueError("Another experiment is already active. Stop it first.")

        total = sum(v["weight"] for v in variants)
        normalized = [{"version": v["version"], "weight": v["weight"] / total} for v in variants]

        exp = {
            "variants": normalized,
            "status": "active",
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stopped_at": None,
        }
        self._meta.setdefault("experiments", {})[name] = exp
        self._write(self._meta)
        return {"name": name, **exp}

    def stop_experiment(self, name: str) -> dict:
        """Stop a running experiment."""
        self._meta = self._read()
        experiments = self._meta.get("experiments", {})
        if name not in experiments:
            raise KeyError(f"Unknown experiment: {name}")
        experiments[name]["status"] = "stopped"
        experiments[name]["stopped_at"] = datetime.now(timezone.utc).isoformat()
        self._write(self._meta)
        return {"name": name, **experiments[name]}

    def resolve_version_for_request(
        self, explicit_version: str | None
    ) -> tuple:
        """
        Decide which model version to use for this request.

        Returns (version, experiment_name, experiment_variant):
        - If explicit_version is given → bypass A/B, return (explicit_version, None, None)
        - If no active experiment → (active_version, None, None)
        - Otherwise → weighted random selection from the active experiment
        """
        if explicit_version:
            return explicit_version, None, None

        exp = self.get_active_experiment()
        if not exp:
            return self.active_version, None, None

        versions = [v["version"] for v in exp["variants"]]
        weights = [v["weight"] for v in exp["variants"]]
        selected = random.choices(versions, weights=weights, k=1)[0]
        return selected, exp["name"], selected


registry = ModelRegistry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    data = registry._read()
    if not data["versions"]:
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


def tta_predict(image_bytes: bytes, artifacts: dict, n_augments: int = 8) -> np.ndarray:
    """Run inference on multiple augmented versions and return averaged probabilities.

    Augmentations: original, horizontal flip, brightness ±20%, rotations ±10°
    (and flipped variants of each). Returns mean probabilities across all variants.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

    variants = [img, flipped]
    for factor in [0.8, 1.2]:
        variants.append(ImageEnhance.Brightness(img).enhance(factor))
        variants.append(ImageEnhance.Brightness(flipped).enhance(factor))
    for angle in [10, -10]:
        variants.append(img.rotate(angle))
        variants.append(flipped.rotate(angle))

    variants = variants[:n_augments]
    batch = np.stack([np.array(v, dtype=np.float32) for v in variants], axis=0)
    batch = preprocess_input(batch)

    all_probs = artifacts["model"].predict(batch, verbose=0)
    return np.mean(all_probs, axis=0)


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
    """Normalised predictive entropy [0, 1]. 0 = certain, 1 = maximally uncertain."""
    num_classes = len(probabilities)
    p = np.clip(probabilities, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(num_classes))


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


def ensemble_predict(img_array: np.ndarray, members: list) -> tuple:
    """Run inference across ensemble members and return weighted-average probabilities.

    Args:
        img_array: preprocessed image array (1, H, W, C)
        members: list of (artifacts, weight) tuples

    Returns:
        avg_probs       — (num_classes,) weighted average probabilities
        per_model       — list of {"version": ..., "weight": ..., "probabilities": {...}}
    """
    class_names = members[0][0]["class_names"]
    weighted_sum = np.zeros(len(class_names))
    per_model = []

    for artifacts, weight in members:
        probs = artifacts["model"].predict(img_array, verbose=0)[0]
        weighted_sum += weight * probs
        per_model.append({
            "weight": round(weight, 4),
            "probabilities": {
                class_names[i]: round(float(probs[i]), 4)
                for i in range(len(class_names))
            },
        })

    return weighted_sum, per_model


# ---------------------------------------------------------------------------
# Model management endpoints
# ---------------------------------------------------------------------------

@app.get("/models")
async def list_models():
    """List all registered model versions and ensembles."""
    return {
        "versions": registry.list_versions(),
        "ensembles": registry.list_ensembles(),
    }


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


@app.post("/models/ensembles/{name}")
async def create_ensemble(
    name: str,
    versions: List[str] = Body(..., description="List of version names to include"),
    weights: Optional[List[float]] = Body(None, description="Per-version weights (normalised automatically). Defaults to equal weights."),
    description: str = Body("", description="Optional description"),
):
    """Create or update a named ensemble."""
    try:
        registry.register_ensemble(name, versions, weights, description)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ensemble": name, "ensembles": registry.list_ensembles()}


@app.delete("/models/ensembles/{name}")
async def delete_ensemble(name: str):
    """Remove a named ensemble (does not unload or delete model files)."""
    try:
        registry.delete_ensemble(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": name, "ensembles": registry.list_ensembles()}


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    heatmap: bool = Query(False, description="Include Grad-CAM heatmap overlay in response"),
    uncertainty: bool = Query(False, description="Include predictive entropy as uncertainty score"),
    version: str = Query(None, description="Model version to use (defaults to active)"),
    tta: bool = Query(False, description="Use test-time augmentation (8 augmented views averaged)"),
):
    version_str, exp_name, exp_variant = registry.resolve_version_for_request(version)
    try:
        artifacts = registry.get_version(version_str)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown version: {version_str}")

    image_bytes = await file.read()

    start_time = time.time()
    if tta:
        probabilities = tta_predict(image_bytes, artifacts)
    else:
        img_array = preprocess_image(image_bytes)
        probabilities = artifacts["model"].predict(img_array, verbose=0)[0]
    inference_time = (time.time() - start_time) * 1000

    result = build_result(probabilities, artifacts["class_names"], CONFIDENCE_THRESHOLD)
    entropy = round(predictive_entropy(probabilities), 4)

    miner.log_prediction(
        image_bytes=image_bytes,
        filename=file.filename,
        model_version=version_str,
        probabilities=probabilities,
        class_names=artifacts["class_names"],
        source="api",
        experiment_name=exp_name,
        experiment_variant=exp_variant,
    )

    response = {
        "model_version": version_str,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "inference_time_ms": round(inference_time, 2),
    }
    if uncertainty:
        response["uncertainty"] = entropy
    if heatmap:
        plain_array = preprocess_image(image_bytes)
        cam = compute_gradcam(plain_array, result["predicted_idx"], artifacts)
        response["heatmap_png_base64"] = overlay_heatmap(image_bytes, cam)
    if tta:
        response["tta"] = True
    if exp_name:
        response["experiment"] = exp_name
        response["experiment_variant"] = exp_variant

    return response


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    heatmap: bool = Query(False, description="Include Grad-CAM heatmap overlay for each image"),
    uncertainty: bool = Query(False, description="Include predictive entropy as uncertainty score"),
    version: str = Query(None, description="Model version to use (defaults to active)"),
):
    version_str, exp_name, exp_variant = registry.resolve_version_for_request(version)
    try:
        artifacts = registry.get_version(version_str)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown version: {version_str}")

    images_bytes = [await f.read() for f in files]
    batch = np.concatenate([preprocess_image(b) for b in images_bytes], axis=0)

    start_time = time.time()
    all_probs = list(artifacts["model"].predict(batch, verbose=0))
    total_inference_time = (time.time() - start_time) * 1000

    results = []
    for i, probabilities in enumerate(all_probs):
        result = build_result(probabilities, artifacts["class_names"], CONFIDENCE_THRESHOLD)
        entropy = round(predictive_entropy(probabilities), 4)

        miner.log_prediction(
            image_bytes=images_bytes[i],
            filename=files[i].filename,
            model_version=version_str,
            probabilities=probabilities,
            class_names=artifacts["class_names"],
            source="api_batch",
            experiment_name=exp_name,
            experiment_variant=exp_variant,
        )

        item = {
            "filename": files[i].filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "all_probabilities": result["all_probabilities"],
        }
        if uncertainty:
            item["uncertainty"] = entropy
        if heatmap:
            img_array = np.expand_dims(batch[i], axis=0)
            cam = compute_gradcam(img_array, result["predicted_idx"], artifacts)
            item["heatmap_png_base64"] = overlay_heatmap(images_bytes[i], cam)
        results.append(item)

    response = {
        "model_version": version_str,
        "results": results,
        "total_images": len(files),
        "total_inference_time_ms": round(total_inference_time, 2),
        "avg_inference_time_ms": round(total_inference_time / len(files), 2),
    }
    if exp_name:
        response["experiment"] = exp_name
        response["experiment_variant"] = exp_variant
    return response


@app.post("/predict/ensemble")
async def predict_ensemble(
    file: UploadFile = File(...),
    ensemble: str = Query(None, description="Named ensemble to use"),
    versions: str = Query(None, description="Ad-hoc comma-separated versions, e.g. 'v1,v2'"),
    weights: str = Query(None, description="Ad-hoc comma-separated weights matching versions, e.g. '0.6,0.4'"),
    uncertainty: bool = Query(False, description="Include predictive entropy of ensemble output"),
    heatmap: bool = Query(False, description="Include Grad-CAM from the first ensemble member"),
):
    """Predict using a named ensemble or an ad-hoc list of versions."""
    if ensemble and versions:
        raise HTTPException(status_code=400, detail="Specify either 'ensemble' or 'versions', not both.")
    if not ensemble and not versions:
        raise HTTPException(status_code=400, detail="Specify 'ensemble' (named) or 'versions' (ad-hoc).")

    try:
        if ensemble:
            members = registry.get_ensemble_members(ensemble)
            ensemble_label = ensemble
        else:
            version_list = [v.strip() for v in versions.split(",")]
            weight_list = (
                [float(w.strip()) for w in weights.split(",")]
                if weights else None
            )
            if weight_list and len(weight_list) != len(version_list):
                raise HTTPException(status_code=400, detail="len(weights) must equal len(versions)")
            if weight_list is None:
                weight_list = [1.0 / len(version_list)] * len(version_list)
            total = sum(weight_list)
            weight_list = [w / total for w in weight_list]
            members = [(registry.get_version(v), w) for v, w in zip(version_list, weight_list)]
            ensemble_label = f"ad-hoc:[{versions}]"
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    start_time = time.time()
    avg_probs, per_model = ensemble_predict(img_array, members)
    inference_time = (time.time() - start_time) * 1000

    # Attach version names to per-model breakdown
    version_names = (
        registry._meta.get("ensembles", {}).get(ensemble, {}).get("versions", [])
        if ensemble
        else [v.strip() for v in versions.split(",")]
    )
    for i, entry in enumerate(per_model):
        entry["version"] = version_names[i] if i < len(version_names) else f"model_{i}"

    class_names = members[0][0]["class_names"]
    result = build_result(avg_probs, class_names, CONFIDENCE_THRESHOLD)
    response = {
        "ensemble": ensemble_label,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "per_model": per_model,
        "inference_time_ms": round(inference_time, 2),
    }
    if uncertainty:
        response["uncertainty"] = round(predictive_entropy(avg_probs), 4)
    if heatmap:
        cam = compute_gradcam(img_array, result["predicted_idx"], members[0][0])
        response["heatmap_png_base64"] = overlay_heatmap(image_bytes, cam)

    return response


# ---------------------------------------------------------------------------
# Hard example mining endpoints
# ---------------------------------------------------------------------------

@app.get("/hard-examples/stats")
async def hard_example_stats():
    """Summary statistics for logged hard examples."""
    return miner.get_stats()


@app.get("/hard-examples")
async def list_hard_examples(
    min_uncertainty: float = Query(None, description="Minimum uncertainty score [0, 1]"),
    max_confidence: float = Query(None, description="Maximum confidence score [0, 1]"),
    only_wrong: bool = Query(False, description="Only return misclassified examples"),
    class_filter: str = Query(None, description="Filter by predicted or true label, e.g. '02-moderate'"),
    limit: int = Query(100, description="Maximum number of results"),
):
    """List hard examples, sorted by difficulty score descending."""
    return miner.get_hard_examples(
        min_uncertainty=min_uncertainty,
        max_confidence=max_confidence,
        only_wrong=only_wrong,
        class_filter=class_filter,
        limit=limit,
    )


@app.post("/hard-examples/{example_id}/label")
async def label_hard_example(
    example_id: str,
    true_label: str = Body(..., description="Ground-truth class label, e.g. '02-moderate'"),
):
    """Add or update the ground-truth label for a logged example."""
    if not miner.label_example(example_id, true_label):
        raise HTTPException(status_code=404, detail=f"Example not found: {example_id}")
    return {"labeled": example_id, "true_label": true_label}


@app.post("/hard-examples/export")
async def export_hard_examples(
    output_dir: str = Query("hard_examples/dataset", description="Root directory for exported dataset"),
    strategy: str = Query(
        "uncertainty",
        description="Export strategy: 'uncertainty' | 'wrong' | 'hard' | 'all'",
    ),
    threshold: float = Query(None, description="Uncertainty threshold (defaults to miner setting)"),
    only_labeled: bool = Query(False, description="Only export examples with a confirmed true label"),
):
    """
    Export hard examples into a labelled directory structure for retraining.

    The output mirrors the layout expected by train_curriculum.py:
        output_dir/01-minor/...
        output_dir/02-moderate/...
        output_dir/03-severe/...
    """
    result = miner.export_dataset(output_dir, strategy, threshold, only_labeled)
    return result


# ---------------------------------------------------------------------------
# A/B Testing endpoints
# ---------------------------------------------------------------------------

@app.post("/experiments")
async def create_experiment(
    name: str = Body(..., description="Unique experiment name"),
    variants: list = Body(..., description="List of {version, weight} dicts"),
    description: str = Body("", description="Optional description"),
):
    """Create and activate a new A/B experiment."""
    try:
        result = registry.create_experiment(name, variants, description)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/experiments")
async def list_experiments():
    """List all experiments (active and stopped)."""
    registry._meta = registry._read()
    return [
        {"name": name, **exp}
        for name, exp in registry._meta.get("experiments", {}).items()
    ]


@app.post("/experiments/{name}/stop")
async def stop_experiment(name: str):
    """Stop an active experiment."""
    try:
        return registry.stop_experiment(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/experiments/{name}/metrics")
async def experiment_metrics(name: str):
    """Return per-variant metrics for an experiment."""
    registry._meta = registry._read()
    experiments = registry._meta.get("experiments", {})
    if name not in experiments:
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {name}")
    exp = experiments[name]
    variants_data = miner.get_experiment_metrics(name)
    return {
        "experiment": name,
        "status": exp["status"],
        "variants": variants_data,
        "total_requests": sum(v["request_count"] for v in variants_data.values()),
    }


@app.get("/health")
async def health():
    """Health check endpoint for Railway and other platforms."""
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    is_production = os.environ.get("ENV", "").lower() == "production"
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=not is_production)
