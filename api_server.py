import io
import json
import time
import base64
import numpy as np

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import uvicorn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import cv2

# Configuration
MODEL_PATH = "car_damage_model.keras"
CLASS_MAPPING_PATH = "class_mapping.json"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.6
GRADCAM_LAYER = "top_conv"  # Last conv layer inside EfficientNet-B0

app = FastAPI(title="Car Damage Severity API")

# Global variables — loaded once at startup
model = None
eff_grad_model = None   # EfficientNet input → (last_conv_output, eff_output)
post_eff_layers = None  # Layers after EfficientNet (GAP, Dense, Dropout, ...)
class_names = None


@app.on_event("startup")
async def load_model():
    global model, eff_grad_model, post_eff_layers, class_names

    model = keras.models.load_model(MODEL_PATH)

    # Grad-CAM requires tracing through the nested EfficientNet submodel.
    # Build a sub-model at the EfficientNet level, then track remaining layers separately.
    efficientnet = model.get_layer("efficientnetb0")
    last_conv = efficientnet.get_layer(GRADCAM_LAYER)
    eff_grad_model = keras.Model(
        inputs=efficientnet.inputs,
        outputs=[last_conv.output, efficientnet.output]
    )
    # Layers in the outer model that come after EfficientNet (index 0=input, 1=efficientnet, 2+=rest)
    post_eff_layers = model.layers[2:]

    with open(CLASS_MAPPING_PATH) as f:
        mapping = json.load(f)
        class_names = [mapping[str(i)] for i in range(len(mapping))]

    print(f"Model loaded. Classes: {class_names}")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def compute_gradcam(img_array, class_idx):
    """Returns a Grad-CAM heatmap as a numpy array (H, W) in [0, 1]."""
    with tf.GradientTape() as tape:
        conv_outputs, eff_out = eff_grad_model(img_array)
        tape.watch(conv_outputs)
        # Pass through remaining outer-model layers to get final predictions
        x = eff_out
        for layer in post_eff_layers:
            x = layer(x)
        loss = x[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)                # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)
    conv_out = conv_outputs[0]                               # (H, W, C)

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]      # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                           # (H, W)
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0, 1]
    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_heatmap(image_bytes, heatmap, alpha=0.4):
    """Blends Grad-CAM heatmap onto the original image, returns base64 PNG."""
    # Original image resized to model input size
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img)

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (1 - alpha) * img_np + alpha * heatmap_color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Encode to base64 PNG
    pil_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    heatmap: bool = Query(False, description="Include Grad-CAM heatmap overlay in response")
):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    inference_time = (time.time() - start_time) * 1000

    probabilities = predictions[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    all_probs = {
        class_names[i]: round(float(probabilities[i]), 4)
        for i in range(len(class_names))
    }

    uncertain = confidence < CONFIDENCE_THRESHOLD

    response = {
        "prediction": "uncertain" if uncertain else predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "inference_time_ms": round(inference_time, 2)
    }

    if heatmap:
        cam = compute_gradcam(img_array, predicted_idx)
        response["heatmap_png_base64"] = overlay_heatmap(image_bytes, cam)

    return response


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
