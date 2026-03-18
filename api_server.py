import os
import io
import json
import time
import numpy as np

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Configuration
MODEL_PATH = "car_damage_model.keras"
CLASS_MAPPING_PATH = "class_mapping.json"
IMG_SIZE = 224

app = FastAPI(title="Car Damage Severity API")

# Global variables — loaded once at startup
model = None
class_names = None


@app.on_event("startup")
async def load_model():
    global model, class_names

    model = keras.models.load_model(MODEL_PATH)

    with open(CLASS_MAPPING_PATH) as f:
        mapping = json.load(f)
        class_names = [mapping[str(i)] for i in range(len(mapping))]

    print(f"Model loaded. Classes: {class_names}")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)

    # Apply same preprocessing as training
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
        "inference_time_ms": round(inference_time, 2)
    }

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

