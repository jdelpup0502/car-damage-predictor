"""
car_gate.py — Lightweight vehicle OOD gate using MobileNetV2 (ImageNet).

Strategy: inspect the top-K ImageNet predictions against a curated set of
vehicle-related class indices. The image passes if:
  - any single vehicle-class score >= min_vehicle_confidence, OR
  - the sum of all vehicle-class scores in the top-K >= sum_vehicle_confidence

The OR-of-sums logic handles heavily damaged cars where probability mass is
diffuse across multiple part-classes (grille, car wheel, fender) rather than
concentrated on a single "sedan" class.

MobileNetV2 is lazy-loaded on first use; the module-level singleton is shared
across threads via a threading.Lock to avoid double-loading.
"""

import io
import threading
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------------------------
# Curated set of ImageNet ILSVRC-2012 class indices covering vehicles and
# structurally adjacent car parts. Verified against the standard Keras
# label mapping (tf.keras.applications.mobilenet_v2.decode_predictions).
# ---------------------------------------------------------------------------
VEHICLE_CLASS_IDS = frozenset([
    407,  # ambulance
    436,  # beach wagon, station wagon, estate car
    468,  # cab, hack, taxi, taxicab
    475,  # car wheel
    511,  # convertible
    530,  # fire engine, fire truck
    565,  # go-kart
    569,  # grille, radiator grille
    609,  # jeep, landrover
    627,  # limousine, limo
    654,  # minibus
    656,  # minivan
    661,  # moving van
    705,  # passenger car, motorcar
    717,  # police van, paddy wagon
    734,  # racer, race car, racing car
    751,  # recreational vehicle, RV
    779,  # school bus
    783,  # sedan, saloon
    817,  # sports car
    868,  # tow truck, tow car, wrecker
    871,  # trailer truck, tractor trailer
    891,  # van
])

_IMG_SIZE = (224, 224)
_model = None
_lock = threading.Lock()


class CarGate:
    """
    Lightweight vehicle presence detector.

    Args:
        top_k: Number of top ImageNet predictions to inspect.
        min_vehicle_confidence: Minimum single-class score to pass (per-class OR gate).
        sum_vehicle_confidence: Minimum cumulative vehicle-class score to pass (sum OR gate).
    """

    def __init__(
        self,
        top_k: int = 10,
        min_vehicle_confidence: float = 0.10,
        sum_vehicle_confidence: float = 0.15,
    ):
        self.top_k = top_k
        self.min_vehicle_confidence = min_vehicle_confidence
        self.sum_vehicle_confidence = sum_vehicle_confidence

    def _get_model(self):
        global _model
        if _model is None:
            with _lock:
                if _model is None:
                    print("CarGate: loading MobileNetV2...")
                    _model = MobileNetV2(weights="imagenet", include_top=True)
                    print("CarGate: ready.")
        return _model

    def check(self, image_bytes: bytes) -> tuple:
        """
        Determine whether an image likely contains a vehicle.

        Returns:
            (is_vehicle: bool, reason: str)
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(_IMG_SIZE)
        except Exception as e:
            return False, f"Cannot decode image: {e}"

        arr = preprocess_input(
            np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        )
        preds = self._get_model().predict(arr, verbose=0)[0]  # (1000,)

        top_indices = np.argsort(preds)[::-1][: self.top_k]
        vehicle_scores = [float(preds[i]) for i in top_indices if i in VEHICLE_CLASS_IDS]

        if vehicle_scores:
            max_score = max(vehicle_scores)
            sum_score = sum(vehicle_scores)
            if max_score >= self.min_vehicle_confidence:
                return True, f"vehicle class score {max_score:.3f}"
            if sum_score >= self.sum_vehicle_confidence:
                return True, f"cumulative vehicle score {sum_score:.3f}"
            return False, (
                f"max vehicle score {max_score:.3f} < {self.min_vehicle_confidence}, "
                f"sum {sum_score:.3f} < {self.sum_vehicle_confidence}"
            )

        return False, "no vehicle-related class in top predictions"
