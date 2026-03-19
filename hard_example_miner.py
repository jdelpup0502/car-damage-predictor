"""
Hard Example Miner for car damage severity predictions.

Logs predictions where the model is uncertain or wrong, and exports
curated datasets of difficult examples for targeted retraining.

Mining strategies:
- uncertainty: entropy >= threshold (model is unsure, regardless of correctness)
- wrong: predicted label != true label (requires ground truth)
- hard: uncertain AND misclassified
- all: all logged hard examples
"""

import os
import json
import uuid
import sqlite3
import shutil
import numpy as np
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS hard_examples (
    id                   TEXT PRIMARY KEY,
    timestamp            TEXT NOT NULL,
    model_version        TEXT NOT NULL,
    source               TEXT NOT NULL,
    original_filename    TEXT,
    image_path           TEXT,
    true_label           TEXT,
    predicted_label      TEXT NOT NULL,
    confidence           REAL NOT NULL,
    uncertainty          REAL NOT NULL,
    difficulty_score     REAL NOT NULL,
    all_probabilities    TEXT NOT NULL,
    is_correct           INTEGER,
    is_hard              INTEGER NOT NULL DEFAULT 1,
    mined_for_retraining INTEGER NOT NULL DEFAULT 0
)
"""


class HardExampleMiner:
    """
    Logs model predictions, identifies difficult examples, and exports them
    for targeted retraining.
    """

    def __init__(
        self,
        db_path: str = "hard_examples.db",
        image_store_dir: str = "hard_examples/images",
        uncertainty_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
    ):
        self.db_path = db_path
        self.image_store_dir = image_store_dir
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        os.makedirs(image_store_dir, exist_ok=True)
        self._init_db()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(SCHEMA)

    @staticmethod
    def _compute_entropy(probabilities) -> float:
        """Normalised predictive entropy [0, 1]. 0 = certain, 1 = maximally uncertain."""
        p = np.clip(np.array(probabilities, dtype=np.float64), 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)) / np.log(len(p)))

    @staticmethod
    def _difficulty_score(uncertainty: float, confidence: float) -> float:
        """Weighted combination of uncertainty and low-confidence. Range [0, 1]."""
        return round(0.6 * uncertainty + 0.4 * (1.0 - confidence), 4)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def log_prediction(
        self,
        image_bytes: bytes,
        filename: str,
        model_version: str,
        probabilities,
        class_names: list,
        true_label: str = None,
        source: str = "api",
    ) -> str:
        """
        Log a prediction. Always records metadata; saves image bytes to disk
        only when the example is hard (uncertain or wrong).

        Returns the unique example ID.
        """
        probabilities = list(probabilities)
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        uncertainty = self._compute_entropy(probabilities)
        difficulty = self._difficulty_score(uncertainty, confidence)

        is_hard = (
            uncertainty >= self.uncertainty_threshold
            or confidence < self.confidence_threshold
        )

        is_correct = None
        if true_label is not None:
            is_correct = int(predicted_label == true_label)
            if not is_correct:
                is_hard = True  # always flag wrong predictions

        example_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Save image file only for hard examples
        image_path = None
        if is_hard and image_bytes:
            ext = Path(filename).suffix if filename else ".jpg"
            image_path = os.path.join(self.image_store_dir, f"{example_id}{ext}")
            with open(image_path, "wb") as f:
                f.write(image_bytes)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO hard_examples
                    (id, timestamp, model_version, source, original_filename,
                     image_path, true_label, predicted_label, confidence, uncertainty,
                     difficulty_score, all_probabilities, is_correct, is_hard,
                     mined_for_retraining)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    example_id,
                    timestamp,
                    model_version,
                    source,
                    filename,
                    image_path,
                    true_label,
                    predicted_label,
                    round(confidence, 4),
                    round(uncertainty, 4),
                    difficulty,
                    json.dumps(
                        {class_names[i]: round(float(probabilities[i]), 4)
                         for i in range(len(class_names))}
                    ),
                    is_correct,
                    int(is_hard),
                ),
            )

        return example_id

    # -------------------------------------------------------------------------
    # Labelling
    # -------------------------------------------------------------------------

    def label_example(self, example_id: str, true_label: str) -> bool:
        """
        Add or update the ground-truth label for a logged example.
        Re-evaluates is_correct and promotes to hard if wrong.
        Returns True if the example was found, False otherwise.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT predicted_label FROM hard_examples WHERE id = ?",
                (example_id,),
            ).fetchone()
            if row is None:
                return False
            is_correct = int(row[0] == true_label)
            # Wrong predictions are always hard
            conn.execute(
                """
                UPDATE hard_examples
                SET true_label = ?, is_correct = ?, is_hard = MAX(is_hard, ?)
                WHERE id = ?
                """,
                (true_label, is_correct, int(not is_correct), example_id),
            )
        return True

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def get_hard_examples(
        self,
        min_uncertainty: float = None,
        max_confidence: float = None,
        only_wrong: bool = False,
        only_hard: bool = True,
        class_filter: str = None,
        limit: int = 100,
    ) -> list:
        """
        Return hard examples, most difficult first.

        Args:
            min_uncertainty: Only return examples with uncertainty >= this value.
            max_confidence:  Only return examples with confidence <= this value.
            only_wrong:      Only return misclassified examples (requires true_label).
            only_hard:       Only return examples flagged as hard (default True).
            class_filter:    Filter by predicted or true label (e.g. '02-moderate').
            limit:           Maximum number of results.
        """
        conditions = []
        params = []

        if only_hard:
            conditions.append("is_hard = 1")
        if min_uncertainty is not None:
            conditions.append("uncertainty >= ?")
            params.append(min_uncertainty)
        if max_confidence is not None:
            conditions.append("confidence <= ?")
            params.append(max_confidence)
        if only_wrong:
            conditions.append("is_correct = 0")
        if class_filter:
            conditions.append("(predicted_label = ? OR true_label = ?)")
            params.extend([class_filter, class_filter])

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT * FROM hard_examples {where} ORDER BY difficulty_score DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Return summary statistics about logged examples."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM hard_examples").fetchone()[0]
            hard = conn.execute(
                "SELECT COUNT(*) FROM hard_examples WHERE is_hard = 1"
            ).fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(*) FROM hard_examples WHERE true_label IS NOT NULL"
            ).fetchone()[0]
            wrong = conn.execute(
                "SELECT COUNT(*) FROM hard_examples WHERE is_correct = 0"
            ).fetchone()[0]
            mined = conn.execute(
                "SELECT COUNT(*) FROM hard_examples WHERE mined_for_retraining = 1"
            ).fetchone()[0]
            avg_unc = conn.execute(
                "SELECT AVG(uncertainty) FROM hard_examples WHERE is_hard = 1"
            ).fetchone()[0]
            per_class = conn.execute(
                """
                SELECT predicted_label, COUNT(*) as cnt
                FROM hard_examples WHERE is_hard = 1
                GROUP BY predicted_label
                """
            ).fetchall()

        return {
            "total_logged": total,
            "hard_examples": hard,
            "labeled": labeled,
            "wrong_predictions": wrong,
            "mined_for_retraining": mined,
            "avg_uncertainty": round(avg_unc or 0.0, 4),
            "per_predicted_class": {row[0]: row[1] for row in per_class},
        }

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_dataset(
        self,
        output_dir: str,
        strategy: str = "uncertainty",
        threshold: float = None,
        only_labeled: bool = False,
    ) -> dict:
        """
        Copy hard examples into a labelled directory structure suitable for
        retraining (mirrors the layout expected by train_curriculum.py).

            output_dir/
                01-minor/   <image files>
                02-moderate/
                03-severe/

        Label priority: true_label > predicted_label.

        Args:
            output_dir:   Root directory for the exported dataset.
            strategy:     One of 'uncertainty' | 'wrong' | 'hard' | 'all'.
            threshold:    Uncertainty threshold (defaults to self.uncertainty_threshold).
            only_labeled: If True, only export examples with a true_label.

        Returns dict with export statistics.
        """
        if threshold is None:
            threshold = self.uncertainty_threshold

        conditions = ["image_path IS NOT NULL"]
        params = []

        if strategy == "uncertainty":
            conditions.append("uncertainty >= ?")
            params.append(threshold)
        elif strategy == "wrong":
            conditions.append("is_correct = 0")
            conditions.append("true_label IS NOT NULL")
        elif strategy == "hard":
            conditions.append("is_correct = 0")
            conditions.append("uncertainty >= ?")
            conditions.append("true_label IS NOT NULL")
            params.append(threshold)
        # 'all' — no additional filter

        if only_labeled:
            conditions.append("true_label IS NOT NULL")

        where = "WHERE " + " AND ".join(conditions)
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM hard_examples {where}", params
            ).fetchall()

        os.makedirs(output_dir, exist_ok=True)
        exported, skipped = 0, 0
        per_class: dict = {}
        exported_ids = []

        for row in rows:
            row = dict(row)
            label = row["true_label"] or row["predicted_label"]
            src = row["image_path"]

            if not label or not src or not os.path.exists(src):
                skipped += 1
                continue

            class_dir = os.path.join(output_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(class_dir, Path(src).name))
            exported += 1
            per_class[label] = per_class.get(label, 0) + 1
            exported_ids.append(row["id"])

        if exported_ids:
            with self._conn() as conn:
                conn.executemany(
                    "UPDATE hard_examples SET mined_for_retraining = 1 WHERE id = ?",
                    [(eid,) for eid in exported_ids],
                )

        return {
            "output_dir": output_dir,
            "strategy": strategy,
            "threshold": threshold,
            "exported": exported,
            "skipped": skipped,
            "per_class": per_class,
        }
