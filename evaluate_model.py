"""
Evaluation script for car damage severity models.

Supports:
- Standard 3-class evaluation with confusion matrix
- Per-class metrics (precision, recall, F1)
- Performance comparison between models
- Sample prediction visualization
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import argparse
import logging
from pathlib import Path

from hard_example_miner import HardExampleMiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_from_directory(data_dir):
    """Load images and labels from directory structure."""
    images = []
    labels = []
    filenames = []

    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
    )
    logger.info(f"Found classes: {class_names}")

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32)
                images.append(img_array)
                labels.append(class_idx)
                filenames.append(img_path)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")

    return np.array(images), np.array(labels), filenames


def preprocess_input(img_array):
    """Apply EfficientNet preprocessing."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img_array.copy())


def evaluate_model(model_path, data_dir, output_dir=None, batch_size=32,
                   mine_hard_examples=False, hard_examples_db="hard_examples.db"):
    """Evaluate a trained model on test data."""

    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    # Load test data
    logger.info(f"Loading test data from: {data_dir}")
    X_test, y_test, filenames = load_dataset_from_directory(data_dir)

    # Preprocess
    logger.info("Preprocessing images...")
    X_test = preprocess_input(X_test)

    # Make predictions
    logger.info("Making predictions...")
    y_pred_proba = model.predict(X_test, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1, 2]
    )

    # Class names
    class_names = ['01-minor', '02-moderate', '03-severe']

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(y_test)}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("\nPer-Class Metrics:")
    logger.info("-" * 40)

    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name}:")
        logger.info(f"  Precision: {precision[i]:.4f}")
        logger.info(f"  Recall:    {recall[i]:.4f}")
        logger.info(f"  F1-Score:  {f1[i]:.4f}")
        logger.info(f"  Support:   {support[i]}")

    logger.info("-" * 40)
    logger.info(f"Macro Average:")
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', labels=[0, 1, 2]
    )
    logger.info(f"  Precision: {macro_precision:.4f}")
    logger.info(f"  Recall:    {macro_recall:.4f}")
    logger.info(f"  F1-Score:  {macro_f1:.4f}")

    logger.info(f"\nClassification Report:")
    logger.info("-" * 40)
    logger.info(classification_report(y_test, y_pred, target_names=class_names))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

        # Save metrics to JSON
        metrics = {
            'accuracy': float(accuracy),
            'per_class': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(3)
            },
            'macro_average': {
                'precision': float(macro_precision),
                'recall': float(macro_recall),
                'f1_score': float(macro_f1)
            }
        }

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"\nResults saved to: {output_dir}")

    if mine_hard_examples:
        model_version = Path(model_path).stem
        miner = HardExampleMiner(
            db_path=hard_examples_db,
            image_store_dir=str(Path(hard_examples_db).parent / "hard_examples" / "images"),
        )
        logger.info("\nMining hard examples from evaluation set...")
        mined_count = 0
        for i, filepath in enumerate(filenames):
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            true_label = class_names[int(y_test[i])]
            example_id = miner.log_prediction(
                image_bytes=image_bytes,
                filename=filepath,
                model_version=model_version,
                probabilities=y_pred_proba[i],
                class_names=class_names,
                true_label=true_label,
                source="evaluation",
            )
            mined_count += 1

        stats = miner.get_stats()
        logger.info(f"Logged {mined_count} samples. Hard examples: {stats['hard_examples']} "
                    f"| Wrong predictions: {stats['wrong_predictions']}")
        if output_dir:
            with open(os.path.join(output_dir, "hard_examples_stats.json"), "w") as f:
                json.dump(stats, f, indent=2)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def compare_models(model_paths, data_dir, output_dir=None):
    """Compare multiple models on the same test set."""

    results = {}
    class_names = ['01-minor', '02-moderate', '03-severe']

    for model_path in model_paths:
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            continue

        logger.info(f"\nEvaluating: {model_path}")
        result = evaluate_model(model_path, data_dir, output_dir=None, batch_size=32)
        results[os.path.basename(model_path)] = result

    # Create comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)

    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Accuracy: {result['accuracy']:.4f}")
        logger.info(f"  Moderate F1: {result['f1'][1]:.4f}")
        logger.info(f"  Severe F1: {result['f1'][2]:.4f}")

    # Bar chart comparison
    if len(results) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = list(results.keys())
        x = np.arange(len(models))
        width = 0.25

        for idx, class_name in enumerate(class_names):
            ax = axes[idx]
            accuracies = [results[m]['accuracy'] for m in models]
            f1_scores = [results[m]['f1'][idx] for m in models]

            ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
            ax.bar(x + width/2, f1_scores, width, label='F1', alpha=0.7)
            ax.set_title(f'{class_name}')
            ax.set_ylabel('Score')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)

        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)

        plt.show()


def predict_single_image(model_path, image_path, class_mapping_path='class_mapping.json'):
    """Make prediction on a single image."""

    model = keras.models.load_model(model_path)

    with open(class_mapping_path) as f:
        class_mapping = json.load(f)

    # Load and preprocess image
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))

    # Map class indices to names
    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]

    result = {
        'prediction': class_names[predicted_idx],
        'confidence': float(predictions[predicted_idx]),
        'all_probabilities': {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    }

    logger.info(f"\nPrediction for: {image_path}")
    logger.info(f"  Predicted: {result['prediction']}")
    logger.info(f"  Confidence: {result['confidence']:.4f}")
    logger.info(f"  Probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        logger.info(f"    {class_name}: {prob:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate car damage severity models')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model file (.keras)')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Additional model paths to compare')
    parser.add_argument('--predict', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--mine-hard-examples', action='store_true',
                       help='Log uncertain/wrong predictions to the hard examples database')
    parser.add_argument('--hard-examples-db', type=str, default='hard_examples.db',
                       help='Path to the hard examples SQLite database')

    args = parser.parse_args()

    if args.predict:
        # Single image prediction
        predict_single_image(args.model, args.predict)
    elif args.compare:
        # Compare multiple models
        model_paths = [args.model] + args.compare
        compare_models(model_paths, args.data_dir, args.output_dir)
    else:
        # Single model evaluation
        evaluate_model(
            args.model, args.data_dir, args.output_dir,
            mine_hard_examples=args.mine_hard_examples,
            hard_examples_db=args.hard_examples_db,
        )


if __name__ == '__main__':
    main()
