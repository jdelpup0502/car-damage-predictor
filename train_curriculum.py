"""
Multi-stage curriculum learning for car damage severity classification.

Strategy:
- Stage 1: Binary classification - Minor vs Non-minor (easier distinction)
- Stage 2: Binary classification - Severe vs Non-severe (easier distinction)
- Stage 3: Full 3-class fine-tuning with moderate class focus

This approach builds confidence on clear cases before tackling ambiguous moderate samples.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 15
EPOCHS_STAGE3 = 25
BASE_LEARNING_RATE = 5e-5

# Class indices
MINOR_IDX = 0
MODERATE_IDX = 1
SEVERE_IDX = 2


def load_dataset_from_directory(data_dir):
    """Load images and labels from directory structure."""
    images = []
    labels = []

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
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype=np.float32)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")

    return np.array(images), np.array(labels)


def preprocess_input(img_array):
    """Apply EfficientNet preprocessing."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    return preprocess_input(img_array.copy())


# Augmentation pipeline applied during training only
_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")


def augment_batch(X_batch):
    """Apply random augmentations to a batch (numpy array → numpy array)."""
    return _augmentation(tf.constant(X_batch), training=True).numpy()


def build_model(num_classes=3, freeze_base=True):
    """Build the EfficientNet-B0 model for severity classification."""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    if freeze_base:
        base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = layers.Dense(num_classes, activation=activation)(x)

    model = models.Model(inputs, outputs)
    return model


def create_binary_labels(labels, target_class, negative_classes):
    """Create binary labels for curriculum stages.

    Args:
        labels: Original multi-class labels
        target_class: The class to treat as positive (1)
        negative_classes: List of classes to treat as negative (0)

    Returns:
        Binary labels array
    """
    binary_labels = np.zeros(len(labels), dtype=np.float32)
    for neg_class in negative_classes:
        binary_labels[labels == neg_class] = 0
    binary_labels[labels == target_class] = 1
    return binary_labels


def compute_class_weights(labels, num_classes):
    """Compute balanced class weights."""
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=labels
    )
    return dict(enumerate(class_weights_array))


class CurriculumScheduler:
    """Scheduler for gradually introducing moderate samples in Stage 3."""

    def __init__(self, train_size, moderate_indices, initial_fraction=0.3,
                 final_fraction=1.0, warmup_epochs=5):
        self.train_size = train_size
        self.moderate_indices = moderate_indices
        self.initial_fraction = initial_fraction
        self.final_fraction = final_fraction
        self.warmup_epochs = warmup_epochs

    def get_samples_for_epoch(self, epoch):
        """Get indices of samples to include in this epoch."""
        if epoch < self.warmup_epochs:
            # Only include non-moderate samples during warmup
            return np.array([i for i in range(self.train_size)
                           if i not in self.moderate_indices])

        # Gradually increase moderate sample inclusion
        progress = (epoch - self.warmup_epochs) / (EPOCHS_STAGE3 - self.warmup_epochs)
        current_fraction = self.initial_fraction + progress * (self.final_fraction - self.initial_fraction)

        num_moderate_to_include = int(len(self.moderate_indices) * current_fraction)

        # Include all non-moderate samples plus subset of moderate
        non_moderate_indices = np.array([i for i in range(self.train_size)
                                        if i not in self.moderate_indices])

        if num_moderate_to_include > 0:
            moderate_subset = np.random.choice(
                self.moderate_indices,
                size=min(num_moderate_to_include, len(self.moderate_indices)),
                replace=False
            )
            return np.concatenate([non_moderate_indices, moderate_subset])
        else:
            return non_moderate_indices


def train_stage1(X_train, y_train, X_val, y_val):
    """Stage 1: Minor vs Non-minor binary classification."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Minor vs Non-minor Binary Classification")
    logger.info("=" * 60)

    # Create binary labels: Minor (0) = 1, Moderate (1) = 0, Severe (2) = 0
    y_train_binary = create_binary_labels(y_train, MINOR_IDX, [MODERATE_IDX, SEVERE_IDX])
    y_val_binary = create_binary_labels(y_val, MINOR_IDX, [MODERATE_IDX, SEVERE_IDX])

    model = build_model(num_classes=1, freeze_base=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=3,
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train_binary,
        validation_data=(X_val, y_val_binary),
        epochs=EPOCHS_STAGE1,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history


def train_stage2(X_train, y_train, X_val, y_val):
    """Stage 2: Severe vs Non-severe binary classification."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Severe vs Non-severe Binary Classification")
    logger.info("=" * 60)

    # Create binary labels: Minor (0) = 0, Moderate (1) = 0, Severe (2) = 1
    y_train_binary = create_binary_labels(y_train, SEVERE_IDX, [MINOR_IDX, MODERATE_IDX])
    y_val_binary = create_binary_labels(y_val, SEVERE_IDX, [MINOR_IDX, MODERATE_IDX])

    model = build_model(num_classes=1, freeze_base=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=3,
        mode='max',
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train_binary,
        validation_data=(X_val, y_val_binary),
        epochs=EPOCHS_STAGE2,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history


def train_stage3(X_train, y_train, X_val, y_val, use_curriculum=True):
    """Stage 3: Full 3-class fine-tuning with curriculum learning.

    Two-phase approach to prevent catastrophic forgetting:
    - Phase A: Freeze backbone, train head only (curriculum intro of moderate)
    - Phase B: Unfreeze top EfficientNet layers, fine-tune end-to-end
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Full 3-Class Fine-tuning with Curriculum Learning")
    logger.info("=" * 60)

    moderate_indices = np.where(y_train == MODERATE_IDX)[0]
    logger.info(f"Training set: {len(y_train)} samples")
    logger.info(f"  - Minor: {np.sum(y_train == MINOR_IDX)}")
    logger.info(f"  - Moderate: {len(moderate_indices)}")
    logger.info(f"  - Severe: {np.sum(y_train == SEVERE_IDX)}")

    class_weights = compute_class_weights(y_train, 3)
    class_weights[MODERATE_IDX] *= 2.0
    logger.info(f"Class weights: {class_weights}")

    validation_data = (X_val, keras.utils.to_categorical(y_val, num_classes=3))

    # ------------------------------------------------------------------ #
    # Phase A: Frozen backbone — train head only
    # ------------------------------------------------------------------ #
    logger.info("\n--- Phase A: Training head with frozen backbone ---")
    model = build_model(num_classes=3, freeze_base=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    phase_a_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, mode='max',
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        )
    ]

    if use_curriculum:
        scheduler = CurriculumScheduler(
            len(y_train), moderate_indices,
            initial_fraction=0.2, final_fraction=1.0, warmup_epochs=3
        )

        class CurriculumDataGenerator(keras.utils.Sequence):
            def __init__(self, X, y, indices_to_use, batch_size):
                self.X = X
                self.y = y
                self.indices_to_use = indices_to_use
                self.batch_size = batch_size

            def __len__(self):
                return int(np.ceil(len(self.indices_to_use) / self.batch_size))

            def __getitem__(self, idx):
                batch_indices = self.indices_to_use[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_indices = np.array(batch_indices)
                X_batch = augment_batch(self.X[batch_indices])
                y_batch = keras.utils.to_categorical(self.y[batch_indices], num_classes=3)
                return X_batch, y_batch

        logger.info("Phase A with curriculum scheduler (20% → 100% moderate over warmup)")
        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(EPOCHS_STAGE3):
            current_indices = scheduler.get_samples_for_epoch(epoch)
            train_generator = CurriculumDataGenerator(X_train, y_train, current_indices, BATCH_SIZE)
            n_moderate = len(np.intersect1d(current_indices, moderate_indices))
            logger.info(f"\n--- Curriculum Epoch {epoch + 1}/{EPOCHS_STAGE3} "
                        f"({len(current_indices)} samples, {n_moderate} moderate) ---")

            history = model.fit(
                train_generator,
                epochs=epoch + 1,
                initial_epoch=epoch,
                validation_data=validation_data,
                class_weight=class_weights,
                callbacks=phase_a_callbacks,
                verbose=1
            )

            val_acc = history.history['val_accuracy'][-1]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 5:
                    logger.info("Early stopping triggered in curriculum phase A")
                    break
    else:
        logger.info("Phase A without curriculum")
        model.fit(
            tf.data.Dataset.from_tensor_slices(
                (X_train, keras.utils.to_categorical(y_train, num_classes=3))
            ).shuffle(len(X_train)).batch(BATCH_SIZE)
             .map(lambda x, y: (_augmentation(x, training=True), y),
                  num_parallel_calls=tf.data.AUTOTUNE)
             .prefetch(tf.data.AUTOTUNE),
            validation_data=validation_data,
            epochs=EPOCHS_STAGE3,
            class_weight=class_weights,
            callbacks=phase_a_callbacks,
            verbose=1
        )

    logger.info(f"\nPhase A complete. Best val_accuracy: {best_val_acc:.4f}" if use_curriculum
                else "\nPhase A complete.")

    # ------------------------------------------------------------------ #
    # Phase B: Unfreeze top EfficientNet layers, fine-tune end-to-end
    # ------------------------------------------------------------------ #
    logger.info("\n--- Phase B: Fine-tuning top EfficientNet layers ---")
    efficientnet = model.get_layer("efficientnetb0")
    efficientnet.trainable = True
    # Freeze all but the last 50 layers (same as base model strategy)
    for layer in efficientnet.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    phase_b_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=7, mode='max',
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=3,
            min_lr=1e-7, verbose=1
        )
    ]

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (X_train, keras.utils.to_categorical(y_train, num_classes=3))
        )
        .shuffle(len(X_train))
        .batch(BATCH_SIZE)
        .map(lambda x, y: (_augmentation(x, training=True), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.fit(
        train_ds,
        validation_data=validation_data,
        epochs=30,
        class_weight=class_weights,
        callbacks=phase_b_callbacks,
        verbose=1
    )

    return model


def main(data_dir=None, output_dir='models_curriculum'):
    """Main training pipeline with multi-stage curriculum learning."""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-stage curriculum learning for car damage severity')
    parser.add_argument('--data-dir', type=str, help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default=None,
                       help='Path to separate validation directory (optional)')
    parser.add_argument('--output-dir', type=str, default='models_curriculum',
                       help='Directory to save models')
    parser.add_argument('--skip-stages', type=str, default='',
                       help='Comma-separated stage numbers to skip (e.g., "1,2")')
    parser.add_argument('--use-curriculum', action='store_false', default=True,
                       help='Disable curriculum learning in stage 3')


    args = parser.parse_args()

    if data_dir is None:
        data_dir = args.data_dir
    if not data_dir:
        # Try to find data in common locations
        possible_dirs = [
            'data',
            'training_data',
            'images',
            '../data',
            '/kaggle/input/car-damage-severity-dataset/training_data'
        ]
        for d in possible_dirs:
            if os.path.isdir(d):
                data_dir = d
                break

    if not data_dir:
        raise ValueError("No data directory specified. Use --data-dir or place data in common locations.")

    logger.info(f"Loading data from: {data_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    X_train, y_train = load_dataset_from_directory(data_dir)
    logger.info(f"Loaded {len(X_train)} training images with {len(np.unique(y_train))} classes")

    if args.val_dir:
        logger.info(f"Loading validation data from: {args.val_dir}")
        X_val, y_val = load_dataset_from_directory(args.val_dir)
        logger.info(f"Loaded {len(X_val)} validation images")
    else:
        logger.info("No --val-dir provided, splitting training data 80/20")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

    # Preprocess all data
    logger.info("Preprocessing images...")
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)

    # Stage 1: Minor vs Non-minor
    stage1_model = None
    if '1' not in args.skip_stages.split(','):
        stage1_model, _ = train_stage1(X_train, y_train, X_val, y_val)
        stage1_model.save(os.path.join(output_dir, 'stage1_minor_vs_nonminor.keras'))

    # Stage 2: Severe vs Non-severe
    stage2_model = None
    if '2' not in args.skip_stages.split(','):
        stage2_model, _ = train_stage2(X_train, y_train, X_val, y_val)
        stage2_model.save(os.path.join(output_dir, 'stage2_severe_vs_nonsevere.keras'))

    # Stage 3: Full 3-class with curriculum (builds its own model from ImageNet weights)
    final_model = train_stage3(
        X_train, y_train, X_val, y_val,
        use_curriculum=args.use_curriculum
    )

    # Save final model
    final_model.save(os.path.join(output_dir, 'final_curriculum_model.keras'))

    # Save class mapping
    class_mapping = {
        '0': '01-minor',
        '1': '02-moderate',
        '2': '03-severe'
    }
    with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)

    logger.info(f"\nTraining complete! Models saved to: {output_dir}")

    return final_model


if __name__ == '__main__':
    main()
