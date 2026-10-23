"""Custom Convolutional Neural Network training module for Fruit Freshness CBP.

Builds, compiles, trains, and serializes the baseline CNN architecture.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import LabelEncoder
import joblib

import config
from src.dataloader import FruitDataLoader
from src.preprocessing import create_tf_dataset


def build_cnn_model(num_classes: int, input_shape=config.INPUT_SHAPE) -> tf.keras.Model:
    """Construct lightweight, regularized CNN for fruit quality classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 4
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Dense Classification Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_cnn(
    epochs: int = config.CNN_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    max_samples_per_class: int = None,
    data_dir: Path = None
):
    """Orchestrate end-to-end dataset loading, model training, and artifact persistence."""
    loader = FruitDataLoader(data_dir or config.RAW_DATA_DIR)
    train_df, val_df, test_df = loader.get_splits(max_samples_per_class=max_samples_per_class)

    # Encode labels
    encoder = LabelEncoder()
    train_df["label_idx"] = encoder.fit_transform(train_df["class"])
    val_df["label_idx"] = encoder.transform(val_df["class"])
    test_df["label_idx"] = encoder.transform(test_df["class"])

    # Persist label encoder
    joblib.dump(encoder, config.LABEL_ENCODER_PATH)
    print(f"[CNN] Saved LabelEncoder to {config.LABEL_ENCODER_PATH} ({len(encoder.classes_)} classes)")

    # Prepare tf.data pipelines
    train_ds = create_tf_dataset(train_df, batch_size=batch_size, is_training=True, normalize_mode="standard")
    val_ds = create_tf_dataset(val_df, batch_size=batch_size, is_training=False, normalize_mode="standard")

    model = build_cnn_model(num_classes=len(encoder.classes_))
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        callbacks.ModelCheckpoint(str(config.CNN_MODEL_PATH), monitor="val_accuracy", save_best_only=True)
    ]

    print(f"\n[CNN] Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1
    )

    # Save final model
    model.save(str(config.CNN_MODEL_PATH))
    print(f"[CNN] Successfully serialized model to {config.CNN_MODEL_PATH}")

    # Save history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = config.REPORTS_DIR / "cnn_training_history.json"
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    return model, encoder, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline CNN model.")
    parser.add_argument("--epochs", type=int, default=config.CNN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    args = parser.parse_args()

    train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples_per_class
    )
