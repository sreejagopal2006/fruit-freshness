"""MobileNetV2 transfer learning module for Fruit Freshness and Quality Prediction CBP.

Utilizes ImageNet pre-trained feature representations, freezes base convolutions,
fine-tunes a specialized classification head, and exports the 1280-D feature extractor.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from sklearn.preprocessing import LabelEncoder
import joblib

import config
from src.dataloader import FruitDataLoader
from src.preprocessing import create_tf_dataset


def build_mobilenet_extractor(input_shape=config.INPUT_SHAPE) -> tf.keras.Model:
    """Instantiate MobileNetV2 feature extractor with global average pooling (1280-D latent vectors)."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        pooling="avg",
        weights="imagenet"
    )
    base_model.trainable = False
    return base_model


def build_mobilenet_classifier(num_classes: int, input_shape=config.INPUT_SHAPE) -> tf.keras.Model:
    """Build complete MobileNetV2 transfer learning model with classification head."""
    extractor = build_mobilenet_extractor(input_shape=input_shape)

    inputs = layers.Input(shape=input_shape)
    feat = extractor(inputs)
    x = layers.BatchNormalization()(feat)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_fruit_classifier")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_mobilenet(
    epochs: int = config.MOBILENET_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    max_samples_per_class: int = None,
    data_dir: Path = None
):
    """Train MobileNetV2 transfer learning model on fruit dataset."""
    loader = FruitDataLoader(data_dir or config.RAW_DATA_DIR)
    train_df, val_df, test_df = loader.get_splits(max_samples_per_class=max_samples_per_class)

    # Encode labels
    if config.LABEL_ENCODER_PATH.exists():
        encoder = joblib.load(config.LABEL_ENCODER_PATH)
    else:
        encoder = LabelEncoder()
        encoder.fit(train_df["class"])
        joblib.dump(encoder, config.LABEL_ENCODER_PATH)

    train_df["label_idx"] = encoder.transform(train_df["class"])
    val_df["label_idx"] = encoder.transform(val_df["class"])
    test_df["label_idx"] = encoder.transform(test_df["class"])

    # Prepare datasets using MobileNet normalization ([-1, 1])
    train_ds = create_tf_dataset(train_df, batch_size=batch_size, is_training=True, normalize_mode="mobilenet")
    val_ds = create_tf_dataset(val_df, batch_size=batch_size, is_training=False, normalize_mode="mobilenet")

    model = build_mobilenet_classifier(num_classes=len(encoder.classes_))
    model.summary()

    # Also save the standalone feature extractor for the Hybrid model
    extractor = build_mobilenet_extractor()
    extractor.save(str(config.FEATURE_EXTRACTOR_PATH))
    print(f"[MobileNet] Saved standalone feature extractor to {config.FEATURE_EXTRACTOR_PATH}")

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        callbacks.ModelCheckpoint(str(config.MOBILENET_MODEL_PATH), monitor="val_accuracy", save_best_only=True)
    ]

    print(f"\n[MobileNet] Training transfer learning model for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        verbose=1
    )

    model.save(str(config.MOBILENET_MODEL_PATH))
    print(f"[MobileNet] Successfully saved classifier to {config.MOBILENET_MODEL_PATH}")

    # Save history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = config.REPORTS_DIR / "mobilenet_training_history.json"
    with open(hist_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV2 transfer learning model.")
    parser.add_argument("--epochs", type=int, default=config.MOBILENET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    args = parser.parse_args()

    train_mobilenet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples_per_class
    )
