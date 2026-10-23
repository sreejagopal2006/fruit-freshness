"""Hybrid Machine Learning training module for Fruit Freshness CBP.

Combines MobileNetV2 deep convolutional representations with a Scikit-learn
Support Vector Machine (SVM) classifier for fast, robust decision boundaries.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

import config
from src.dataloader import FruitDataLoader
from src.preprocessing import create_tf_dataset
from src.trainmobilenet import build_mobilenet_extractor


def extract_features_in_batches(
    extractor: tf.keras.Model,
    df: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """Pass images through MobileNetV2 extractor to extract 1280-D feature vectors."""
    print(f"[Hybrid] Extracting 1280-D features for {len(df)} samples...")
    ds = create_tf_dataset(df, batch_size=batch_size, is_training=False, normalize_mode="mobilenet")

    feature_batches = []
    for item in ds:
        # ds returns (images, labels)
        if isinstance(item, (tuple, list)):
            batch_imgs = item[0]
        else:
            batch_imgs = item

        feats = extractor(batch_imgs, training=False).numpy()
        feature_batches.append(feats)

    features = np.vstack(feature_batches)
    labels = df["label_idx"].values
    return features, labels


def train_hybrid(
    kernel: str = config.SVM_KERNEL,
    c_param: float = config.SVM_C,
    max_samples_per_class: int = None,
    data_dir: Path = None
):
    """Execute deep feature extraction and train Support Vector Machine."""
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

    # Load or build feature extractor
    if config.FEATURE_EXTRACTOR_PATH.exists():
        print(f"[Hybrid] Loading existing feature extractor from {config.FEATURE_EXTRACTOR_PATH}")
        extractor = tf.keras.models.load_model(str(config.FEATURE_EXTRACTOR_PATH))
    else:
        print("[Hybrid] Building MobileNetV2 feature extractor...")
        extractor = build_mobilenet_extractor()
        extractor.save(str(config.FEATURE_EXTRACTOR_PATH))

    # 1. Feature Extraction
    X_train, y_train = extract_features_in_batches(extractor, train_df)
    X_val, y_val = extract_features_in_batches(extractor, val_df)
    X_test, y_test = extract_features_in_batches(extractor, test_df)

    # 2. Feature Standardization
    print("[Hybrid] Fitting StandardScaler on 1280-D feature vectors...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Persist scaler
    joblib.dump(scaler, config.HYBRID_SCALER_PATH)

    # 3. Train Support Vector Classifier
    print(f"[Hybrid] Fitting Support Vector Classifier (kernel={kernel}, C={c_param})...")
    svm_clf = SVC(
        C=c_param,
        kernel=kernel,
        gamma=config.SVM_GAMMA,
        probability=config.SVM_PROBABILITY,
        random_state=config.RANDOM_SEED
    )
    svm_clf.fit(X_train_scaled, y_train)

    # 4. Validation & Test Evaluation
    val_preds = svm_clf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"\n[Hybrid] Validation Accuracy: {val_acc * 100:.2f}%")

    test_preds = svm_clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"[Hybrid] Test Accuracy: {test_acc * 100:.2f}%\n")

    report = classification_report(
        y_test,
        test_preds,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    print(classification_report(y_test, test_preds, target_names=encoder.classes_, zero_division=0))

    # 5. Persist Model and Report
    joblib.dump(svm_clf, config.HYBRID_SVM_PATH)
    print(f"[Hybrid] Successfully saved Hybrid SVM model to {config.HYBRID_SVM_PATH}")

    report_path = config.REPORTS_DIR / "hybrid_svm_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return svm_clf, scaler, encoder, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid MobileNetV2 + SVM model.")
    parser.add_argument("--kernel", type=str, default=config.SVM_KERNEL)
    parser.add_argument("--c", type=float, default=config.SVM_C)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    args = parser.parse_args()

    train_hybrid(
        kernel=args.kernel,
        c_param=args.c,
        max_samples_per_class=args.max_samples_per_class
    )
