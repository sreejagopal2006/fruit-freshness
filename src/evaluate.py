"""Comprehensive model evaluation and comparative benchmarking module.

Evaluates Custom CNN, MobileNetV2, and Hybrid MobileNetV2+SVM on the test split,
generates confusion matrices, calculates precision/recall/F1, and outputs comparison metrics.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
import tensorflow as tf

import config
from src.dataloader import FruitDataLoader
from src.preprocessing import create_tf_dataset


def evaluate_all_models(data_dir: Path = None) -> Dict[str, Any]:
    """Evaluate all trained models and generate comparison artifacts."""
    loader = FruitDataLoader(data_dir or config.RAW_DATA_DIR)
    _, _, test_df = loader.get_splits()

    if not config.LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"LabelEncoder not found at {config.LABEL_ENCODER_PATH}. "
            f"Please run a training script first (e.g., python -m src.trainhybrid)."
        )

    encoder = joblib.load(config.LABEL_ENCODER_PATH)
    test_df["label_idx"] = encoder.transform(test_df["class"])
    y_true = test_df["label_idx"].values
    class_names = encoder.classes_

    comparison_results = {}

    # -------------------------------------------------------------
    # 1. Evaluate Hybrid MobileNetV2 + SVM Model
    # -------------------------------------------------------------
    if config.HYBRID_SVM_PATH.exists() and config.HYBRID_SCALER_PATH.exists() and config.FEATURE_EXTRACTOR_PATH.exists():
        print("\n[Evaluate] Assessing Hybrid MobileNetV2 + SVM Model...")
        svm_clf = joblib.load(config.HYBRID_SVM_PATH)
        scaler = joblib.load(config.HYBRID_SCALER_PATH)
        extractor = tf.keras.models.load_model(str(config.FEATURE_EXTRACTOR_PATH))

        test_ds = create_tf_dataset(test_df, batch_size=config.BATCH_SIZE, is_training=False, normalize_mode="mobilenet")
        feats_list = [
            extractor(batch_x[0] if isinstance(batch_x, (tuple, list)) else batch_x, training=False).numpy()
            for batch_x in test_ds
        ]
        X_test = np.vstack(feats_list)
        X_test_scaled = scaler.transform(X_test)

        y_pred = svm_clf.predict(X_test_scaled)
        comparison_results["Hybrid MobileNetV2 + SVM"] = compute_metrics(
            y_true, y_pred, class_names, "Hybrid MobileNetV2 + SVM"
        )

    # -------------------------------------------------------------
    # 2. Evaluate MobileNetV2 Transfer Learning Model
    # -------------------------------------------------------------
    if config.MOBILENET_MODEL_PATH.exists():
        print("\n[Evaluate] Assessing MobileNetV2 Transfer Learning Model...")
        mobilenet_model = tf.keras.models.load_model(str(config.MOBILENET_MODEL_PATH))
        test_ds = create_tf_dataset(test_df, batch_size=config.BATCH_SIZE, is_training=False, normalize_mode="mobilenet")

        probs = mobilenet_model.predict(test_ds, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        comparison_results["MobileNetV2 (End-to-End)"] = compute_metrics(
            y_true, y_pred, class_names, "MobileNetV2"
        )

    # -------------------------------------------------------------
    # 3. Evaluate Custom CNN Baseline
    # -------------------------------------------------------------
    if config.CNN_MODEL_PATH.exists():
        print("\n[Evaluate] Assessing Custom CNN Baseline Model...")
        cnn_model = tf.keras.models.load_model(str(config.CNN_MODEL_PATH))
        test_ds = create_tf_dataset(test_df, batch_size=config.BATCH_SIZE, is_training=False, normalize_mode="standard")

        probs = cnn_model.predict(test_ds, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        comparison_results["Custom CNN Baseline"] = compute_metrics(
            y_true, y_pred, class_names, "Custom CNN Baseline"
        )

    # Save comparison summary
    with open(config.COMPARISON_REPORT_PATH, "w") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n=======================================================")
    print("MODEL COMPARATIVE BENCHMARK SUMMARY (Test Set)")
    print("=======================================================")
    for model_name, res in comparison_results.items():
        print(f"Model: {model_name:<30} | Accuracy: {res['accuracy']*100:.2f}% | F1-Score: {res['weighted_f1']*100:.2f}%")
    print("=======================================================\n")
    print(f"Comparison report saved to: {config.COMPARISON_REPORT_PATH}")

    return comparison_results


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    model_tag: str
) -> Dict[str, Any]:
    """Calculate statistical metrics and save Seaborn confusion matrix plot."""
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix: {model_tag}\nAccuracy: {acc*100:.2f}%", fontsize=14, pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    tag_clean = model_tag.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
    cm_path = config.REPORTS_DIR / f"cm_{tag_clean}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Also save primary confusion matrix to config path
    if "hybrid" in tag_clean:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix: {model_tag}\nAccuracy: {acc*100:.2f}%", fontsize=14, pad=12)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("True Class", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=300)
        plt.close()

    clf_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "weighted_precision": float(p),
        "weighted_recall": float(r),
        "weighted_f1": float(f1),
        "confusion_matrix_path": str(cm_path.relative_to(config.BASE_DIR)) if cm_path.is_relative_to(config.BASE_DIR) else str(cm_path),
        "classification_report": clf_rep
    }


if __name__ == "__main__":
    evaluate_all_models()
