"""Unified inference and freshness grading engine for Fruit Freshness CBP.

Loads trained models, performs preprocessing, extracts deep representations,
computes class probabilities, freshness scores, quality grades, and visual diagnostics.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib

import config
from src.preprocessing import load_and_preprocess_image, extract_visual_features


class FruitFreshnessPredictor:
    """Production inference pipeline supporting Hybrid SVM, MobileNetV2, and CNN models."""

    def __init__(self, model_type: str = "hybrid"):
        self.model_type = model_type.lower()
        self.encoder = None
        self.extractor = None
        self.svm_clf = None
        self.scaler = None
        self.nn_model = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load necessary serialized artifacts based on selected model architecture."""
        if not config.LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(
                f"Label encoder not found at {config.LABEL_ENCODER_PATH}. "
                "Please train a model first using src.trainhybrid, src.trainmobilenet, or src.traincnn."
            )
        self.encoder = joblib.load(config.LABEL_ENCODER_PATH)

        if self.model_type in ("hybrid", "svm"):
            if not (config.HYBRID_SVM_PATH.exists() and config.HYBRID_SCALER_PATH.exists()):
                raise FileNotFoundError(
                    f"Hybrid artifacts missing ({config.HYBRID_SVM_PATH} or {config.HYBRID_SCALER_PATH})."
                )
            self.svm_clf = joblib.load(config.HYBRID_SVM_PATH)
            self.scaler = joblib.load(config.HYBRID_SCALER_PATH)

            if config.FEATURE_EXTRACTOR_PATH.exists():
                self.extractor = tf.keras.models.load_model(str(config.FEATURE_EXTRACTOR_PATH))
            else:
                from src.trainmobilenet import build_mobilenet_extractor
                self.extractor = build_mobilenet_extractor()

        elif self.model_type == "mobilenet":
            if not config.MOBILENET_MODEL_PATH.exists():
                raise FileNotFoundError(f"MobileNet model not found at {config.MOBILENET_MODEL_PATH}.")
            self.nn_model = tf.keras.models.load_model(str(config.MOBILENET_MODEL_PATH))

        elif self.model_type == "cnn":
            if not config.CNN_MODEL_PATH.exists():
                raise FileNotFoundError(f"CNN model not found at {config.CNN_MODEL_PATH}.")
            self.nn_model = tf.keras.models.load_model(str(config.CNN_MODEL_PATH))
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Choose 'hybrid', 'mobilenet', or 'cnn'.")

    def predict(self, image_input: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """Execute complete inference, scoring, grading, and visual explanation pipeline."""
        # Preprocess image
        raw_rgb = load_and_preprocess_image(image_input, normalize_mode="raw")
        visual_diag = extract_visual_features(raw_rgb)

        if self.model_type in ("hybrid", "svm"):
            # Normalize with MobileNet scheme ([-1, 1])
            input_tensor = load_and_preprocess_image(image_input, normalize_mode="mobilenet")
            batch = np.expand_dims(input_tensor, axis=0)

            # Extract 1280-D deep feature vector
            deep_features = self.extractor(batch, training=False).numpy()
            deep_features_scaled = self.scaler.transform(deep_features)

            probs = self.svm_clf.predict_proba(deep_features_scaled)[0]
            pred_idx = int(np.argmax(probs))

        elif self.model_type == "mobilenet":
            input_tensor = load_and_preprocess_image(image_input, normalize_mode="mobilenet")
            batch = np.expand_dims(input_tensor, axis=0)
            probs = self.nn_model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))

        elif self.model_type == "cnn":
            input_tensor = load_and_preprocess_image(image_input, normalize_mode="standard")
            batch = np.expand_dims(input_tensor, axis=0)
            probs = self.nn_model.predict(batch, verbose=0)[0]
            pred_idx = int(np.argmax(probs))

        predicted_class = self.encoder.classes_[pred_idx]
        confidence = float(probs[pred_idx])

        # Assemble probability dictionary sorted by confidence
        class_probabilities = {
            self.encoder.classes_[i]: float(probs[i])
            for i in np.argsort(probs)[::-1]
        }

        # Compute Freshness Quality Metrics
        freshness_score, quality_grade, is_edible, recommendation = self._compute_freshness_grade(
            predicted_class, confidence, visual_diag["blemish_ratio"]
        )

        # Do not assign a definitive freshness grade to uncertain predictions.
        if confidence < 0.50:
            freshness_score = None
            quality_grade = "Uncertain: Low Model Confidence"
            is_edible = None
            recommendation = (
                "The model is uncertain. Try another clear image. "
                "This prediction cannot determine whether the fruit is safe to eat."
            )

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "model_type": self.model_type,
            "freshness_score": freshness_score,
            "quality_grade": quality_grade,
            "is_edible": is_edible,
            "recommendation": recommendation,
            "class_probabilities": class_probabilities,
            "visual_diagnostics": visual_diag
        }

    @staticmethod
    def _compute_freshness_grade(
        predicted_class: str,
        confidence: float,
        blemish_ratio: float
    ) -> tuple:
        """Derive objective freshness score (0-100) and market quality grade."""
        cls_lower = predicted_class.lower()

        if "fresh" in cls_lower:
            base_score = 90.0
            blemish_penalty = min(20.0, blemish_ratio * 40.0)
            freshness_score = max(75.0, round(base_score * confidence - blemish_penalty, 1))
            quality_grade = "Grade A: Premium Fresh"
            is_edible = True
            recommendation = "Optimal for immediate consumption, premium retail sale, or export."

        elif "normal" in cls_lower:
            base_score = 70.0
            blemish_penalty = min(25.0, blemish_ratio * 35.0)
            freshness_score = max(50.0, round(base_score * confidence - blemish_penalty, 1))
            quality_grade = "Grade B: Standard Edible"
            is_edible = True
            recommendation = "Good for standard consumer market, domestic cooking, or quick processing."

        else:  # Rotten / Spoiled
            base_score = 30.0
            freshness_score = min(45.0, max(5.0, round(base_score * (1.0 - confidence) + (1.0 - blemish_ratio) * 15.0, 1)))
            if freshness_score < 25.0:
                quality_grade = "Grade D: Spoiled / Severely Rotten"
                is_edible = False
                recommendation = "Unsafe for human consumption. Dispose or allocate to industrial compost/biogas."
            else:
                quality_grade = "Grade C: Degraded / Sub-standard"
                is_edible = False
                recommendation = "Visible spoilage detected. Unfit for retail; divert from food supply chain."

        return freshness_score, quality_grade, is_edible, recommendation


def predict_single_image(image_path: str, model_type: str = "hybrid") -> Dict[str, Any]:
    """Helper function to run inference on a single image path."""
    predictor = FruitFreshnessPredictor(model_type=model_type)
    return predictor.predict(image_path)
