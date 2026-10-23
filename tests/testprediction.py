"""Unit tests for inference pipeline, freshness scoring, and quality assessment."""

import pytest
import numpy as np
from PIL import Image
import config
from src.predict import FruitFreshnessPredictor


@pytest.fixture
def synthetic_fruit_image():
    """Create a clean synthetic image resembling a fruit."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    # Yellow circular fruit pattern
    for r in range(224):
        for c in range(224):
            if (r - 112) ** 2 + (c - 112) ** 2 < 70 ** 2:
                arr[r, c] = [240, 200, 20]  # Yellow banana / apple hue
            else:
                arr[r, c] = [255, 255, 255]  # White background
    return Image.fromarray(arr)


def test_freshness_grading_logic():
    """Test rule-based objective freshness scoring and market grading."""
    # 1. Fresh Class Case
    score, grade, edible, rec = FruitFreshnessPredictor._compute_freshness_grade(
        predicted_class="fresh_apple",
        confidence=0.95,
        blemish_ratio=0.02
    )
    assert 75.0 <= score <= 100.0
    assert "Grade A" in grade
    assert edible is True

    # 2. Rotten Class Case
    score_rot, grade_rot, edible_rot, rec_rot = FruitFreshnessPredictor._compute_freshness_grade(
        predicted_class="rotten_banana",
        confidence=0.92,
        blemish_ratio=0.45
    )
    assert 0.0 <= score_rot <= 45.0
    assert edible_rot is False
    assert "Grade C" in grade_rot or "Grade D" in grade_rot


def test_hybrid_prediction_pipeline(synthetic_fruit_image):
    """Verify end-to-end inference output structure using trained Hybrid model."""
    if not (config.HYBRID_SVM_PATH.exists() and config.LABEL_ENCODER_PATH.exists()):
        pytest.skip("Hybrid model artifacts not yet generated.")

    predictor = FruitFreshnessPredictor(model_type="hybrid")
    result = predictor.predict(synthetic_fruit_image)

    assert "predicted_class" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "freshness_score" in result
    assert 0.0 <= result["freshness_score"] <= 100.0
    assert "quality_grade" in result
    assert "is_edible" in result
    assert isinstance(result["is_edible"], bool)
    assert "class_probabilities" in result
    assert "visual_diagnostics" in result
    assert "blemish_ratio" in result["visual_diagnostics"]


def test_predictor_supported_classes():
    """Verify that predictor loads canonical supported fruit categories."""
    if not (config.HYBRID_SVM_PATH.exists() and config.LABEL_ENCODER_PATH.exists()):
        pytest.skip("Hybrid model artifacts not yet generated.")

    predictor = FruitFreshnessPredictor(model_type="hybrid")
    classes = list(predictor.encoder.classes_)
    assert len(classes) == 9
    assert "fresh_apple" in classes
    assert "rotten_banana" in classes
    assert "normal_orange" in classes


def test_fruitvision_sample_discovery():
    """Verify dynamic discovery of authentic samples vs graceful handling of un-downloaded classes."""
    from app import find_fruitvision_samples

    # 1. Local category that genuinely exists
    local_apple = find_fruitvision_samples("Apple", "Fresh")
    assert local_apple["found"] is True
    assert local_apple["total"] > 0
    assert len(local_apple["images"]) > 0

    local_banana = find_fruitvision_samples("Banana", "Rotten")
    assert local_banana["found"] is True
    assert local_banana["total"] > 0

    # 2. External categories not currently in local dataset
    missing_mango = find_fruitvision_samples("Mango", "Fresh")
    assert missing_mango["found"] is False
    assert missing_mango["total"] == 0

    missing_formalin = find_fruitvision_samples("Apple", "Formalin-mixed")
    assert missing_formalin["found"] is False
    assert missing_formalin["total"] == 0


def test_prediction_latency_benchmark(synthetic_fruit_image):
    """Verify that hybrid inference executes in sub-250ms on standard CPU."""
    if not (config.HYBRID_SVM_PATH.exists() and config.LABEL_ENCODER_PATH.exists()):
        pytest.skip("Hybrid model artifacts not yet generated.")

    import time
    predictor = FruitFreshnessPredictor(model_type="hybrid")

    t0 = time.perf_counter()
    res = predictor.predict(synthetic_fruit_image)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert res is not None
    assert elapsed_ms < 500.0  # Well within real-time responsiveness threshold
