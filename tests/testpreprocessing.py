"""Unit tests for preprocessing, image transformations, and OpenCV visual extraction."""

import pytest
import numpy as np
from PIL import Image
import config
from src.preprocessing import load_and_preprocess_image, extract_visual_features


@pytest.fixture
def sample_pil_image():
    """Generate a clean synthetic RGB test image."""
    arr = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_rgba_image():
    """Generate a synthetic RGBA test image to verify alpha channel handling."""
    arr = np.random.randint(50, 200, (150, 150, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def test_standard_preprocessing_shape(sample_pil_image):
    """Verify output shape matches configuration (224, 224, 3)."""
    processed = load_and_preprocess_image(sample_pil_image, normalize_mode="standard")
    assert processed.shape == (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_mobilenet_normalization_range(sample_pil_image):
    """Verify MobileNet normalization scales between -1.0 and 1.0."""
    processed = load_and_preprocess_image(sample_pil_image, normalize_mode="mobilenet")
    assert processed.shape == (224, 224, 3)
    assert processed.min() >= -1.01
    assert processed.max() <= 1.01


def test_rgba_to_rgb_conversion(sample_rgba_image):
    """Verify RGBA images are gracefully converted to 3-channel RGB."""
    processed = load_and_preprocess_image(sample_rgba_image, normalize_mode="raw")
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.uint8


def test_visual_feature_extraction():
    """Verify OpenCV diagnostics return expected dictionary keys and shapes."""
    dummy_img = np.full((224, 224, 3), 120, dtype=np.uint8)
    # Add a fake dark blemish region
    dummy_img[50:100, 50:100] = [20, 20, 20]

    diag = extract_visual_features(dummy_img)

    assert "rgb" in diag
    assert "hsv" in diag
    assert "gray" in diag
    assert "edges" in diag
    assert "blemish_mask" in diag
    assert "blemish_ratio" in diag
    assert 0.0 <= diag["blemish_ratio"] <= 1.0
    assert diag["edges"].shape == (224, 224)
    assert diag["blemish_mask"].shape == (224, 224)


def test_invalid_input_type():
    """Verify TypeError raised when invalid image object is passed."""
    with pytest.raises(TypeError):
        load_and_preprocess_image(12345)


def test_corrupt_image_bytes():
    """Verify handling when invalid or corrupt bytes are passed."""
    with pytest.raises(Exception):
        load_and_preprocess_image(b"not_an_image_corrupt_bytes")


def test_uniform_blank_image_variance():
    """Verify that a uniform blank image has near-zero standard deviation."""
    blank_arr = np.zeros((100, 100, 3), dtype=np.uint8)
    assert float(np.std(blank_arr)) == 0.0
    processed = load_and_preprocess_image(blank_arr, normalize_mode="standard")
    assert processed.shape == (224, 224, 3)
