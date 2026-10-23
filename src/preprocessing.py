"""Image preprocessing, computer vision transformations, and feature engineering.

Provides standardization to 224x224x3, normalization schemes, OpenCV
color space analysis (HSV, RGB), blemish detection, edge maps, and data augmentation.
"""

from typing import Tuple, Union, Optional, Dict, Any
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

import config


def load_and_preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
    target_size: Tuple[int, int] = (config.IMG_WIDTH, config.IMG_HEIGHT),
    normalize_mode: str = "standard"
) -> np.ndarray:
    """Load an image from various input types and standardize to target_size RGB.

    Args:
        image_input: File path, PIL Image, raw bytes, or numpy array.
        target_size: (width, height) tuple, defaults to (224, 224).
        normalize_mode: 'standard' ([0, 1]), 'mobilenet' ([-1, 1]), or 'raw' ([0, 255]).

    Returns:
        np.ndarray of shape (height, width, 3), dtype float32 (or uint8 if raw).
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    elif isinstance(image_input, bytes):
        import io
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        # Convert BGR or Grayscale numpy array if needed
        if image_input.ndim == 2:
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB))
        elif image_input.shape[2] == 4:
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB))
        elif image_input.shape[2] == 3:
            # Assume uint8 RGB or BGR
            img = Image.fromarray(image_input.astype(np.uint8))
        else:
            raise ValueError(f"Unsupported array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Ensure 3-channel RGB mode (handles PNG alpha channels or grayscale)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize with high-quality Lanczos resampling
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32)

    if normalize_mode == "standard":
        return arr / 255.0
    elif normalize_mode == "mobilenet":
        return mobilenet_preprocess(arr.copy())
    elif normalize_mode == "raw":
        return arr.astype(np.uint8)
    else:
        raise ValueError(f"Unknown normalize_mode: '{normalize_mode}'. Choose 'standard', 'mobilenet', or 'raw'.")


def extract_visual_features(image_rgb: np.ndarray) -> Dict[str, Any]:
    """Perform computer vision analysis using OpenCV: HSV color, Canny edges, and blemish mask.

    Args:
        image_rgb: Numpy array (224, 224, 3) in [0, 255] or [0, 1].

    Returns:
        Dictionary containing:
            - 'hsv': HSV representation
            - 'gray': Grayscale representation
            - 'edges': Canny edge detection map
            - 'blemish_mask': Binary mask of brown/dark decayed regions
            - 'blemish_ratio': Estimated percentage of surface with discoloration
            - 'mean_color_rgb': Average RGB values
    """
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            uint8_img = (image_rgb * 255.0).clip(0, 255).astype(np.uint8)
        else:
            uint8_img = image_rgb.clip(0, 255).astype(np.uint8)
    else:
        uint8_img = image_rgb

    # 1. Grayscale
    gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

    # 2. Canny Edge Detection (wrinkling / structural degradation)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 3. HSV Color Space Conversion
    hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)

    # 4. Decay/Blemish Detection (brown/black necrotic tissue thresholding in HSV)
    # Brown hues in HSV typically span H: [10, 30], S: [40, 255], V: [20, 150]
    lower_brown = np.array([5, 40, 20])
    upper_brown = np.array([30, 255, 140])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Dark spots (low value)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 60])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    blemish_mask = cv2.bitwise_or(brown_mask, dark_mask)

    # Compute foreground mask (exclude bright plain white or black background)
    non_bg_mask = cv2.inRange(gray, 20, 245)
    foreground_pixels = max(1, cv2.countNonZero(non_bg_mask))
    decayed_pixels = cv2.countNonZero(cv2.bitwise_and(blemish_mask, non_bg_mask))
    blemish_ratio = float(decayed_pixels / foreground_pixels)

    mean_rgb = np.mean(uint8_img, axis=(0, 1)).round(2).tolist()

    return {
        "rgb": uint8_img,
        "hsv": hsv,
        "gray": gray,
        "edges": edges,
        "blemish_mask": blemish_mask,
        "blemish_ratio": blemish_ratio,
        "mean_color_rgb": mean_rgb
    }


def create_tf_dataset(
    df,
    batch_size: int = config.BATCH_SIZE,
    is_training: bool = True,
    normalize_mode: str = "standard",
    shuffle_buffer: int = 1000
) -> tf.data.Dataset:
    """Build high-performance tf.data.Dataset pipeline with parallel mapping and prefetching."""
    image_paths = df["image_path"].values
    labels = df["label_idx"].values if "label_idx" in df.columns else None

    def _load_and_prep(path, label=None):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [config.IMG_HEIGHT, config.IMG_WIDTH])

        if normalize_mode == "standard":
            img = img / 255.0
        elif normalize_mode == "mobilenet":
            img = mobilenet_preprocess(img)

        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)

        if label is not None:
            return img, label
        return img

    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(_load_and_prep, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(_load_and_prep, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(shuffle_buffer, seed=config.RANDOM_SEED)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
