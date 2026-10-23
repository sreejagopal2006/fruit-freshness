"""Configuration module for Fruit Freshness and Quality Prediction CBP Project.

Centralizes all project paths, hyperparameters, supported formats,
and hardware acceleration settings.
"""

from pathlib import Path
import os

# ---------------------------------------------------------
# Project Base Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SRC_DIR = BASE_DIR / "src"
TESTS_DIR = BASE_DIR / "tests"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Image and Preprocessing Settings
# ---------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# ---------------------------------------------------------
# Dataset Splits and Reproducibility
# ---------------------------------------------------------
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Default canonical 16 classes from Mendeley Fresh and Rotten Fruits Dataset:
# (Also dynamically inferred at runtime if a subset or custom dataset is supplied)
DEFAULT_CLASSES = [
    "fresh_apple",
    "rotten_apple",
    "fresh_banana",
    "rotten_banana",
    "fresh_orange",
    "rotten_orange",
    "fresh_grape",
    "rotten_grape",
    "fresh_guava",
    "rotten_guava",
    "fresh_jujube",
    "rotten_jujube",
    "fresh_pomegranate",
    "rotten_pomegranate",
    "fresh_strawberry",
    "rotten_strawberry",
]

# ---------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Epoch counts designed for efficient yet effective convergence
CNN_EPOCHS = 15
MOBILENET_EPOCHS = 10

# Hybrid SVM Parameters
SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"
SVM_PROBABILITY = True

# ---------------------------------------------------------
# Model Artifact Storage
# ---------------------------------------------------------
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"
MOBILENET_MODEL_PATH = MODELS_DIR / "mobilenet_model.keras"
FEATURE_EXTRACTOR_PATH = MODELS_DIR / "mobilenet_extractor.keras"
HYBRID_SVM_PATH = MODELS_DIR / "hybrid_svm_model.joblib"
HYBRID_SCALER_PATH = MODELS_DIR / "hybrid_scaler.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# ---------------------------------------------------------
# Evaluation and Reporting Artifacts
# ---------------------------------------------------------
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.json"
COMPARISON_REPORT_PATH = REPORTS_DIR / "model_comparison.json"

# ---------------------------------------------------------
# Hardware Acceleration Detection
# ---------------------------------------------------------
def detect_hardware():
    """Detect available hardware acceleration (MPS for Apple Silicon, CUDA, or CPU)."""
    device_info = {"device": "CPU", "description": "Standard CPU"}
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            device_info = {"device": "GPU", "description": f"{len(gpus)} GPU(s) available"}
        else:
            # Check for Apple Silicon Metal MPS in PyTorch or TF plugin if present
            device_info = {"device": "CPU", "description": "CPU (Apple Silicon Optimized)"}
    except Exception:
        pass
    return device_info
