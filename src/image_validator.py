"""Image validation stage that runs BEFORE the freshness classifier.

Two clearly separate checks:

1. Image-quality / integrity checks (no AI): empty file, corrupted file, tiny
   image, blank image, extreme blur. These say nothing about *what* is in the
   picture.
2. Semantic fruit check: a pretrained ImageNet MobileNetV2 *classification head*
   (1000 classes) decides whether the picture looks like a supported fruit.
   ImageNet has no "person" class, so a selfie is rejected because probability
   mass is NOT on fruit classes, not because a face was detected.

Fail-closed policy: if the semantic model is unavailable, or it errors, the
image is NOT accepted and freshness prediction is not run.

Known limitations (be honest about these in the demo):
* ImageNet only has a "Granny Smith" apple class, plus orange and banana. Red
  apples are often scored as pomegranate, so ``DEFAULT_ALIASES`` is lenient
  (pomegranate -> apple, lemon -> orange). A real pomegranate or lemon photo
  can therefore be accepted and classified as apple/orange.
* Very rotten fruit may look unlike ImageNet fruit and can be falsely rejected.
  Thresholds are NOT calibrated until you run scripts/calibrate_validator.py
  on your own fruit and non-fruit images.
* A selfie in which the person holds a banana can legitimately pass.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------
ACCEPTED = "accepted"
REJECTED_NOT_FRUIT = "rejected_not_fruit"
REJECTED_UNSUPPORTED_FRUIT = "rejected_unsupported_fruit"
REJECTED_QUALITY = "rejected_quality"
REJECTED_INVALID_FILE = "rejected_invalid_file"
UNAVAILABLE = "validator_unavailable"
ERROR = "validator_error"

# ImageNet-1k fruit classes (indices are fixed by the ImageNet class list).
FRUIT_INDEX_NAMES: Dict[int, str] = {
    948: "Granny Smith apple",
    949: "strawberry",
    950: "orange",
    951: "lemon",
    952: "fig",
    953: "pineapple",
    954: "banana",
    955: "jackfruit",
    956: "custard apple",
    957: "pomegranate",
}

# Supported fruit -> ImageNet class indices accepted as evidence for it.
DEFAULT_ALIASES: Dict[str, Tuple[int, ...]] = {
    "apple": (948, 957),
    "orange": (950, 951),
    "banana": (954,),
}

NUM_IMAGENET_CLASSES = 1000


class ValidatorUnavailable(RuntimeError):
    """The semantic validation model cannot be loaded (fail closed)."""


class InvalidImageError(ValueError):
    """The input could not be read as an image."""


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
def format_fruit_list(fruits: Sequence[str]) -> str:
    """['apple','banana','orange'] -> 'an Apple, Banana, or Orange'."""
    names = [f.strip().title() for f in fruits if f.strip()]
    if not names:
        return "a supported fruit"
    article = "an" if names[0][0].upper() in "AEIOU" else "a"
    if len(names) == 1:
        return f"{article} {names[0]}"
    if len(names) == 2:
        return f"{article} {names[0]} or {names[1]}"
    return f"{article} {', '.join(names[:-1])}, or {names[-1]}"


@dataclass
class ValidationResult:
    status: str
    image: Optional[Image.Image] = None
    fruit: Optional[str] = None  # supported fruit the validator believes it sees
    reason: str = ""  # specific, human-readable detail
    scores: Dict[str, float] = field(default_factory=dict)
    supported_fruits: Tuple[str, ...] = ("apple", "banana", "orange")

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    @property
    def user_message(self) -> str:
        fruits = format_fruit_list(self.supported_fruits)
        tail = "Freshness prediction was not performed."
        if self.status == ACCEPTED:
            return ""
        if self.status == REJECTED_NOT_FRUIT:
            return (
                "❌ Invalid Image: No supported fruit detected. Please upload or "
                f"capture a clear image of {fruits}. {tail}"
            )
        if self.status == REJECTED_UNSUPPORTED_FRUIT:
            return (
                f"❌ Unsupported fruit: the image looks like {self.reason}, but this "
                f"model only supports {fruits}. {tail}"
            )
        if self.status == REJECTED_QUALITY:
            return (
                f"❌ Image quality too low: {self.reason}. Please upload or capture a "
                f"clear, well-lit, in-focus photo of {fruits}. {tail}"
            )
        if self.status == REJECTED_INVALID_FILE:
            return (
                f"❌ Could not read this image: {self.reason}. Please upload a valid "
                f"JPG or PNG file. {tail}"
            )
        if self.status == UNAVAILABLE:
            return (
                f"⚠️ Image validation is unavailable ({self.reason}), so freshness "
                f"prediction is disabled to avoid accepting non-fruit images. {tail}"
            )
        return f"⚠️ Image validation failed ({self.reason}). {tail}"


# ---------------------------------------------------------------------------
# Loading and quality checks (no AI)
# ---------------------------------------------------------------------------
def load_image(source) -> Image.Image:
    """Return an upright RGB PIL image, or raise InvalidImageError."""
    try:
        if isinstance(source, Image.Image):
            if source.width == 0 or source.height == 0:
                raise InvalidImageError("the image has zero width or height")
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise InvalidImageError("the file was not found")
            data = path.read_bytes()
        elif isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        elif hasattr(source, "getvalue"):
            data = source.getvalue()
        elif hasattr(source, "read"):
            data = source.read()
        else:
            raise InvalidImageError(f"unsupported input type {type(source).__name__}")
    except InvalidImageError:
        raise
    except OSError as exc:
        raise InvalidImageError(f"the file could not be read ({exc})") from exc

    if not data:
        raise InvalidImageError("the file is empty (0 bytes)")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        EOFError,
        Image.DecompressionBombError,
    ) as exc:
        raise InvalidImageError("the file is corrupted or is not a valid image") from exc


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian (higher = sharper)."""
    g = gray.astype(np.float32)
    lap = (
        -4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def check_quality(
    img: Image.Image, min_side: int, blank_std: float, blur_threshold: float
) -> Optional[str]:
    """Return a reason string if the image fails a quality check, else None."""
    w, h = img.size
    if min(w, h) < min_side:
        return f"the image is very small ({w}×{h} px; at least {min_side}px needed)"
    work = img.convert("L")
    work.thumbnail((512, 512))
    gray = np.asarray(work, dtype=np.float32)
    if float(gray.std()) < blank_std:
        return "the image looks blank or a single flat colour"
    sharp = laplacian_variance(gray)
    if sharp < blur_threshold:
        return f"the image is too blurry (sharpness score {sharp:.1f})"
    return None


# ---------------------------------------------------------------------------
# Semantic decision (pure function so it can be unit-tested with mocked probs)
# ---------------------------------------------------------------------------
def decide_from_probabilities(
    probs: np.ndarray,
    supported_fruits: Sequence[str],
    aliases: Dict[str, Tuple[int, ...]],
    accept_mass: float,
) -> ValidationResult:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    supported = tuple(f.lower() for f in supported_fruits)
    if probs.shape[0] != NUM_IMAGENET_CLASSES or not np.all(np.isfinite(probs)):
        raise ValueError(
            f"classifier returned an invalid probability vector (shape {probs.shape})"
        )

    per_fruit = {
        fruit: float(sum(probs[i] for i in aliases.get(fruit, ()))) for fruit in supported
    }
    supported_idx = {i for f in supported for i in aliases.get(f, ())}
    supported_mass = sum(per_fruit.values())
    other_fruit_idx = [i for i in FRUIT_INDEX_NAMES if i not in supported_idx]
    other_fruit_mass = float(sum(probs[i] for i in other_fruit_idx))

    non_fruit = probs.copy()
    for i in FRUIT_INDEX_NAMES:
        non_fruit[i] = 0.0
    top_non_fruit_idx = int(np.argmax(non_fruit))
    top_non_fruit_prob = float(non_fruit[top_non_fruit_idx])

    scores = {
        "supported_fruit_mass": float(supported_mass),
        "other_fruit_mass": other_fruit_mass,
        "top_non_fruit_prob": top_non_fruit_prob,
        "top_non_fruit_index": float(top_non_fruit_idx),
        **{f"mass_{k}": v for k, v in per_fruit.items()},
    }
    best_fruit = max(per_fruit, key=per_fruit.get) if per_fruit else None

    if (
        supported_mass >= accept_mass
        and supported_mass >= top_non_fruit_prob
        and supported_mass >= other_fruit_mass
    ):
        return ValidationResult(
            status=ACCEPTED, fruit=best_fruit, scores=scores, supported_fruits=supported
        )

    if other_fruit_mass >= accept_mass and other_fruit_mass > supported_mass:
        best_other = max(other_fruit_idx, key=lambda i: probs[i])
        return ValidationResult(
            status=REJECTED_UNSUPPORTED_FRUIT,
            reason=FRUIT_INDEX_NAMES[best_other],
            scores=scores,
            supported_fruits=supported,
        )

    return ValidationResult(
        status=REJECTED_NOT_FRUIT, scores=scores, supported_fruits=supported
    )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
Classifier = Callable[[Image.Image], np.ndarray]


class ImageValidator:
    """Quality checks + ImageNet semantic check. ``classifier`` maps a PIL RGB
    image to a length-1000 probability vector (injectable for tests)."""

    def __init__(
        self,
        classifier: Classifier,
        supported_fruits: Sequence[str] = ("apple", "banana", "orange"),
        aliases: Optional[Dict[str, Tuple[int, ...]]] = None,
        accept_mass: float = 0.30,
        min_side: int = 64,
        blank_std: float = 4.0,
        blur_threshold: float = 8.0,
    ):
        self.classifier = classifier
        self.supported_fruits = tuple(f.lower() for f in supported_fruits)
        self.aliases = dict(aliases or DEFAULT_ALIASES)
        self.accept_mass = accept_mass
        self.min_side = min_side
        self.blank_std = blank_std
        self.blur_threshold = blur_threshold

    def validate(self, source) -> ValidationResult:
        sf = self.supported_fruits
        try:
            img = load_image(source)
        except InvalidImageError as exc:
            return ValidationResult(
                status=REJECTED_INVALID_FILE, reason=str(exc), supported_fruits=sf
            )

        problem = check_quality(img, self.min_side, self.blank_std, self.blur_threshold)
        if problem:
            return ValidationResult(
                status=REJECTED_QUALITY, image=img, reason=problem, supported_fruits=sf
            )

        try:
            probs = self.classifier(img)
            result = decide_from_probabilities(
                probs, sf, self.aliases, self.accept_mass
            )
        except Exception as exc:  # fail closed on ANY validator failure
            return ValidationResult(
                status=ERROR, image=img, reason=str(exc), supported_fruits=sf
            )
        result.image = img
        return result


class UnavailableValidator:
    """Stand-in used when the semantic model cannot be loaded. Never accepts."""

    def __init__(self, reason: str, supported_fruits: Sequence[str] = ("apple", "banana", "orange")):
        self.reason = reason
        self.supported_fruits = tuple(f.lower() for f in supported_fruits)

    def validate(self, source) -> ValidationResult:
        try:
            img = load_image(source)
        except InvalidImageError:
            img = None
        return ValidationResult(
            status=UNAVAILABLE,
            image=img,
            reason=self.reason,
            supported_fruits=self.supported_fruits,
        )


# ---------------------------------------------------------------------------
# Default ImageNet classifier loading (never downloads by itself)
# ---------------------------------------------------------------------------
IMAGENET_WEIGHTS_FILE = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"


def default_imagenet_weights_path() -> Path:
    keras_home = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras"))
    return keras_home / "models" / IMAGENET_WEIGHTS_FILE


def load_imagenet_classifier(weights_path: Optional[os.PathLike] = None) -> Classifier:
    """Load the ImageNet MobileNetV2 *with* classification head from local disk.

    Raises ValidatorUnavailable if TensorFlow or the weights file is missing.
    The project's ``mobilenet_extractor.keras`` has no ImageNet head, so it
    cannot be reused for this purpose.
    """
    path = Path(weights_path) if weights_path else default_imagenet_weights_path()
    if not path.is_file():
        raise ValidatorUnavailable(
            f"ImageNet classifier weights not found at {path}. Run "
            "`python -m scripts.download_imagenet_weights --yes` once (about 14 MB)"
        )
    try:
        import tensorflow as tf  # noqa: WPS433 (deferred heavy import)
    except ImportError as exc:
        raise ValidatorUnavailable("TensorFlow is not installed") from exc
    try:
        model = tf.keras.applications.MobileNetV2(
            weights=str(path), include_top=True, input_shape=(224, 224, 3)
        )
    except Exception as exc:
        raise ValidatorUnavailable(f"could not load ImageNet weights: {exc}") from exc

    def classify(img: Image.Image) -> np.ndarray:
        arr = np.asarray(
            img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32
        )
        arr = arr / 127.5 - 1.0  # MobileNetV2 preprocess_input
        return np.asarray(model(arr[None, ...], training=False))[0]

    return classify


def build_validator(
    supported_fruits: Sequence[str] = ("apple", "banana", "orange"),
    weights_path: Optional[os.PathLike] = None,
    **kwargs,
):
    """Return a real ImageValidator, or an UnavailableValidator (fail closed)."""
    try:
        classifier = load_imagenet_classifier(weights_path)
    except ValidatorUnavailable as exc:
        return UnavailableValidator(str(exc), supported_fruits)
    return ImageValidator(classifier, supported_fruits=supported_fruits, **kwargs)
