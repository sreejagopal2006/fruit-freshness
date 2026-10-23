"""Routing: Input -> Validation -> (only if accepted) Freshness prediction.

The freshness predictor is only ever called from ``analyze`` and only after the
validator accepted the image. Upload, camera and dataset-sample inputs all go
through this one function.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Sequence

from PIL import Image

from src.image_validator import (
    ERROR,
    UNAVAILABLE,
    ValidationResult,
)

# Keys of the predictor's result that are safe to show. Edibility and
# "recommendation" strings are deliberately NOT passed through: a visual
# classifier cannot establish that food is safe or unsafe to eat.
_REQUIRED_KEYS = ("predicted_class", "confidence", "class_probabilities")
_OPTIONAL_KEYS = ("freshness_score", "quality_grade", "visual_diagnostics")


def supported_fruits_from_classes(classes: Sequence[str]) -> tuple:
    """['fresh_apple','rotten_banana',...] -> ('apple','banana')."""
    fruits = []
    for name in classes:
        fruit = fruit_of_class(str(name))
        if fruit and fruit not in fruits:
            fruits.append(fruit)
    return tuple(sorted(fruits))


def fruit_of_class(class_name: str) -> str:
    return class_name.rsplit("_", 1)[-1].lower() if "_" in class_name else ""


def sanitize_prediction(raw: Dict[str, Any]) -> Dict[str, Any]:
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"predictor result is missing {missing}")
    clean = {k: raw[k] for k in _REQUIRED_KEYS}
    clean.update({k: raw[k] for k in _OPTIONAL_KEYS if k in raw})
    return clean


@dataclass
class AnalysisOutcome:
    validation: ValidationResult
    prediction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None  # inference error (validation passed)
    elapsed_ms: float = 0.0
    fruit_mismatch: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.prediction is not None

    @property
    def message(self) -> str:
        if not self.validation.accepted:
            return self.validation.user_message
        return self.error or ""


def analyze(source, validator, predictor) -> AnalysisOutcome:
    """Validate, then predict only if the image was accepted."""
    t0 = time.perf_counter()

    if validator is None:
        validation = ValidationResult(status=UNAVAILABLE, reason="no validator configured")
    else:
        try:
            validation = validator.validate(source)
        except Exception as exc:  # fail closed
            validation = ValidationResult(status=ERROR, reason=str(exc))

    if not validation.accepted:
        return AnalysisOutcome(validation, elapsed_ms=_ms(t0))

    try:
        prediction = sanitize_prediction(predictor.predict(validation.image))
    except Exception as exc:
        return AnalysisOutcome(
            validation, error=f"Prediction failed: {exc}", elapsed_ms=_ms(t0)
        )

    mismatch = None
    predicted_fruit = fruit_of_class(str(prediction["predicted_class"]))
    if validation.fruit and predicted_fruit and validation.fruit != predicted_fruit:
        mismatch = (
            f"The fruit check saw {validation.fruit}, but the freshness classifier "
            f"predicted {predicted_fruit}. Treat this result with extra caution."
        )
    return AnalysisOutcome(
        validation, prediction=prediction, elapsed_ms=_ms(t0), fruit_mismatch=mismatch
    )


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Stale-result handling
# ---------------------------------------------------------------------------
def image_fingerprint(source) -> str:
    h = hashlib.sha1()
    if isinstance(source, Image.Image):
        h.update(repr(source.size).encode())
        h.update(source.tobytes())
    elif isinstance(source, (str, Path)):
        p = Path(source)
        h.update(str(p).encode())
        try:
            h.update(p.read_bytes())
        except OSError:
            pass
    elif isinstance(source, (bytes, bytearray, memoryview)):
        h.update(bytes(source))
    elif hasattr(source, "getvalue"):
        h.update(source.getvalue())
    else:
        h.update(repr(source).encode())
    return h.hexdigest()


class ResultStore:
    """Keeps the current validation/outcome in a mapping (st.session_state works).

    Selecting a different image clears the previous outcome immediately, so a
    rejected image can never be shown next to an old prediction.
    """

    _FP, _VAL, _OUT = "_fruit_fp", "_fruit_validation", "_fruit_outcome"

    def __init__(self, state: MutableMapping):
        self.state = state

    @property
    def validation(self) -> Optional[ValidationResult]:
        return self.state.get(self._VAL)

    @property
    def outcome(self) -> Optional[AnalysisOutcome]:
        return self.state.get(self._OUT)

    def clear(self) -> None:
        for key in (self._FP, self._VAL, self._OUT):
            self.state.pop(key, None)

    def clear_outcome(self) -> None:
        self.state.pop(self._OUT, None)

    def select(self, source, validator) -> ValidationResult:
        """Register the currently selected image and validate it (cached per image)."""
        fp = image_fingerprint(source)
        if self.state.get(self._FP) != fp:
            self.clear()
            self.state[self._FP] = fp
            if validator is None:
                result = ValidationResult(status=UNAVAILABLE, reason="no validator configured")
            else:
                try:
                    result = validator.validate(source)
                except Exception as exc:
                    result = ValidationResult(status=ERROR, reason=str(exc))
            self.state[self._VAL] = result
        return self.state[self._VAL]

    def analyze(self, source, validator, predictor) -> AnalysisOutcome:
        outcome = analyze(source, validator, predictor)
        self.state[self._FP] = image_fingerprint(source)
        self.state[self._VAL] = outcome.validation
        self.state[self._OUT] = outcome
        return outcome
