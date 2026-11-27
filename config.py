"""Central configuration for Smart Gate.

All paths, thresholds and tunable parameters live here so that the
processing modules never hard-code values. The Settings screen of the
interface can override the thresholds at runtime (stored in the database).
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CAPTURES_DIR = DATA_DIR / "captures"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
DATABASE_PATH = DATA_DIR / "smart_gate.db"

# YOLO weights for license plate detection:
# * published: downloaded and hash-checked by scripts/download_model.py (default)
# * finetuned: produced by scripts/train_detector.py --install
# Set the environment variable SMART_GATE_MODEL=finetuned to use the second one.
PUBLISHED_MODEL_PATH = MODELS_DIR / "plate_detector.pt"
FINETUNED_MODEL_PATH = MODELS_DIR / "plate_detector_finetuned.pt"
MODEL_ENV_VAR = "SMART_GATE_MODEL"


def select_model_path(choice: str | None) -> Path:
    """Weights chosen by ``SMART_GATE_MODEL`` (anything but ``finetuned`` -> published)."""
    return FINETUNED_MODEL_PATH if (choice or "").strip().lower() == "finetuned" else PUBLISHED_MODEL_PATH


PLATE_MODEL_PATH = select_model_path(os.environ.get(MODEL_ENV_VAR))

# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_SIZE_MB = 10
MIN_IMAGE_SIDE_PX = 64          # images smaller than this cannot contain a readable plate
MAX_IMAGE_SIDE_PX = 4096        # larger images are downscaled before processing

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
SESSION_READINGS_LIMIT = 30     # readings (with their images) kept in memory per browser session


@dataclass
class DetectionConfig:
    """Parameters of the plate detector."""

    confidence_threshold: float = 0.25   # minimum YOLO box confidence
    iou_threshold: float = 0.45          # NMS overlap threshold
    image_size: int = 640                # YOLO input size (larger helps small/distant plates)
    max_detections: int = 5              # plates kept per image
    crop_padding: float = 0.05           # extra margin (fraction of box size) around the crop
    retry_with_illumination_fix: bool = True  # second YOLO pass on a CLAHE-enhanced image
    use_classical_fallback: bool = True  # use OpenCV contour detector if YOLO finds nothing
    min_plate_aspect: float = 2.0        # classical detector: width/height range of a car plate
    max_plate_aspect: float = 6.0        # (Brazilian car plate is 400x130 mm -> ~3.1)


@dataclass
class ProcessingConfig:
    """Parameters of the OpenCV pre-processing pipeline applied to the plate crop."""

    target_height: int = 120             # plate crop is resized to this height (keeps aspect)
    deskew: bool = True                  # correct small rotations of the plate crop
    max_deskew_angle: float = 20.0       # ignore line angles larger than this (degrees)
    bilateral_d: int = 11                # bilateral filter neighbourhood diameter
    bilateral_sigma: int = 17            # bilateral filter colour/space sigma
    equalization: str = "clahe"          # "clahe" (local), "global" (equalizeHist) or "none"
    clahe_clip_limit: float = 2.0        # CLAHE contrast limit
    clahe_tile_grid: int = 8             # CLAHE tile grid size
    sharpen_amount: float = 1.0          # unsharp-mask strength (0 disables)
    threshold_method: str = "adaptive"   # "adaptive" (local mean) or "otsu" (global)
    adaptive_block_size: int = 31        # adaptive threshold neighbourhood (odd)
    adaptive_c: int = 10                 # constant subtracted from the local mean
    morph_kernel_size: int = 3           # structuring element for opening/closing


@dataclass
class OCRConfig:
    """Parameters of the character recognition stage."""

    languages: list[str] = field(default_factory=lambda: ["en"])  # latin alphabet + digits
    use_gpu: bool = True                  # falls back to CPU automatically if CUDA is missing
    min_confidence: float = 0.45          # below this the read goes to MANUAL REVIEW. Lowered from 0.60:
                                          # EasyOCR (generic, not trained on plates) gives clear Mercosul
                                          # plates ~0.49 because it confuses 1/T. Not yet calibrated on
                                          # annotated plates (see docs/evaluation/README.md).
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    max_corrections: int = 2              # more position-based fixes than this -> MANUAL REVIEW
    min_text_height_ratio: float = 0.55   # ignore text smaller than this fraction of the tallest
                                          # fragment ("BRASIL" strip, city name on old plates)


@dataclass
class AppConfig:
    """Aggregates every configuration block."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    gate_open_seconds: int = 5           # how long the simulated gate stays open
    duplicate_window_seconds: int = 10   # same plate read again within this window is not re-logged


def ensure_directories() -> None:
    """Create runtime directories if they do not exist yet."""
    for directory in (DATA_DIR, CAPTURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = AppConfig()

# ---------------------------------------------------------------------------
# Runtime overrides (``settings`` table, edited on the Settings screen)
# ---------------------------------------------------------------------------
SETTING_OCR_MIN_CONFIDENCE = "ocr_min_confidence"
SETTING_YOLO_CONFIDENCE = "yolo_confidence"
SETTING_DUPLICATE_WINDOW = "duplicate_window_seconds"
SETTING_GATE_OPEN_SECONDS = "gate_open_seconds"

# key -> (type, minimum, maximum); values outside the range are ignored
SETTING_LIMITS: dict[str, tuple[type, float, float]] = {
    SETTING_OCR_MIN_CONFIDENCE: (float, 0.0, 1.0),
    SETTING_YOLO_CONFIDENCE: (float, 0.01, 1.0),
    SETTING_DUPLICATE_WINDOW: (int, 0, 3600),
    SETTING_GATE_OPEN_SECONDS: (int, 1, 60),
}


def _parse_setting(key: str, raw: str) -> float | int | None:
    """Convert a stored string to its type, or ``None`` if invalid / out of range."""
    kind, low, high = SETTING_LIMITS[key]
    try:
        value = kind(float(raw)) if kind is int else float(raw)
    except (TypeError, ValueError):
        return None
    return value if low <= value <= high else None


def apply_settings(base: AppConfig, settings: Mapping[str, str]) -> AppConfig:
    """Return a copy of ``base`` with the values stored in the database applied.

    Unknown keys, unparsable values and out-of-range values are ignored, so a
    corrupted settings row can never break the pipeline: the default is used.
    """
    cfg = copy.deepcopy(base)
    values = {key: _parse_setting(key, settings[key]) for key in SETTING_LIMITS if key in settings}
    if values.get(SETTING_OCR_MIN_CONFIDENCE) is not None:
        cfg.ocr.min_confidence = values[SETTING_OCR_MIN_CONFIDENCE]
    if values.get(SETTING_YOLO_CONFIDENCE) is not None:
        cfg.detection.confidence_threshold = values[SETTING_YOLO_CONFIDENCE]
    if values.get(SETTING_DUPLICATE_WINDOW) is not None:
        cfg.duplicate_window_seconds = values[SETTING_DUPLICATE_WINDOW]
    if values.get(SETTING_GATE_OPEN_SECONDS) is not None:
        cfg.gate_open_seconds = values[SETTING_GATE_OPEN_SECONDS]
    return cfg
