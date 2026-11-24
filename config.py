"""Central configuration for Smart Gate.

All paths, thresholds and tunable parameters live here so that the
processing modules never hard-code values. The Settings screen of the
interface can override the thresholds at runtime (stored in the database).
"""

from __future__ import annotations

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

# Default YOLO weights file for license plate detection (see scripts/download_model.py)
PLATE_MODEL_PATH = MODELS_DIR / "plate_detector.pt"

# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_SIZE_MB = 10
MIN_IMAGE_SIDE_PX = 64          # images smaller than this cannot contain a readable plate
MAX_IMAGE_SIDE_PX = 4096        # larger images are downscaled before processing


@dataclass
class DetectionConfig:
    """Parameters of the plate detector."""

    confidence_threshold: float = 0.25   # minimum YOLO box confidence
    iou_threshold: float = 0.45          # NMS overlap threshold
    crop_padding: float = 0.05           # extra margin (fraction of box size) around the crop
    use_classical_fallback: bool = True  # use OpenCV contour detector if YOLO finds nothing


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
    min_confidence: float = 0.60          # below this the read goes to MANUAL REVIEW
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


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
