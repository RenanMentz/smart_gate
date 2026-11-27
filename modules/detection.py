"""License plate detection.

Two detectors are available:

1. **YOLO11 (deep learning, primary)** - a convolutional neural network that
   predicts bounding boxes and confidences in a single forward pass ("You
   Only Look Once"). The image is resized to ``image_size`` (letterboxed),
   a CSPDarknet-style backbone extracts features at several scales, the neck
   (FPN/PAN) fuses them, and an anchor-free head predicts, for each grid
   cell, a box and a class score. Overlapping boxes are merged with
   Non-Maximum Suppression (``iou_threshold``). The weights were fine-tuned
   for the single class "license plate" (see ``scripts/download_model.py``).

2. **Classical OpenCV detector (fallback)** - used when the YOLO weights are
   missing or YOLO finds nothing. It relies on the geometry of plates:
   bilateral filter -> Canny edges -> morphological closing -> contours ->
   keep rectangles with a readable size and the aspect ratio of a car plate,
   then *verify* each candidate: after Otsu binarization its interior must
   contain 5-10 connected components shaped like characters (similar
   height, narrow, side by side). This verification removes windows, signs
   and other rectangles. Its score is a *heuristic*, not a calibrated
   probability, and the method is reported as ``"classical"`` so results
   are never confused with the neural network's.

If the first YOLO pass finds no plate, a second pass is made on an image
with normalized illumination (CLAHE on the L channel), which helps with
night/backlit pictures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config
from config import DetectionConfig
from modules.image_processing import normalize_illumination

logger = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels, inclusive-exclusive)


@dataclass
class PlateDetection:
    """One detected plate."""

    bbox: BBox
    confidence: float
    method: str              # "yolo" or "classical"
    crop: np.ndarray


@dataclass
class DetectionResult:
    """Output of :meth:`PlateDetector.detect`."""

    detections: list[PlateDetection] = field(default_factory=list)
    annotated: np.ndarray | None = None
    method: str = "none"                 # detector that produced the detections
    illumination_fixed: bool = False     # True if the CLAHE retry was needed
    messages: list[str] = field(default_factory=list)

    @property
    def best(self) -> PlateDetection | None:
        return self.detections[0] if self.detections else None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def crop_with_padding(image: np.ndarray, bbox: BBox, padding: float) -> np.ndarray:
    """Crop ``bbox`` enlarged by ``padding`` (fraction of the box size), clipped to the image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = round((x2 - x1) * padding)
    pad_y = round((y2 - y1) * padding)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    return image[y1:y2, x1:x2].copy()


def draw_detections(image: np.ndarray, detections: list[PlateDetection]) -> np.ndarray:
    """Return a copy of ``image`` with bounding boxes and confidence labels."""
    annotated = image.copy()
    thickness = max(2, round(max(image.shape[:2]) / 400))
    font_scale = max(0.5, max(image.shape[:2]) / 1400)
    for i, det in enumerate(detections):
        color = (0, 200, 0) if i == 0 else (0, 200, 255)  # best plate green, others orange
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.method} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(annotated, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1)
        cv2.putText(annotated, label, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return annotated


def iou(a: BBox, b: BBox) -> float:
    """Intersection over Union of two boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# Classical detector
# --------------------------------------------------------------------------
MIN_PLATE_WIDTH_PX = 60    # below this the characters are too small to be read anyway
MIN_PLATE_HEIGHT_PX = 15


def count_character_blobs(region: np.ndarray) -> int:
    """Count connected components that look like plate characters.

    The region is resized to 60 px height and binarized with Otsu so that
    the characters become the white foreground. A component counts as a
    character if its height is 35-95 % of the region height, it is narrower
    than a quarter of the width and not a thin line.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0
    gray = cv2.resize(gray, (max(1, round(w * 60 / h)), 60), interpolation=cv2.INTER_CUBIC)
    _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(fg == 255) > 0.5:          # light characters on dark background
        fg = cv2.bitwise_not(fg)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    rh, rw = fg.shape
    count = 0
    for i in range(1, n):                 # 0 is the background
        _, _, bw, bh, area = stats[i]
        if 0.35 * rh <= bh <= 0.95 * rh and bw <= 0.25 * rw and bw >= 2 and area >= 0.15 * bw * bh:
            count += 1
    return count


def classical_detect(image: np.ndarray, cfg: DetectionConfig | None = None) -> list[tuple[BBox, float]]:
    """Find plate candidates with edges + contours. Returns ``[(bbox, score), ...]`` best first."""
    cfg = cfg or DetectionConfig()
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)

    # characters are vertical strokes close to each other: a horizontal closing
    # merges them into one blob with the shape of the plate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 60), max(3, h // 120)))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[BBox, float]] = []
    max_area = (w * h) * 0.25
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < MIN_PLATE_WIDTH_PX or ch < MIN_PLATE_HEIGHT_PX or cw * ch > max_area:
            continue
        aspect = cw / ch
        if not (cfg.min_plate_aspect <= aspect <= cfg.max_plate_aspect):
            continue
        blobs = count_character_blobs(image[y:y + ch, x:x + cw])
        if not 5 <= blobs <= 10:          # a plate has 7 characters (some may merge or split)
            continue
        rectangularity = cv2.contourArea(contour) / (cw * ch)          # 1.0 = perfect rectangle
        aspect_score = 1.0 - min(1.0, abs(aspect - 3.1) / 3.1)          # Brazilian plate ~3.1
        blob_score = 1.0 - abs(blobs - 7) / 7
        score = 0.45 * blob_score + 0.30 * aspect_score + 0.25 * rectangularity
        candidates.append(((x, y, x + cw, y + ch), round(float(score), 3)))

    candidates.sort(key=lambda c: c[1], reverse=True)
    kept: list[tuple[BBox, float]] = []
    for bbox, score in candidates:                                       # simple NMS
        if all(iou(bbox, k[0]) < 0.3 for k in kept):
            kept.append((bbox, score))
    return kept[: cfg.max_detections]


# --------------------------------------------------------------------------
# Main detector
# --------------------------------------------------------------------------
class PlateDetector:
    """Locates license plates, YOLO first with OpenCV fallback.

    The YOLO model is loaded lazily on first use so that the interface starts
    fast and the class can be constructed even when the weights are missing.
    """

    def __init__(self, model_path: str | Path | None = None, cfg: DetectionConfig | None = None,
                 device: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else config.PLATE_MODEL_PATH
        self.cfg = cfg or DetectionConfig()
        self.device = device
        self._model = None
        self.load_error: str | None = None

    # ---------------------------------------------------------------- model
    @property
    def loaded(self) -> bool:
        """True once the YOLO weights are in memory (does not trigger loading)."""
        return self._model is not None

    @property
    def yolo_available(self) -> bool:
        """True if the YOLO weights exist and load correctly."""
        return self._load_model() is not None

    def _load_model(self):
        if self._model is not None or self.load_error is not None:
            return self._model
        if not self.model_path.is_file():
            self.load_error = (f"YOLO weights not found at {self.model_path}. "
                               "Run: python -m scripts.download_model")
            logger.warning(self.load_error)
            return None
        try:
            from ultralytics import YOLO

            from modules.device import cuda_usable

            if self.device is None:
                self.device = "0" if cuda_usable() else "cpu"
            self._model = YOLO(str(self.model_path))
        except Exception as exc:  # corrupted file, incompatible version...
            self.load_error = f"Could not load YOLO model: {exc}"
            logger.exception("YOLO load failed")
            return None
        return self._model

    def _yolo_predict(self, image: np.ndarray) -> list[tuple[BBox, float]]:
        model = self._load_model()
        if model is None:
            return []
        results = model.predict(
            image, conf=self.cfg.confidence_threshold, iou=self.cfg.iou_threshold,
            imgsz=self.cfg.image_size, max_det=self.cfg.max_detections,
            device=self.device, verbose=False,
        )
        boxes: list[tuple[BBox, float]] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
            if x2 > x1 and y2 > y1:
                boxes.append(((x1, y1, x2, y2), float(box.conf[0])))
        boxes.sort(key=lambda b: b[1], reverse=True)
        return boxes

    # --------------------------------------------------------------- detect
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect plates in a BGR image.

        Strategy: YOLO -> YOLO on illumination-normalized image -> classical.
        Crops are always taken from the *original* image.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image.")
        result = DetectionResult()
        boxes: list[tuple[BBox, float]] = []

        if self._load_model() is not None:
            boxes = self._yolo_predict(image)
            if boxes:
                result.method = "yolo"
            elif self.cfg.retry_with_illumination_fix:
                boxes = self._yolo_predict(normalize_illumination(image))
                if boxes:
                    result.method = "yolo"
                    result.illumination_fixed = True
                    result.messages.append("Plate found after illumination normalization (CLAHE).")
        else:
            result.messages.append(self.load_error or "YOLO model unavailable.")

        if not boxes and self.cfg.use_classical_fallback:
            boxes = classical_detect(image, self.cfg)
            if boxes:
                result.method = "classical"
                result.messages.append("YOLO found no plate; using the classical OpenCV detector "
                                       "(heuristic score, lower reliability).")

        if not boxes:
            result.messages.append("No license plate detected.")

        result.detections = [
            PlateDetection(bbox=b, confidence=c, method=result.method,
                           crop=crop_with_padding(image, b, self.cfg.crop_padding))
            for b, c in boxes
        ]
        result.annotated = draw_detections(image, result.detections)
        return result
