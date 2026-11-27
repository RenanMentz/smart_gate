"""End-to-end recognition pipeline.

One call to :meth:`RecognitionPipeline.process` runs the whole chain::

    image -> detection -> crop -> OpenCV pre-processing -> OCR
          -> access decision -> access history

and returns a :class:`PipelineResult` with everything the interface needs:
original and annotated images, the plate crop, every processing step, the OCR
result, the decision and the record written to the history.

Failure handling
----------------
Each stage catches its own exceptions. A failure never crashes the caller:
the result carries a :class:`PipelineStatus`, a message explaining what
failed, and the access decision falls back to ``MANUAL_REVIEW`` so an operator
can type the plate. Stages that already succeeded keep their output (e.g. an
OCR failure still returns the crop and the processing steps). The history is
the last stage: if it fails, the decision is still returned and
``history_error`` says why.

Settings
--------
Before each image the thresholds stored in the database (Settings screen) are
applied on top of the base configuration with :func:`config.apply_settings`,
so changing a threshold takes effect on the next reading without restarting.
The detector, OCR and history objects handed to the constructor have their
``cfg`` replaced by that effective configuration.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

import config
from config import AppConfig
from database.database import AccessLog, Database
from modules.access_control import AccessController, AccessDecision, ManualReviewOutcome
from modules.detection import DetectionResult, PlateDetector
from modules.history import AccessHistory, RecordOutcome
from modules.image_processing import ProcessingResult, ProcessingStep, preprocess_plate
from modules.ocr import OCRResult, PlateOCR

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """How far the pipeline got. Anything other than ``OK`` forces manual review."""

    OK = "OK"
    NO_PLATE = "NO_PLATE"                    # detection ran but found no plate
    DETECTION_FAILED = "DETECTION_FAILED"    # detector raised an exception
    PREPROCESS_FAILED = "PREPROCESS_FAILED"  # OpenCV chain raised an exception
    OCR_FAILED = "OCR_FAILED"                # OCR unavailable or raised an exception


@dataclass
class PipelineResult:
    """Everything produced by one run of the pipeline."""

    original: np.ndarray
    annotated: np.ndarray                         # original with bounding boxes (== original if none)
    status: PipelineStatus
    decision: AccessDecision
    gate_open_seconds: int                        # how long the simulated gate stays open
    crop: np.ndarray | None = None
    detection: DetectionResult | None = None
    processing: ProcessingResult | None = None
    ocr: OCRResult | None = None
    record: RecordOutcome | None = None
    history_error: str | None = None
    messages: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def steps(self) -> list[ProcessingStep]:
        return self.processing.steps if self.processing else []

    @property
    def log(self) -> AccessLog | None:
        return self.record.log if self.record else None

    @property
    def duplicate(self) -> bool:
        """True if the read was suppressed because the same plate was just logged."""
        return bool(self.record and self.record.duplicate)

    @property
    def detection_method(self) -> str:
        return self.detection.method if self.detection else "none"


class RecognitionPipeline:
    """Runs detection, pre-processing, OCR, access control and history in one call.

    Models are loaded lazily by the detector and the OCR, so building a
    pipeline is cheap. A lock serializes :meth:`process` because one instance is
    shared by every browser session of the Streamlit app.
    """

    def __init__(
        self,
        db: Database,
        *,
        detector: PlateDetector | None = None,
        ocr: PlateOCR | None = None,
        history: AccessHistory | None = None,
        base_config: AppConfig | None = None,
        captures_dir: str | Path | None = None,
        use_db_settings: bool = True,
    ) -> None:
        self.db = db
        self.base_config = base_config or config.DEFAULT_CONFIG
        self.use_db_settings = use_db_settings
        self.config = self.base_config
        self.detector = detector or PlateDetector(cfg=self.base_config.detection)
        self.ocr = ocr or PlateOCR(cfg=self.base_config.ocr)
        self.history = history or AccessHistory(db, captures_dir, self.base_config.duplicate_window_seconds)
        self.controller = AccessController(db, self.history)
        self._lock = threading.Lock()
        self.refresh_config()

    # --------------------------------------------------------------- config
    def refresh_config(self) -> AppConfig:
        """Re-read the database settings and push them to every stage."""
        settings = self.db.get_all_settings() if self.use_db_settings else {}
        self.config = config.apply_settings(self.base_config, settings)
        self.detector.cfg = self.config.detection
        self.ocr.cfg = self.config.ocr
        self.history.duplicate_window_seconds = self.config.duplicate_window_seconds
        return self.config

    # -------------------------------------------------------------- process
    def process(self, image: np.ndarray) -> PipelineResult:
        """Recognize the plate in a BGR image and decide access.

        Raises:
            ValueError: ``image`` is empty (validate uploads with
                :func:`modules.image_io.load_image_from_bytes` first).
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image.")
        with self._lock:
            return self._process(image)

    def _process(self, image: np.ndarray) -> PipelineResult:
        started = time.perf_counter()
        cfg = self.refresh_config()
        messages: list[str] = []
        status = PipelineStatus.OK
        detection = crop = processing = ocr = None
        failure: str | None = None       # why the OCR result is missing, for the decision

        # 1. detection ------------------------------------------------------
        try:
            detection = self.detector.detect(image)
            messages.extend(detection.messages)
            best = detection.best
            if best is None:
                status = PipelineStatus.NO_PLATE
                failure = "No license plate detected."
            else:
                crop = best.crop
                if len(detection.detections) > 1:
                    messages.append(f"{len(detection.detections)} plates detected; "
                                    "reading the most confident one.")
        except Exception as exc:
            logger.exception("Detection failed")
            status = PipelineStatus.DETECTION_FAILED
            failure = f"Plate detection failed: {exc}"

        # 2. pre-processing -------------------------------------------------
        if crop is not None:
            try:
                processing = preprocess_plate(crop, cfg.processing)
            except Exception as exc:
                logger.exception("Pre-processing failed")
                status = PipelineStatus.PREPROCESS_FAILED
                failure = f"Image pre-processing failed: {exc}"

        # 3. OCR --------------------------------------------------------------
        if processing is not None:
            try:
                ocr = self.ocr.recognize({"crop": crop, "enhanced": processing.enhanced,
                                          "binary": processing.binary})
            except Exception as exc:       # includes "no OCR engine available" (missing model files)
                logger.exception("OCR failed")
                status = PipelineStatus.OCR_FAILED
                failure = f"Character recognition failed: {exc}"

        if failure:
            messages.append(failure)

        # 4. access decision (never raises: only reads the database) -------------
        decision = self.controller.evaluate(ocr, error=failure)

        # 5. history ------------------------------------------------------------
        record = None
        history_error = None
        try:
            record = self.history.record(
                decision,
                raw_text=ocr.raw_text if ocr else "",
                confidence=ocr.confidence if ocr else None,
                detection_method=detection.method if detection else "none",
                image=image,
                notes=decision.reason,
            )
        except Exception as exc:
            logger.exception("History write failed")
            history_error = f"The access was not saved to the history: {exc}"
            messages.append(history_error)

        return PipelineResult(
            original=image,
            annotated=detection.annotated if detection and detection.annotated is not None else image,
            status=status,
            decision=decision,
            gate_open_seconds=cfg.gate_open_seconds,
            crop=crop,
            detection=detection,
            processing=processing,
            ocr=ocr,
            record=record,
            history_error=history_error,
            messages=messages,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    # -------------------------------------------------------- manual review
    def review_manually(self, result: PipelineResult, plate_text: str) -> ManualReviewOutcome:
        """Resolve a reading with the plate typed or confirmed by the operator.

        Reuses the image already stored by the automatic attempt and keeps its
        OCR data for traceability.

        Raises:
            ValueError: ``plate_text`` is not a valid Brazilian plate.
        """
        self.refresh_config()
        original_log = result.log
        return self.controller.review_manually(
            plate_text,
            image=result.original,
            image_path=original_log.image_path if original_log else None,
            raw_text=result.ocr.raw_text if result.ocr else "",
            confidence=result.ocr.confidence if result.ocr else None,
            detection_method=result.detection_method,
            original_log=original_log,
        )
