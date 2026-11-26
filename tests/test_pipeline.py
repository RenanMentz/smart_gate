"""Tests for the end-to-end pipeline.

Detector and OCR are replaced by stubs so every stage, failure path and
setting can be checked quickly and deterministically. The pre-processing,
access control, history and database are the real ones. Stub plates are
SYNTHETIC and only exercise the plumbing; no recognition accuracy is
measured here. The test with the real models is marked ``slow`` and lives in
``test_pipeline_integration.py``.
"""

import numpy as np
import pytest

import config
from database.database import AccessResult, Database
from modules.detection import DetectionResult, PlateDetection
from modules.pipeline import PipelineStatus, RecognitionPipeline
from tests.conftest import make_ocr_result, make_synthetic_plate

SCENE = np.full((300, 500, 3), 90, dtype=np.uint8)


class StubDetector:
    """Returns the given detections (or raises) instead of running YOLO."""

    def __init__(self, detections=None, raises=None, method="yolo"):
        self.cfg = config.DetectionConfig()
        self.detections = detections if detections is not None else [self.make_detection()]
        self.raises = raises
        self.method = method

    @staticmethod
    def make_detection(confidence=0.9, crop=None):
        crop = make_synthetic_plate() if crop is None else crop
        return PlateDetection(bbox=(10, 10, 110, 50), confidence=confidence, method="yolo", crop=crop)

    def detect(self, image):
        if self.raises:
            raise self.raises
        annotated = image.copy()
        annotated[:5] = 255                      # visible difference from the original
        return DetectionResult(detections=self.detections, annotated=annotated,
                               method=self.method if self.detections else "none",
                               messages=[] if self.detections else ["No license plate detected."])


class StubOCR:
    """Returns a fixed OCRResult (or raises) instead of running EasyOCR."""

    def __init__(self, result=None, raises=None):
        self.cfg = config.OCRConfig()
        self.result = result
        self.raises = raises
        self.calls = []

    def recognize(self, images):
        self.calls.append(sorted(images))
        if self.raises:
            raise self.raises
        return self.result


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


def build(db, tmp_path, detector=None, ocr=None):
    return RecognitionPipeline(
        db, detector=detector or StubDetector(), ocr=ocr or StubOCR(make_ocr_result("ABC1D23")),
        captures_dir=tmp_path / "captures",
    )


# ------------------------------------------------------------------ success
def test_full_run_returns_every_artifact(db, tmp_path):
    vehicle = db.add_vehicle("ABC1D23", owner="Ana")
    ocr = StubOCR(make_ocr_result("ABC1D23", confidence=0.95))
    result = build(db, tmp_path, ocr=ocr).process(SCENE)

    assert result.status is PipelineStatus.OK
    assert result.original is SCENE and result.annotated.shape == SCENE.shape
    assert not np.array_equal(result.annotated, SCENE)
    assert result.crop is not None and result.detection.method == "yolo"
    assert [s.name for s in result.steps][:2] == ["Original", "Grayscale"] and len(result.steps) >= 8
    assert ocr.calls == [["binary", "enhanced"]]                 # both candidates reach the OCR
    assert result.ocr.plate == "ABC1D23"
    assert result.decision.result is AccessResult.AUTHORIZED and result.decision.vehicle.id == vehicle.id
    assert result.gate_open_seconds == config.DEFAULT_CONFIG.gate_open_seconds
    assert result.elapsed_ms > 0 and result.history_error is None

    log = result.log
    assert (log.plate, log.result, log.source) == ("ABC1D23", AccessResult.AUTHORIZED, "AUTO")
    assert (log.confidence, log.detection_method) == (0.95, "yolo")
    assert log.image_path and (tmp_path / "captures").exists()
    assert not result.duplicate


def test_unregistered_plate_is_denied(db, tmp_path):
    result = build(db, tmp_path).process(SCENE)
    assert result.status is PipelineStatus.OK
    assert result.decision.result is AccessResult.DENIED and result.log.result is AccessResult.DENIED


def test_low_confidence_read_goes_to_manual_review(db, tmp_path):
    db.add_vehicle("ABC1D23")
    ocr = StubOCR(make_ocr_result("ABC1D23", needs_review=True, reasons=["Low OCR confidence."]))
    result = build(db, tmp_path, ocr=ocr).process(SCENE)
    assert result.decision.result is AccessResult.MANUAL_REVIEW
    assert result.log.plate == "ABC1D23"            # the doubtful plate is kept for the operator


def test_most_confident_plate_is_read_when_several_are_found(db, tmp_path):
    detector = StubDetector([StubDetector.make_detection(0.9), StubDetector.make_detection(0.5)])
    result = build(db, tmp_path, detector=detector).process(SCENE)
    assert any("2 plates detected" in m for m in result.messages)


def test_second_identical_read_is_reported_as_duplicate(db, tmp_path):
    pipeline = build(db, tmp_path)
    first = pipeline.process(SCENE)
    second = pipeline.process(SCENE)
    assert not first.duplicate and second.duplicate
    assert second.log.id == first.log.id and len(db.list_access_logs()) == 1


# ----------------------------------------------------------------- failures
def test_no_plate_goes_to_manual_review_and_is_logged(db, tmp_path):
    result = build(db, tmp_path, detector=StubDetector(detections=[])).process(SCENE)
    assert result.status is PipelineStatus.NO_PLATE
    assert result.crop is None and result.processing is None and result.ocr is None
    assert result.decision.result is AccessResult.MANUAL_REVIEW
    assert any("No license plate detected" in m for m in result.messages)
    assert result.log.plate is None and result.log.detection_method == "none"


def test_detector_exception_is_contained(db, tmp_path):
    detector = StubDetector(raises=RuntimeError("CUDA out of memory"))
    result = build(db, tmp_path, detector=detector).process(SCENE)
    assert result.status is PipelineStatus.DETECTION_FAILED
    assert result.decision.result is AccessResult.MANUAL_REVIEW
    assert result.annotated is SCENE                      # falls back to the original image
    assert any("CUDA out of memory" in m for m in result.messages)
    assert result.log is not None


def test_preprocessing_exception_is_contained(db, tmp_path):
    detector = StubDetector([StubDetector.make_detection(crop=np.zeros((0, 0, 3), np.uint8))])
    result = build(db, tmp_path, detector=detector).process(SCENE)
    assert result.status is PipelineStatus.PREPROCESS_FAILED
    assert result.processing is None and result.decision.result is AccessResult.MANUAL_REVIEW


def test_ocr_unavailable_keeps_crop_and_steps(db, tmp_path):
    ocr = StubOCR(raises=RuntimeError("Could not load EasyOCR: no internet"))
    result = build(db, tmp_path, ocr=ocr).process(SCENE)
    assert result.status is PipelineStatus.OCR_FAILED
    assert result.crop is not None and len(result.steps) >= 8          # earlier stages are kept
    assert result.ocr is None and result.decision.result is AccessResult.MANUAL_REVIEW
    assert "no internet" in result.decision.reason
    assert result.log.confidence is None


def test_history_failure_does_not_lose_the_decision(db, tmp_path, monkeypatch):
    pipeline = build(db, tmp_path)
    db.add_vehicle("ABC1D23")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline.history, "record", boom)
    result = pipeline.process(SCENE)
    assert result.decision.result is AccessResult.AUTHORIZED
    assert result.record is None and "disk full" in result.history_error


@pytest.mark.parametrize("bad", [None, np.zeros((0, 0, 3), np.uint8)])
def test_empty_image_is_rejected(db, tmp_path, bad):
    with pytest.raises(ValueError):
        build(db, tmp_path).process(bad)


# ----------------------------------------------------------------- settings
def test_database_settings_are_applied_to_every_stage(db, tmp_path):
    detector, ocr = StubDetector(), StubOCR(make_ocr_result("ABC1D23"))
    pipeline = build(db, tmp_path, detector=detector, ocr=ocr)
    db.set_setting(config.SETTING_OCR_MIN_CONFIDENCE, "0.8")
    db.set_setting(config.SETTING_YOLO_CONFIDENCE, "0.4")
    db.set_setting(config.SETTING_DUPLICATE_WINDOW, "0")
    db.set_setting(config.SETTING_GATE_OPEN_SECONDS, "9")

    result = pipeline.process(SCENE)
    assert ocr.cfg.min_confidence == 0.8
    assert detector.cfg.confidence_threshold == 0.4
    assert pipeline.history.duplicate_window_seconds == 0
    assert result.gate_open_seconds == 9
    assert not pipeline.process(SCENE).duplicate                      # window 0 -> always logged


def test_settings_do_not_modify_the_shared_default_config(db, tmp_path):
    pipeline = build(db, tmp_path)
    db.set_setting(config.SETTING_GATE_OPEN_SECONDS, "30")
    pipeline.refresh_config()
    assert config.DEFAULT_CONFIG.gate_open_seconds == 5 and pipeline.config.gate_open_seconds == 30


def test_pipeline_can_ignore_database_settings(db, tmp_path):
    db.set_setting(config.SETTING_GATE_OPEN_SECONDS, "30")
    pipeline = RecognitionPipeline(db, detector=StubDetector(), ocr=StubOCR(make_ocr_result("ABC1D23")),
                                   captures_dir=tmp_path / "c", use_db_settings=False)
    assert pipeline.config.gate_open_seconds == 5


# ------------------------------------------------------------ manual review
def test_manual_review_of_an_unreadable_plate(db, tmp_path):
    db.add_vehicle("ABC1D23")
    pipeline = build(db, tmp_path, detector=StubDetector(detections=[]))
    result = pipeline.process(SCENE)
    assert result.decision.result is AccessResult.MANUAL_REVIEW

    outcome = pipeline.review_manually(result, "abc-1d23")
    assert outcome.decision.result is AccessResult.AUTHORIZED
    log = outcome.record.log
    assert log.source == "MANUAL" and log.image_path == result.log.image_path
    assert len(list((tmp_path / "captures").iterdir())) == 1          # image stored once
    assert [e.source for e in db.list_access_logs()] == ["MANUAL", "AUTO"]


def test_manual_review_rejects_invalid_plate(db, tmp_path):
    pipeline = build(db, tmp_path, detector=StubDetector(detections=[]))
    result = pipeline.process(SCENE)
    with pytest.raises(ValueError):
        pipeline.review_manually(result, "12345")
    assert len(db.list_access_logs()) == 1


def test_manual_review_saves_the_image_when_history_had_failed(db, tmp_path, monkeypatch):
    pipeline = build(db, tmp_path)
    monkeypatch.setattr(pipeline.history, "record", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    result = pipeline.process(SCENE)
    monkeypatch.undo()
    outcome = pipeline.review_manually(result, "ABC1D23")
    assert outcome.record.log.image_path is not None
