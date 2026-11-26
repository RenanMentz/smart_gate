"""Integration test of the pipeline with the REAL YOLO and EasyOCR models.

The images are SYNTHETIC, so this test only checks that the whole chain runs
and stays consistent (every artifact present, record written, decision
matching the OCR output). It does NOT measure recognition accuracy: that is
done on real photos in the evaluation of step 10.
"""

import cv2
import numpy as np
import pytest

import config
from database.database import AccessResult, Database
from modules.detection import PlateDetector
from modules.ocr import PlateOCR
from modules.pipeline import PipelineStatus, RecognitionPipeline
from tests.conftest import make_synthetic_plate

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not config.PLATE_MODEL_PATH.is_file(), reason="run: python -m scripts.download_model"),
]


@pytest.fixture(scope="module")
def models():
    detector, ocr = PlateDetector(), PlateOCR()
    if not detector.yolo_available:
        pytest.skip(detector.load_error)
    if not ocr.available:
        pytest.skip(ocr.load_error)
    return detector, ocr


@pytest.fixture
def pipeline(models, tmp_path):
    detector, ocr = models
    db = Database(tmp_path / "test.db")
    return RecognitionPipeline(db, detector=detector, ocr=ocr, captures_dir=tmp_path / "captures")


def scene_with_plate() -> np.ndarray:
    scene = np.full((480, 720, 3), 110, dtype=np.uint8)
    cv2.rectangle(scene, (120, 180), (600, 440), (60, 60, 70), -1)         # car-like body
    scene[260:390, 160:560] = cv2.resize(make_synthetic_plate("ABC1D23"), (400, 130))
    return scene


def test_pipeline_runs_end_to_end_with_real_models(pipeline):
    result = pipeline.process(scene_with_plate())

    assert result.status in (PipelineStatus.OK, PipelineStatus.NO_PLATE)
    assert result.annotated.shape == result.original.shape
    assert result.decision.result in set(AccessResult)
    assert result.log is not None and result.history_error is None
    assert result.log.result is result.decision.result
    if result.status is PipelineStatus.OK:
        assert result.crop is not None and len(result.steps) >= 8
        assert result.ocr is not None and result.log.raw_text == result.ocr.raw_text
        if result.ocr.plate is None or result.ocr.needs_review:
            assert result.decision.result is AccessResult.MANUAL_REVIEW
    else:
        assert result.decision.result is AccessResult.MANUAL_REVIEW


def test_scene_without_any_plate_goes_to_manual_review(pipeline):
    gradient = np.tile(np.linspace(30, 200, 640, dtype=np.uint8), (480, 1))
    result = pipeline.process(cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR))
    assert result.status is PipelineStatus.NO_PLATE
    assert result.decision.result is AccessResult.MANUAL_REVIEW
    assert result.log.plate is None
