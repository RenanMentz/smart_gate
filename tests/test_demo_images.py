"""Regression test on the real demonstration images (``assets/demo``, CC BY 4.0).

The expected plates are the characters printed on the plates, visible in the
photos. The tests load YOLO and EasyOCR, so they are marked ``slow``. Each
model is pinned explicitly (not taken from ``SMART_GATE_MODEL``) and its
tests are skipped when its weights are not on disk.
"""

import pytest

import config
from database.database import AccessResult, Database
from modules.detection import PlateDetector
from modules.image_io import load_image_from_bytes
from modules.pipeline import PipelineStatus, RecognitionPipeline

DEMO_DIR = config.ASSETS_DIR / "demo"

pytestmark = pytest.mark.slow

published_only = pytest.mark.skipif(not config.PUBLISHED_MODEL_PATH.is_file(), reason="published weights missing")
finetuned_only = pytest.mark.skipif(not config.FINETUNED_MODEL_PATH.is_file(),
                                    reason="fine-tuned weights missing (scripts/train_detector.py --install)")


def make_pipeline(tmp, model_path):
    db = Database(tmp / "demo.db")
    db.add_vehicle("NZM8373")                      # registered and authorized
    return RecognitionPipeline(db, detector=PlateDetector(model_path=model_path), captures_dir=tmp / "captures",
                               use_db_settings=False)


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    return make_pipeline(tmp_path_factory.mktemp("published"), config.PUBLISHED_MODEL_PATH)


@pytest.fixture(scope="module")
def finetuned(tmp_path_factory):
    return make_pipeline(tmp_path_factory.mktemp("finetuned"), config.FINETUNED_MODEL_PATH)


def run(pipeline, name):
    path = DEMO_DIR / name
    return pipeline.process(load_image_from_bytes(path.read_bytes(), path.name))


CLEAR_PLATES = [("01_moto_mercosul.jpg", "ASS2193"), ("02_moto_antiga.jpg", "NZM8373"),
                ("03_moto_antiga.jpg", "JSY8838")]


# ------------------------------------------------------- published weights
@published_only
@pytest.mark.parametrize(("name", "plate"), CLEAR_PLATES)
def test_clear_plates_are_read(published, name, plate):
    result = run(published, name)
    assert result.status is PipelineStatus.OK and result.detection_method == "yolo"
    assert result.ocr.plate == plate and not result.ocr.needs_review


@published_only
def test_registered_plate_opens_the_gate(published):
    assert run(published, "02_moto_antiga.jpg").decision.result is AccessResult.AUTHORIZED


@published_only
def test_doubtful_read_goes_to_manual_review(published):
    result = run(published, "04_onibus_camera_transito.jpg")       # plate BAH-5201, small and blurred
    assert result.detection_method == "yolo"
    assert result.decision.result is AccessResult.MANUAL_REVIEW


@published_only
def test_distant_motorcycle_is_not_detected_and_goes_to_review(published):
    result = run(published, "05_moto_distante.jpg")
    assert result.status is PipelineStatus.NO_PLATE
    assert result.decision.result is AccessResult.MANUAL_REVIEW


# ------------------------------------------------------ fine-tuned weights
@finetuned_only
@pytest.mark.parametrize(("name", "plate"), [*CLEAR_PLATES, ("04_onibus_camera_transito.jpg", "BAH5201")])
def test_finetuned_reads_the_clear_plates_and_the_bus(finetuned, name, plate):
    result = run(finetuned, name)
    assert result.ocr is not None and result.ocr.plate == plate


@finetuned_only
def test_finetuned_detects_the_distant_motorcycle_but_never_authorizes_an_unreadable_plate(finetuned):
    result = run(finetuned, "05_moto_distante.jpg")
    assert result.detection_method == "yolo"
    assert result.decision.result is AccessResult.MANUAL_REVIEW
