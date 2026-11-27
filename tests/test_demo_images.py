"""Regression test on the real demonstration images (``assets/demo``, CC BY 4.0).

The expected plates are the characters printed on the plates, visible in the
photos. The test loads YOLO and EasyOCR, so it is marked ``slow`` and skipped
when the weights are not installed.
"""

import pytest

import config
from database.database import AccessResult, Database
from modules.image_io import load_image_from_bytes
from modules.pipeline import PipelineStatus, RecognitionPipeline

DEMO_DIR = config.ASSETS_DIR / "demo"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not config.PLATE_MODEL_PATH.is_file(), reason="YOLO weights not downloaded"),
]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("demo")
    db = Database(tmp / "demo.db")
    db.add_vehicle("NZM8373")                      # registered and authorized
    return RecognitionPipeline(db, captures_dir=tmp / "captures", use_db_settings=False)


def run(pipeline, name):
    path = DEMO_DIR / name
    return pipeline.process(load_image_from_bytes(path.read_bytes(), path.name))


@pytest.mark.parametrize(("name", "plate"), [
    ("01_moto_mercosul.jpg", "ASS2193"),
    ("02_moto_antiga.jpg", "NZM8373"),
    ("03_moto_antiga.jpg", "JSY8838"),
])
def test_clear_plates_are_read(pipeline, name, plate):
    result = run(pipeline, name)
    assert result.status is PipelineStatus.OK and result.detection_method == "yolo"
    assert result.ocr.plate == plate and not result.ocr.needs_review


def test_registered_plate_opens_the_gate(pipeline):
    assert run(pipeline, "02_moto_antiga.jpg").decision.result is AccessResult.AUTHORIZED


def test_doubtful_read_goes_to_manual_review(pipeline):
    result = run(pipeline, "04_onibus_camera_transito.jpg")       # plate BAH-5201, small and blurred
    assert result.detection_method == "yolo"
    assert result.decision.result is AccessResult.MANUAL_REVIEW


def test_distant_motorcycle_is_not_detected_and_goes_to_review(pipeline):
    result = run(pipeline, "05_moto_distante.jpg")
    assert result.status is PipelineStatus.NO_PLATE
    assert result.decision.result is AccessResult.MANUAL_REVIEW
