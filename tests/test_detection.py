"""Tests for plate detection.

Negative cases use the real photos shipped with Ultralytics (``bus.jpg`` and
``zidane.jpg``), which contain no readable license plate. Positive cases for
the classical detector use a synthetic plate pasted on a noisy background.
Recognition accuracy on real Brazilian plates is evaluated separately on the
public dataset (tests/real_images).
"""

import cv2
import numpy as np
import pytest

import config
from config import DetectionConfig
from modules.detection import (
    PlateDetection,
    PlateDetector,
    classical_detect,
    count_character_blobs,
    crop_with_padding,
    draw_detections,
    iou,
)
from modules.image_io import load_image_from_path
from tests.conftest import make_synthetic_plate

PLATE_BOX = (500, 400, 740, 478)


@pytest.fixture
def synthetic_scene():
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.integers(60, 120, (720, 1280, 3), dtype=np.uint8), (7, 7), 0)
    x1, y1, x2, y2 = PLATE_BOX
    scene[y1:y2, x1:x2] = cv2.resize(make_synthetic_plate(), (x2 - x1, y2 - y1))
    return scene


def real_negative_images():
    from ultralytics.utils import ASSETS

    return [load_image_from_path(ASSETS / name) for name in ("bus.jpg", "zidane.jpg")]


# ------------------------------------------------------------------ helpers
def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_crop_with_padding_is_clipped_to_image():
    img = np.zeros((100, 200, 3), np.uint8)
    assert crop_with_padding(img, (50, 20, 150, 60), 0.1).shape == (48, 120, 3)
    assert crop_with_padding(img, (0, 0, 200, 100), 0.5).shape == (100, 200, 3)


def test_draw_detections_returns_copy():
    img = np.zeros((100, 200, 3), np.uint8)
    det = PlateDetection((10, 10, 80, 40), 0.9, "yolo", img[10:40, 10:80])
    out = draw_detections(img, [det])
    assert out.shape == img.shape and out.any() and not img.any()


def test_count_character_blobs_on_plate_and_blank():
    assert 5 <= count_character_blobs(make_synthetic_plate()) <= 10
    assert count_character_blobs(np.full((50, 150, 3), 200, np.uint8)) == 0


# -------------------------------------------------------- classical detector
def test_classical_detector_finds_synthetic_plate(synthetic_scene):
    found = classical_detect(synthetic_scene)
    assert found, "plate not found"
    assert iou(found[0][0], PLATE_BOX) > 0.7


def test_classical_detector_has_no_false_positives_on_real_photos():
    for image in real_negative_images():
        assert classical_detect(image) == []


def test_classical_detector_on_blank_image():
    assert classical_detect(np.full((480, 640, 3), 128, np.uint8)) == []


# ------------------------------------------------------------- PlateDetector
def test_missing_weights_falls_back_to_classical(tmp_path, synthetic_scene):
    detector = PlateDetector(model_path=tmp_path / "missing.pt")
    assert detector.yolo_available is False
    result = detector.detect(synthetic_scene)
    assert result.method == "classical"
    assert result.best is not None and result.best.method == "classical"
    assert result.best.crop.size > 0
    assert any("not found" in m for m in result.messages)


def test_missing_weights_without_fallback_detects_nothing(tmp_path, synthetic_scene):
    detector = PlateDetector(model_path=tmp_path / "missing.pt",
                             cfg=DetectionConfig(use_classical_fallback=False))
    result = detector.detect(synthetic_scene)
    assert result.detections == [] and result.method == "none"
    assert result.annotated.shape == synthetic_scene.shape


def test_detect_rejects_empty_image(tmp_path):
    with pytest.raises(ValueError):
        PlateDetector(model_path=tmp_path / "x.pt").detect(np.zeros((0, 0, 3), np.uint8))


needs_weights = pytest.mark.skipif(
    not config.PLATE_MODEL_PATH.is_file(), reason="run: python -m scripts.download_model"
)


@pytest.mark.slow
@needs_weights
def test_yolo_loads_and_finds_no_plate_on_photos_without_plates():
    detector = PlateDetector()
    assert detector.yolo_available, detector.load_error
    for image in real_negative_images():
        result = detector.detect(image)
        assert all(d.method != "yolo" for d in result.detections)
