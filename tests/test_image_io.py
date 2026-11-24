"""Tests for upload validation and decoding."""

import cv2
import numpy as np
import pytest

import config
from modules.image_io import ImageValidationError, bgr_to_rgb, load_image_from_bytes, load_image_from_path


def encode(image: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, image)
    assert ok
    return buf.tobytes()


def test_valid_png_and_jpg_are_decoded(synthetic_plate):
    for ext, name in ((".png", "car.png"), (".jpg", "car.JPG"), (".jpg", "car.jpeg")):
        img = load_image_from_bytes(encode(synthetic_plate, ext), name)
        assert img.shape == synthetic_plate.shape and img.dtype == np.uint8


@pytest.mark.parametrize("name", ["car.gif", "car.bmp", "car", "car.png.exe"])
def test_unsupported_extension_is_rejected(synthetic_plate, name):
    with pytest.raises(ImageValidationError, match="Unsupported"):
        load_image_from_bytes(encode(synthetic_plate), name)


def test_empty_file_is_rejected():
    with pytest.raises(ImageValidationError, match="empty"):
        load_image_from_bytes(b"", "a.png")


def test_corrupted_file_is_rejected():
    with pytest.raises(ImageValidationError, match="decoded"):
        load_image_from_bytes(b"this is not an image" * 100, "fake.jpg")


def test_too_small_image_is_rejected():
    with pytest.raises(ImageValidationError, match="too small"):
        load_image_from_bytes(encode(np.zeros((20, 20, 3), np.uint8)), "tiny.png")


def test_too_large_file_is_rejected(monkeypatch, synthetic_plate):
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_MB", 0.001)
    with pytest.raises(ImageValidationError, match="too large"):
        load_image_from_bytes(encode(synthetic_plate), "big.png")


def test_grayscale_and_alpha_are_converted_to_bgr(synthetic_plate):
    gray = cv2.cvtColor(synthetic_plate, cv2.COLOR_BGR2GRAY)
    assert load_image_from_bytes(encode(gray), "g.png").shape[2] == 3
    bgra = cv2.cvtColor(synthetic_plate, cv2.COLOR_BGR2BGRA)
    assert load_image_from_bytes(encode(bgra), "a.png").shape[2] == 3


def test_huge_image_is_downscaled(monkeypatch, synthetic_plate):
    monkeypatch.setattr(config, "MAX_IMAGE_SIDE_PX", 200)
    img = load_image_from_bytes(encode(synthetic_plate), "x.png")
    assert max(img.shape[:2]) == 200


def test_load_from_path(tmp_path, synthetic_plate):
    path = tmp_path / "plate.png"
    path.write_bytes(encode(synthetic_plate))
    assert load_image_from_path(path).shape == synthetic_plate.shape
    with pytest.raises(ImageValidationError, match="not found"):
        load_image_from_path(tmp_path / "missing.png")


def test_bgr_to_rgb_swaps_channels():
    pixel = np.array([[[255, 0, 0]]], dtype=np.uint8)  # blue in BGR
    assert bgr_to_rgb(pixel)[0, 0].tolist() == [0, 0, 255]
