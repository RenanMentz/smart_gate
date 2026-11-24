"""Tests for the OpenCV pre-processing techniques."""

import cv2
import numpy as np
import pytest

from config import ProcessingConfig
from modules import image_processing as ip
from tests.conftest import rotate_ccw


def test_grayscale_and_resize(synthetic_plate):
    gray = ip.to_grayscale(synthetic_plate)
    assert gray.ndim == 2 and gray.shape == synthetic_plate.shape[:2]
    resized = ip.resize_to_height(gray, 60)
    assert resized.shape[0] == 60
    assert resized.shape[1] == pytest.approx(gray.shape[1] * 60 / gray.shape[0], abs=1)


@pytest.mark.parametrize("degrees", [-10.0, -5.0, 6.0, 12.0])
def test_skew_is_estimated_and_corrected(synthetic_plate, degrees):
    gray = ip.to_grayscale(rotate_ccw(synthetic_plate, degrees))
    estimated = ip.estimate_skew_angle(gray)
    assert estimated == pytest.approx(degrees, abs=2.0)
    corrected = ip.rotate(gray, estimated)
    assert abs(ip.estimate_skew_angle(corrected)) < 2.0


def test_skew_of_straight_plate_is_zero(synthetic_plate):
    assert abs(ip.estimate_skew_angle(ip.to_grayscale(synthetic_plate))) < 1.0


def test_skew_of_blank_image_is_zero():
    assert ip.estimate_skew_angle(np.full((100, 300), 200, np.uint8)) == 0.0


def test_denoise_reduces_noise(synthetic_plate):
    gray = ip.to_grayscale(synthetic_plate)
    rng = np.random.default_rng(0)
    noisy = np.clip(gray.astype(int) + rng.normal(0, 25, gray.shape), 0, 255).astype(np.uint8)
    error_before = np.mean(np.abs(noisy.astype(int) - gray))
    error_after = np.mean(np.abs(ip.denoise(noisy).astype(int) - gray))
    assert error_after < error_before


@pytest.mark.parametrize("method", ["clahe", "global"])
def test_equalization_increases_contrast(method):
    low_contrast = np.tile(np.linspace(100, 140, 300, dtype=np.uint8), (100, 1))
    out = ip.equalize(low_contrast, method)
    assert out.std() > low_contrast.std()


def test_equalize_invalid_method():
    with pytest.raises(ValueError):
        ip.equalize(np.zeros((10, 10), np.uint8), "magic")


def test_sharpen_increases_edge_strength(synthetic_plate):
    gray = cv2.GaussianBlur(ip.to_grayscale(synthetic_plate), (0, 0), 2)
    laplacian = lambda img: cv2.Laplacian(img, cv2.CV_64F).var()  # noqa: E731
    assert laplacian(ip.sharpen(gray, 1.5)) > laplacian(gray)
    assert ip.sharpen(gray, 0) is gray


def test_contrast_brightness_saturates():
    img = np.array([[100, 200]], dtype=np.uint8)
    assert ip.adjust_contrast_brightness(img, alpha=2.0, beta=10).tolist() == [[210, 255]]


def test_polarity_detection(synthetic_plate):
    gray = ip.to_grayscale(synthetic_plate)
    assert ip.has_light_text(gray) is False
    assert ip.has_light_text(cv2.bitwise_not(gray)) is True


@pytest.mark.parametrize("method", ["adaptive", "otsu"])
def test_threshold_is_binary_with_dark_text(synthetic_plate, method):
    gray = ip.to_grayscale(synthetic_plate)
    binary = ip.threshold(gray, method)
    assert set(np.unique(binary)) <= {0, 255}
    # background (most pixels) must be white
    assert np.mean(binary == 255) > 0.5


def test_morphology_removes_isolated_specks():
    binary = np.full((100, 100), 255, np.uint8)
    binary[50, 50] = 0                      # isolated noise pixel
    binary[10:40, 10:20] = 0                # a real stroke
    cleaned = ip.morphology_clean(binary, 3)
    assert cleaned[50, 50] == 255           # speck removed
    assert np.mean(cleaned[12:38, 12:18] == 0) == 1.0  # stroke kept


def test_histogram_has_256_bins_and_counts_all_pixels(synthetic_plate):
    hist = ip.compute_histogram(synthetic_plate)
    assert hist.shape == (256,)
    assert hist.sum() == synthetic_plate.shape[0] * synthetic_plate.shape[1]


def test_normalize_illumination_brightens_dark_scene(synthetic_plate):
    dark = (synthetic_plate * 0.25).astype(np.uint8)
    out = ip.normalize_illumination(dark)
    assert out.shape == dark.shape
    assert ip.to_grayscale(out).std() > ip.to_grayscale(dark).std()


def test_full_pipeline_records_all_steps(synthetic_plate):
    result = ip.preprocess_plate(rotate_ccw(synthetic_plate, 7), ProcessingConfig(target_height=100))
    names = [s.name for s in result.steps]
    assert names == [
        "Original", "Grayscale", "Resize", "Deskew",
        "Denoise", "Equalize", "Sharpen", "Threshold", "Morphology",
    ]
    assert result.enhanced.shape[0] == 100 and result.binary.shape[0] == 100
    assert result.skew_angle == pytest.approx(7, abs=2)
    assert result.inverted is False


def test_full_pipeline_handles_inverted_plate(synthetic_plate):
    result = ip.preprocess_plate(cv2.bitwise_not(synthetic_plate))
    assert result.inverted is True
    assert "Invert" in [s.name for s in result.steps]
    assert np.mean(result.binary == 255) > 0.5  # still dark text on light background


def test_full_pipeline_accepts_grayscale_input(synthetic_plate):
    result = ip.preprocess_plate(ip.to_grayscale(synthetic_plate))
    assert result.steps[0].image.ndim == 3


def test_full_pipeline_rejects_empty_crop():
    with pytest.raises(ValueError):
        ip.preprocess_plate(np.zeros((0, 0, 3), np.uint8))
