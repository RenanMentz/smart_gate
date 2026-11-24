"""Shared test fixtures.

The images produced here are SYNTHETIC (drawn with OpenCV). They exist only
to verify that the image-processing operations behave as expected; they are
never used to report recognition accuracy. Accuracy is measured on real
photos from a public dataset (see tests/real_images).
"""

import cv2
import numpy as np
import pytest


def make_synthetic_plate(text: str = "ABC1D23", width: int = 400, height: int = 130) -> np.ndarray:
    """Draw a plate-like BGR image: dark characters on a light background with a border."""
    plate = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.rectangle(plate, (4, 4), (width - 5, height - 5), (20, 20, 20), 4)
    cv2.rectangle(plate, (4, 4), (width - 5, 28), (160, 60, 20), -1)  # Mercosul-like blue strip
    cv2.putText(plate, text, (22, height - 22), cv2.FONT_HERSHEY_DUPLEX, 2.4, (15, 15, 15), 6, cv2.LINE_AA)
    return plate


def rotate_ccw(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate counter-clockwise by ``degrees`` (simulates a tilted plate)."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


@pytest.fixture
def synthetic_plate() -> np.ndarray:
    return make_synthetic_plate()
