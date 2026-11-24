"""Image input: validation and decoding of uploaded / captured images.

Every image that enters the system (file upload or webcam snapshot) goes
through :func:`load_image_from_bytes`, which guarantees the rest of the
pipeline always receives a valid 3-channel BGR ``numpy`` array of bounded
size, or a clear :class:`ImageValidationError` explaining what is wrong.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config


class ImageValidationError(ValueError):
    """The provided file is not an acceptable image."""


def validate_extension(filename: str) -> None:
    """Accept only the extensions listed in ``config.ALLOWED_EXTENSIONS``."""
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise ImageValidationError(f"Unsupported file type '{ext or '(none)'}'. Allowed: {allowed}.")


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so that the longest side is at most ``max_side`` (keeps aspect ratio)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def load_image_from_bytes(data: bytes, filename: str | None = None) -> np.ndarray:
    """Validate and decode raw image bytes into a BGR image.

    Checks performed, in order:
    1. extension (when a filename is given),
    2. file size limit,
    3. the bytes really decode as an image (catches renamed/corrupted files),
    4. minimum resolution,
    5. channel normalisation (grayscale / BGRA -> BGR) and max-size downscale.

    Raises:
        ImageValidationError: with a user-readable message.
    """
    if filename is not None:
        validate_extension(filename)
    if not data:
        raise ImageValidationError("The file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        raise ImageValidationError(
            f"File too large ({size_mb:.1f} MB). Maximum is {config.MAX_UPLOAD_SIZE_MB} MB."
        )

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageValidationError("The file could not be decoded as an image (corrupted or not JPG/PNG).")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:  # e.g. 16-bit PNG
        image = cv2.convertScaleAbs(image, alpha=255.0 / max(float(image.max()), 1.0))

    h, w = image.shape[:2]
    if min(h, w) < config.MIN_IMAGE_SIDE_PX:
        raise ImageValidationError(
            f"Image too small ({w}x{h}). Minimum side is {config.MIN_IMAGE_SIDE_PX} px."
        )
    return resize_max_side(image, config.MAX_IMAGE_SIDE_PX)


def load_image_from_path(path: str | Path) -> np.ndarray:
    """Read an image file from disk with the same validation as uploads."""
    path = Path(path)
    if not path.is_file():
        raise ImageValidationError(f"File not found: {path}")
    return load_image_from_bytes(path.read_bytes(), path.name)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV uses BGR; Streamlit/PIL expect RGB."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
