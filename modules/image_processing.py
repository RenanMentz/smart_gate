"""OpenCV pre-processing of license plate crops before OCR.

The detector delivers a colour crop of the plate. Real crops suffer from
noise, uneven lighting, low contrast, blur and small rotations, all of which
degrade character recognition. This module applies a classic image
processing chain and records every intermediate image so the interface can
show the effect of each technique.

Pipeline (:func:`preprocess_plate`)
-----------------------------------
1. **Grayscale** ........ ``cv2.cvtColor`` - characters are defined by
   luminance, colour only adds noise for OCR.
2. **Resize** ........... ``cv2.resize`` (bicubic) to a fixed height - OCR
   models work best with characters of ~30-60 px; also makes the kernel
   sizes below scale-independent.
3. **Deskew** ........... ``cv2.Canny`` + ``cv2.HoughLinesP`` estimate the
   dominant near-horizontal edge angle (plate borders / character baseline)
   and ``cv2.warpAffine`` rotates it back to horizontal.
4. **Denoise** .......... ``cv2.bilateralFilter`` - smooths noise while
   preserving edges (it weights neighbours by spatial *and* intensity
   distance, so it does not blur across character strokes).
5. **Equalize** ......... ``cv2.createCLAHE`` (local, contrast-limited
   histogram equalization) or ``cv2.equalizeHist`` (global) - stretches the
   intensity histogram to compensate for poor/uneven illumination.
6. **Sharpen** .......... unsharp mask (``GaussianBlur`` + ``addWeighted``)
   - enhances character edges softened by motion or focus blur.
7. **Threshold** ........ ``cv2.adaptiveThreshold`` (Gaussian local mean)
   or Otsu - binarizes into text/background; adaptive copes with shadows.
8. **Morphology** ....... ``cv2.morphologyEx`` opening (removes isolated
   noise specks) followed by closing (fills small gaps in strokes).

The OCR stage receives both the *enhanced grayscale* image (step 6) and the
*binary* image (step 8) and keeps the read with the higher confidence: deep
learning OCR often prefers grayscale, while binarization helps on
low-contrast or shadowed plates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from config import ProcessingConfig


@dataclass
class ProcessingStep:
    """One stage of the pipeline, kept for visualization."""

    name: str
    description: str
    image: np.ndarray


@dataclass
class ProcessingResult:
    """Output of :func:`preprocess_plate`."""

    enhanced: np.ndarray          # grayscale image after denoise/equalize/sharpen
    binary: np.ndarray            # black characters on white background
    skew_angle: float             # rotation that was corrected (degrees)
    inverted: bool                # True if the plate had light characters on dark background
    steps: list[ProcessingStep] = field(default_factory=list)


# --------------------------------------------------------------------------
# Individual techniques (public so they can be demonstrated/tested on their own)
# --------------------------------------------------------------------------
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR to single-channel grayscale (no-op if already gray)."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Resize keeping the aspect ratio. Bicubic when enlarging, area when shrinking."""
    h, w = image.shape[:2]
    if h == height:
        return image
    scale = height / h
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(image, (max(1, round(w * scale)), height), interpolation=interpolation)


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 20.0) -> float:
    """Estimate the rotation of the plate text in degrees (positive = counter-clockwise).

    Uses the probabilistic Hough transform over Canny edges and returns the
    length-weighted median angle of the near-horizontal segments. Returns
    ``0.0`` when no reliable line is found.
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape[:2]
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=max(20, w // 8),
        minLineLength=max(10, w // 4), maxLineGap=max(5, w // 30),
    )
    if lines is None:
        return 0.0
    angles, weights = [], []
    # OpenCV 4 returns shape (N, 1, 4) and OpenCV 5 returns (N, 4): flatten both
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        # image y grows downward, so negate to get the usual counter-clockwise angle
        angle = float(np.degrees(np.arctan2(-(y2 - y1), x2 - x1)))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        if abs(angle) <= max_angle:
            angles.append(angle)
            weights.append(np.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    cumulative = np.cumsum(np.asarray(weights)[order])
    median_idx = order[np.searchsorted(cumulative, cumulative[-1] / 2)]
    return float(angles[median_idx])


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate around the centre by ``-angle`` to undo a skew of ``angle`` degrees."""
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise(gray: np.ndarray, d: int = 11, sigma: int = 17) -> np.ndarray:
    """Edge-preserving smoothing with a bilateral filter."""
    return cv2.bilateralFilter(gray, d, sigma, sigma)


def equalize(gray: np.ndarray, method: str = "clahe", clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Histogram equalization: ``"clahe"`` (local), ``"global"`` or ``"none"``."""
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        return clahe.apply(gray)
    if method == "global":
        return cv2.equalizeHist(gray)
    if method == "none":
        return gray
    raise ValueError(f"Unknown equalization method '{method}'.")


def sharpen(gray: np.ndarray, amount: float = 1.0, sigma: float = 3.0) -> np.ndarray:
    """Unsharp mask: ``img + amount * (img - blur(img))``."""
    if amount <= 0:
        return gray
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)


def adjust_contrast_brightness(image: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """Linear point operation ``alpha * pixel + beta`` with saturation to [0, 255]."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def has_light_text(gray: np.ndarray) -> bool:
    """True if the characters are lighter than the background.

    After Otsu binarization the characters are the *minority* class, so if
    most pixels are dark the polarity is light-text-on-dark-background.
    """
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(np.mean(otsu == 0)) > 0.5


def threshold(gray: np.ndarray, method: str = "adaptive", block_size: int = 31, c: int = 10) -> np.ndarray:
    """Binarize to black characters (0) on white background (255)."""
    if method == "adaptive":
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
        )
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    raise ValueError(f"Unknown threshold method '{method}'.")


def morphology_clean(binary: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Opening then closing on the characters (foreground = black pixels)."""
    if kernel_size <= 1:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    foreground = cv2.bitwise_not(binary)                 # characters become white
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(foreground)


def compute_histogram(gray: np.ndarray) -> np.ndarray:
    """256-bin intensity histogram (for the visualization of equalization)."""
    return cv2.calcHist([to_grayscale(gray)], [0], None, [256], [0, 256]).ravel()


def normalize_illumination(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Improve a full colour scene under poor lighting (CLAHE on the L channel of LAB).

    Only the lightness is equalized, so colours are not distorted. Used
    before detection as a second attempt on dark images.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def preprocess_plate(crop: np.ndarray, cfg: ProcessingConfig | None = None) -> ProcessingResult:
    """Run the complete pre-processing chain on a plate crop.

    Args:
        crop: BGR (or grayscale) image of the plate region.
        cfg: parameters; defaults to :class:`config.ProcessingConfig`.

    Returns:
        :class:`ProcessingResult` with the two OCR candidates and every
        intermediate step.
    """
    if crop is None or crop.size == 0:
        raise ValueError("Empty plate crop.")
    cfg = cfg or ProcessingConfig()
    steps: list[ProcessingStep] = []

    original = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    steps.append(ProcessingStep("Original", "Plate crop delivered by the detector.", original))

    gray = to_grayscale(crop)
    steps.append(ProcessingStep("Grayscale", "cv2.cvtColor BGR -> GRAY.", gray))

    gray = resize_to_height(gray, cfg.target_height)
    steps.append(ProcessingStep("Resize", f"cv2.resize to {cfg.target_height} px height (bicubic).", gray))

    angle = 0.0
    if cfg.deskew:
        angle = estimate_skew_angle(gray, cfg.max_deskew_angle)
        gray = rotate(gray, angle)
        steps.append(ProcessingStep("Deskew", f"Canny + HoughLinesP: corrected {angle:+.1f} deg.", gray))

    inverted = has_light_text(gray)
    if inverted:
        gray = cv2.bitwise_not(gray)
        steps.append(ProcessingStep("Invert", "Light text on dark background -> inverted.", gray))

    gray = denoise(gray, cfg.bilateral_d, cfg.bilateral_sigma)
    steps.append(ProcessingStep("Denoise", "cv2.bilateralFilter (edge preserving).", gray))

    gray = equalize(gray, cfg.equalization, cfg.clahe_clip_limit, cfg.clahe_tile_grid)
    steps.append(ProcessingStep("Equalize", f"Histogram equalization ({cfg.equalization}).", gray))

    enhanced = sharpen(gray, cfg.sharpen_amount)
    steps.append(ProcessingStep("Sharpen", f"Unsharp mask (amount {cfg.sharpen_amount}).", enhanced))

    binary = threshold(enhanced, cfg.threshold_method, cfg.adaptive_block_size, cfg.adaptive_c)
    steps.append(ProcessingStep("Threshold", f"{cfg.threshold_method} thresholding.", binary))

    binary = morphology_clean(binary, cfg.morph_kernel_size)
    steps.append(ProcessingStep("Morphology", f"Opening + closing ({cfg.morph_kernel_size}x{cfg.morph_kernel_size}).", binary))

    return ProcessingResult(enhanced=enhanced, binary=binary, skew_angle=angle, inverted=inverted, steps=steps)
