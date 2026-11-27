"""Character recognition (OCR) of license plates.

How EasyOCR works
-----------------
EasyOCR runs two neural networks:

1. **CRAFT** (Character Region Awareness for Text) - a fully convolutional
   network that outputs, for every pixel, the probability of being the
   centre of a character and of being the gap between two characters. Text
   boxes are obtained by grouping those regions.
2. **CRNN** recognizer - each box is resized to a fixed height, a CNN
   (ResNet/VGG) extracts a sequence of visual features from left to right,
   a bidirectional LSTM models the context between them, and a **CTC**
   (Connectionist Temporal Classification) decoder turns the per-column
   character probabilities into a string, without needing to segment the
   characters explicitly. The confidence returned is derived from those
   probabilities.

We restrict the output alphabet to ``A-Z0-9`` (``allowlist``) because a
plate never contains other symbols.

From raw text to a plate
------------------------
OCR output is noisy, so :func:`interpret_fragments` applies, in order:

1. **Fragment filtering** - plates carry small text that is not part of the
   registration (the "BRASIL" strip on Mercosul plates, the city/state on
   old plates). Fragments much shorter than the tallest one are dropped.
2. **Reading order** - fragments are grouped into rows (two-row motorcycle
   plates) and sorted left to right.
3. **Cleaning** - uppercase, remove anything that is not A-Z/0-9.
4. **Position-based correction** - both Brazilian layouts require letters at
   positions 0-2 and digits at positions 3, 5 and 6. Only at those
   positions, visually similar characters are swapped (``0``<->``O``,
   ``1``<->``I``, ``8``<->``B``...). Position 4 is a digit in the old layout
   and a letter in Mercosul, so it is **never** changed: the system cannot
   know which one is correct and does not guess.
5. **Validation** - the result must match ``ABC1234`` or ``ABC1D23``.

The read is flagged for **manual review** (never silently "fixed") when the
text is not a valid plate, the confidence is below ``min_confidence``, more
than ``max_corrections`` characters had to be changed, or the grayscale and
binary images produced two *different* valid plates. A disagreement is not a
conflict when it only involves characters *guessed* by the position-based
correction: a character read directly wins over one that was substituted
(:func:`observed_consensus`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config import OCRConfig
from modules.plate_format import PLATE_LENGTH, PlateFormat, clean_plate_text, detect_format

logger = logging.getLogger(__name__)

# Positions that are always letters / always digits in both Brazilian layouts
LETTER_POSITIONS = (0, 1, 2)
DIGIT_POSITIONS = (3, 5, 6)
AMBIGUOUS_POSITION = 4

# Visually confusable characters (OCR mistakes)
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1",
                   "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}

# (bounding box as 4 points, text, confidence) - the format returned by EasyOCR
Fragment = tuple[list[list[float]], str, float]


@dataclass
class Correction:
    """A character changed by the position-based correction."""

    position: int
    original: str
    corrected: str

    def __str__(self) -> str:
        return f"pos {self.position + 1}: {self.original}->{self.corrected}"


@dataclass
class PlateReading:
    """Interpretation of one OCR pass over one image."""

    source: str                       # which image was read ("enhanced", "binary", ...)
    raw_text: str                     # text exactly as returned by the OCR (kept fragments joined)
    plate: str | None                 # valid normalized plate, or None
    format: PlateFormat
    confidence: float                 # 0..1, weighted by number of characters
    corrections: list[Correction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.plate is not None


@dataclass
class OCRResult:
    """Final OCR decision for a plate crop."""

    plate: str | None
    raw_text: str
    confidence: float
    format: PlateFormat
    needs_review: bool
    reasons: list[str]
    corrections: list[Correction]
    source: str
    readings: list[PlateReading] = field(default_factory=list)


# --------------------------------------------------------------------------
# Pure text logic (testable without loading any model)
# --------------------------------------------------------------------------
def _box_metrics(box: list[list[float]]) -> tuple[float, float, float]:
    """Return (height, centre_x, centre_y) of a 4-point box."""
    pts = np.asarray(box, dtype=float)
    return float(pts[:, 1].max() - pts[:, 1].min()), float(pts[:, 0].mean()), float(pts[:, 1].mean())


def order_fragments(fragments: list[Fragment], min_height_ratio: float = 0.55) -> list[Fragment]:
    """Drop small text and sort the remaining fragments in reading order."""
    fragments = [f for f in fragments if clean_plate_text(f[1])]
    if not fragments:
        return []
    metrics = [_box_metrics(f[0]) for f in fragments]
    tallest = max(m[0] for m in metrics)
    kept = [(f, m) for f, m in zip(fragments, metrics) if m[0] >= min_height_ratio * tallest]

    # group into rows: a fragment starts a new row if its centre is below the
    # current row by more than half a character height
    kept.sort(key=lambda fm: fm[1][2])
    rows: list[list[tuple[Fragment, tuple[float, float, float]]]] = []
    for item in kept:
        if rows and abs(item[1][2] - rows[-1][0][1][2]) <= 0.5 * tallest:
            rows[-1].append(item)
        else:
            rows.append([item])
    ordered: list[Fragment] = []
    for row in rows:
        ordered.extend(f for f, _ in sorted(row, key=lambda fm: fm[1][1]))
    return ordered


def correct_by_position(text: str) -> tuple[str, list[Correction]]:
    """Fix confusable characters where the plate layout leaves no doubt.

    ``text`` must already be cleaned and have exactly 7 characters.
    Position 4 (index 4) is never modified.
    """
    if len(text) != PLATE_LENGTH:
        raise ValueError("correct_by_position expects a 7-character string.")
    chars = list(text)
    corrections: list[Correction] = []
    for i in LETTER_POSITIONS:
        if chars[i].isdigit() and chars[i] in DIGIT_TO_LETTER:
            corrections.append(Correction(i, chars[i], DIGIT_TO_LETTER[chars[i]]))
            chars[i] = DIGIT_TO_LETTER[chars[i]]
    for i in DIGIT_POSITIONS:
        if chars[i].isalpha() and chars[i] in LETTER_TO_DIGIT:
            corrections.append(Correction(i, chars[i], LETTER_TO_DIGIT[chars[i]]))
            chars[i] = LETTER_TO_DIGIT[chars[i]]
    return "".join(chars), corrections


def position4_alternatives(plate: str) -> list[str]:
    """Plates that differ from ``plate`` only by a confusable character at position 4.

    Example: ``ABC1B23`` (Mercosul) <-> ``ABC1823`` (old). Used by access
    control to send near-matches to manual review instead of denying them.
    """
    if len(plate) != PLATE_LENGTH:
        return []
    ch = plate[AMBIGUOUS_POSITION]
    swap = LETTER_TO_DIGIT.get(ch) if ch.isalpha() else DIGIT_TO_LETTER.get(ch)
    if not swap:
        return []
    candidate = plate[:AMBIGUOUS_POSITION] + swap + plate[AMBIGUOUS_POSITION + 1:]
    return [candidate] if detect_format(candidate) is not PlateFormat.INVALID else []


def _try_plate(text: str, max_corrections: int) -> tuple[str | None, list[Correction]]:
    corrected, corrections = correct_by_position(text)
    if detect_format(corrected) is PlateFormat.INVALID or len(corrections) > max_corrections:
        return None, corrections
    return corrected, corrections


def interpret_fragments(fragments: list[Fragment], source: str, cfg: OCRConfig | None = None) -> PlateReading:
    """Turn raw EasyOCR fragments into a :class:`PlateReading`."""
    cfg = cfg or OCRConfig()
    ordered = order_fragments(fragments, cfg.min_text_height_ratio)
    raw_text = " ".join(f[1] for f in ordered)
    text = clean_plate_text(raw_text)
    total_chars = sum(len(clean_plate_text(f[1])) for f in ordered)
    confidence = (sum(f[2] * len(clean_plate_text(f[1])) for f in ordered) / total_chars) if total_chars else 0.0

    reading = PlateReading(source=source, raw_text=raw_text, plate=None,
                           format=PlateFormat.INVALID, confidence=round(float(confidence), 4))
    if not text:
        reading.notes.append("No characters recognized.")
        return reading

    if len(text) == PLATE_LENGTH:
        plate, corrections = _try_plate(text, cfg.max_corrections)
        reading.corrections = corrections
        if plate is None:
            reading.notes.append(f"'{text}' is not a valid plate even after position-based correction.")
    elif len(text) > PLATE_LENGTH:
        # extra characters (e.g. part of "BRASIL" or a screw read as "I"):
        # accept only if exactly ONE 7-character window is a valid plate
        valid = {}
        for start in range(len(text) - PLATE_LENGTH + 1):
            plate_w, corr_w = _try_plate(text[start:start + PLATE_LENGTH], cfg.max_corrections)
            if plate_w is not None:
                valid.setdefault(plate_w, corr_w)
        if len(valid) == 1:
            plate, reading.corrections = next(iter(valid.items()))
            reading.notes.append(f"Extra characters ignored: '{text}' -> '{plate}'.")
        else:
            plate = None
            reading.notes.append(f"'{text}' has {len(text)} characters; "
                                 f"{'no' if not valid else 'more than one'} valid plate inside it.")
    else:
        plate = None
        reading.notes.append(f"Only {len(text)} characters recognized ('{text}'); a plate has 7.")

    if plate is not None:
        reading.plate = plate
        reading.format = detect_format(plate)
    return reading


def observed_consensus(readings: list[PlateReading]) -> str | None:
    """Plate on which the readings agree once *guessed* characters are discounted.

    A character changed by :func:`correct_by_position` is a guess (``T`` at a
    digit position may be ``1`` or ``7``); a character the OCR read directly is
    an observation. Where valid readings disagree, the position is resolved
    only if every reading that read it *directly* saw the same character.
    Returns the plate of a reading that matches all observations, or ``None``
    if the observations themselves disagree (a real conflict).
    """
    plates = [r.plate for r in readings if r.plate]
    if not plates:
        return None
    for i in range(PLATE_LENGTH):
        if len({p[i] for p in plates}) == 1:
            continue
        observed = {r.plate[i] for r in readings
                    if r.plate and all(c.position != i for c in r.corrections)}
        if len(observed) != 1:
            return None
    # the readings agree on every directly observed character: pick one that has them all
    for r in readings:
        if r.plate and all(
            len({p[i] for p in plates}) == 1 or all(c.position != i for c in r.corrections)
            for i in range(PLATE_LENGTH)
        ):
            return r.plate
    return None


def choose_reading(readings: list[PlateReading], cfg: OCRConfig | None = None) -> OCRResult:
    """Pick the best reading and decide whether a human must review it."""
    cfg = cfg or OCRConfig()
    if not readings:
        return OCRResult(None, "", 0.0, PlateFormat.INVALID, True, ["OCR produced no reading."], [], "none")

    valid = [r for r in readings if r.is_valid]
    consensus = observed_consensus(valid) if len({r.plate for r in valid}) > 1 else None
    pool = [r for r in valid if r.plate == consensus] if consensus else (valid or readings)
    best = max(pool, key=lambda r: (r.confidence, -len(r.corrections)))
    reasons: list[str] = []

    if not best.is_valid:
        reasons.append("Recognized text is not a valid Brazilian plate.")
        reasons.extend(best.notes)
    else:
        if best.confidence < cfg.min_confidence:
            reasons.append(f"Low OCR confidence ({best.confidence:.0%} < {cfg.min_confidence:.0%}).")
        distinct = {r.plate for r in valid}
        if len(distinct) > 1 and consensus is None:
            reasons.append("Different images produced different plates: " + ", ".join(sorted(distinct)) + ".")

    return OCRResult(
        plate=best.plate,
        raw_text=best.raw_text,
        confidence=best.confidence,
        format=best.format,
        needs_review=bool(reasons),
        reasons=reasons,
        corrections=best.corrections,
        source=best.source,
        readings=readings,
    )


# --------------------------------------------------------------------------
# OCR engines
# --------------------------------------------------------------------------
PLATE_MODEL_SOURCE = "plate_model"


def plate_model_reading(text: str, char_probs, region: str | None = None,
                        cfg: OCRConfig | None = None) -> PlateReading:
    """Turn a fast-plate-ocr prediction into a :class:`PlateReading`.

    The plate confidence is the **lowest** probability among the characters:
    one doubtful character is enough to doubt the plate. The model is often
    over-confident (it can be 100 % sure of an ``I`` where a digit must be), so
    the text still goes through the same position-based correction and
    validation as EasyOCR's output.
    """
    text = text.replace("_", "")
    probs = [float(p) for p in (char_probs if char_probs is not None else [])][: max(1, len(text))]
    confidence = min(probs) if probs and text else 0.0
    box = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]       # the model reads the whole crop
    reading = interpret_fragments([(box, text, confidence)] if text else [], PLATE_MODEL_SOURCE, cfg)
    if region:
        reading.notes.append(f"Plate region predicted by the model: {region}.")
    return reading


class PlateOCR:
    """Reads plate crops.

    Engines (models loaded lazily on first use):

    * **fast-plate-ocr** (default, ``cfg.engine == "fast_plate"``) - a small
      CCT (Compact Convolutional Transformer) trained only on license plates
      of ~65 regions, Brazil included. It reads the *colour crop* in one pass:
      the image is resized to 128 x 64, a convolutional tokenizer turns it
      into patches, transformer layers relate them, and one classification
      head per character slot outputs the character (or padding). No
      separate text detection is needed.
    * **EasyOCR** (fallback) - generic text OCR (CRAFT + CRNN/CTC) run on the
      pre-processed grayscale and binary images. Used when the plate model
      is unavailable or does not return a valid plate.
    """

    def __init__(self, cfg: OCRConfig | None = None) -> None:
        self.cfg = cfg or OCRConfig()
        self._reader = None                   # EasyOCR
        self._plate_model = None              # fast-plate-ocr
        self.load_error: str | None = None    # EasyOCR load error
        self.plate_model_error: str | None = None
        self.gpu: bool = False

    # ---------------------------------------------------------------- loading
    @property
    def uses_plate_model(self) -> bool:
        return self.cfg.engine == "fast_plate"

    def _load_plate_model(self):
        if not self.uses_plate_model:
            return None
        if self._plate_model is not None or self.plate_model_error is not None:
            return self._plate_model
        try:
            from fast_plate_ocr import LicensePlateRecognizer

            self._plate_model = LicensePlateRecognizer(self.cfg.plate_model, device="cpu")
        except Exception as exc:  # package missing, no internet on the first download...
            self.plate_model_error = f"Could not load fast-plate-ocr ({self.cfg.plate_model}): {exc}"
            logger.exception("fast-plate-ocr load failed")
        return self._plate_model

    def _load_reader(self):
        if self._reader is not None or self.load_error is not None:
            return self._reader
        try:
            import easyocr

            from modules.device import cuda_usable

            self.gpu = bool(self.cfg.use_gpu and cuda_usable())
            self._reader = easyocr.Reader(self.cfg.languages, gpu=self.gpu, verbose=False)
        except Exception as exc:  # missing model files, no internet on first run...
            self.load_error = f"Could not load EasyOCR: {exc}"
            logger.exception("EasyOCR load failed")
        return self._reader

    @property
    def engine_name(self) -> str:
        """Engine that reads first (for the interface)."""
        return f"fast-plate-ocr ({self.cfg.plate_model})" if self.uses_plate_model else "EasyOCR"

    @property
    def loaded(self) -> bool:
        """True once the first engine is in memory (does not trigger loading)."""
        return (self._plate_model is not None) if self.uses_plate_model else (self._reader is not None)

    @property
    def available(self) -> bool:
        """At least one engine can read (loads the models)."""
        plate_model = self._load_plate_model()
        return plate_model is not None or self._load_reader() is not None

    # ---------------------------------------------------------------- reading
    def read_plate_model(self, crop_bgr: np.ndarray) -> PlateReading:
        """fast-plate-ocr on the colour plate crop (BGR, as OpenCV loads it)."""
        model = self._load_plate_model()
        if model is None:
            raise RuntimeError(self.plate_model_error or "fast-plate-ocr unavailable.")
        image = crop_bgr[:, :, ::-1] if crop_bgr.ndim == 3 else crop_bgr   # the model expects RGB
        prediction = model.run(np.ascontiguousarray(image), return_confidence=True)[0]
        return plate_model_reading(prediction.plate, getattr(prediction, "char_probs", None),
                                   getattr(prediction, "region", None), self.cfg)

    def read_fragments(self, image: np.ndarray) -> list[Fragment]:
        """Raw EasyOCR output for one image."""
        reader = self._load_reader()
        if reader is None:
            raise RuntimeError(self.load_error or "EasyOCR unavailable.")
        results = reader.readtext(
            image, allowlist=self.cfg.allowlist, detail=1, paragraph=False, decoder="greedy",
        )
        return [(np.asarray(box).tolist(), str(text), float(conf)) for box, text, conf in results]

    def recognize(self, images: dict[str, np.ndarray]) -> OCRResult:
        """Read the plate and return the decision.

        Args:
            images: mapping ``source name -> image``. ``"crop"`` is the colour
                crop (read by the plate model); the others, e.g.
                ``{"enhanced": ..., "binary": ...}``, are read by EasyOCR.

        Raises:
            RuntimeError: no engine could read.
        """
        readings: list[PlateReading] = []
        crop = images.get("crop")
        if self.uses_plate_model and crop is not None and crop.size and self._load_plate_model() is not None:
            readings.append(self.read_plate_model(crop))
            if readings[0].is_valid:
                return choose_reading(readings, self.cfg)

        easy_images = {k: v for k, v in images.items() if k != "crop" and v is not None and v.size}
        if easy_images and self._load_reader() is not None:
            for source, image in easy_images.items():
                readings.append(interpret_fragments(self.read_fragments(image), source, self.cfg))
        if not readings:
            errors = [e for e in (self.plate_model_error, self.load_error) if e]
            raise RuntimeError(" ".join(errors) or "No OCR engine available.")
        return choose_reading(readings, self.cfg)
