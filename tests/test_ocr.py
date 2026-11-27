"""Tests for OCR post-processing (pure logic) and the EasyOCR integration.

The pure-logic tests feed hand-written fragments in the EasyOCR output
format, so every rule (filtering, ordering, correction, review triggers) is
checked deterministically. The slow test runs the real EasyOCR model on a
SYNTHETIC plate only to check the integration works end to end.
"""

import numpy as np
import pytest

from config import OCRConfig
from modules.ocr import (
    PlateOCR,
    choose_reading,
    correct_by_position,
    interpret_fragments,
    order_fragments,
    plate_model_reading,
    position4_alternatives,
)
from modules.plate_format import PlateFormat


def frag(text, x, y, w=100, h=40, conf=0.9):
    """Build an EasyOCR-like fragment with its top-left corner at (x, y)."""
    return ([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], text, conf)


# ------------------------------------------------------- position correction
@pytest.mark.parametrize(
    ("raw", "expected", "n_fixes"),
    [
        ("ABC1234", "ABC1234", 0),
        ("A8C1234", "ABC1234", 1),     # 8 in a letter position -> B
        ("ABCI234", "ABC1234", 1),     # I in a digit position  -> 1
        ("0BC1D2S", "OBC1D25", 2),
        ("ABC1B23", "ABC1B23", 0),     # position 5 (index 4) is never changed
        ("ABC1823", "ABC1823", 0),
    ],
)
def test_correct_by_position(raw, expected, n_fixes):
    corrected, fixes = correct_by_position(raw)
    assert corrected == expected
    assert len(fixes) == n_fixes


def test_correct_by_position_requires_seven_chars():
    with pytest.raises(ValueError):
        correct_by_position("ABC12")


def test_position4_alternatives():
    assert position4_alternatives("ABC1B23") == ["ABC1823"]
    assert position4_alternatives("ABC1823") == ["ABC1B23"]
    assert position4_alternatives("ABC1K23") == []      # K has no look-alike digit
    assert position4_alternatives("ABC") == []


# --------------------------------------------------------- fragment handling
def test_small_text_is_ignored_and_order_is_left_to_right():
    fragments = [
        frag("1D23", 200, 50),
        frag("BRASIL", 120, 0, w=80, h=12),            # small header strip
        frag("ABC", 50, 50),
    ]
    assert [f[1] for f in order_fragments(fragments)] == ["ABC", "1D23"]


def test_two_row_plate_is_read_top_to_bottom():
    fragments = [frag("1D23", 10, 60), frag("ABC", 10, 0)]   # motorcycle layout
    assert [f[1] for f in order_fragments(fragments)] == ["ABC", "1D23"]


def test_interpret_valid_plate_with_hyphen_and_spaces():
    reading = interpret_fragments([frag("abc-12", 0, 0, conf=0.8), frag("34", 120, 0, conf=1.0)], "t")
    assert reading.plate == "ABC1234"
    assert reading.format is PlateFormat.OLD
    # 5 chars at 0.8 + 2 chars at 1.0 -> weighted mean
    assert reading.confidence == pytest.approx((5 * 0.8 + 2 * 1.0) / 7, abs=1e-4)


def test_interpret_too_short_is_invalid():
    reading = interpret_fragments([frag("ABC12", 0, 0)], "t")
    assert reading.plate is None and "Only 5 characters" in reading.notes[0]


def test_interpret_extra_characters_single_valid_window():
    reading = interpret_fragments([frag("ABC1D23I", 0, 0)], "t")
    assert reading.plate == "ABC1D23"


def test_interpret_extra_characters_ambiguous_is_rejected():
    # windows "ABCD123" (-> ABC0123 after D->0) and "BCD1234" are both valid:
    # the system must not pick one on its own
    reading = interpret_fragments([frag("ABCD1234", 0, 0)], "t")
    assert reading.plate is None and "more than one" in reading.notes[0]
    reading = interpret_fragments([frag("AAA1111AAA1111", 0, 0)], "t")
    assert reading.plate is None


def test_interpret_too_many_corrections_is_rejected():
    reading = interpret_fragments([frag("0811234", 0, 0)], "t", OCRConfig(max_corrections=2))
    assert reading.plate is None and len(reading.corrections) == 3


def test_interpret_nothing_read():
    reading = interpret_fragments([], "t")
    assert reading.plate is None and reading.confidence == 0.0


# ------------------------------------------------------------ final decision
def test_choose_high_confidence_valid_reading_is_accepted():
    r1 = interpret_fragments([frag("ABC1D23", 0, 0, conf=0.95)], "binary")
    r2 = interpret_fragments([frag("ABC1D2", 0, 0, conf=0.99)], "enhanced")
    result = choose_reading([r1, r2])
    assert result.plate == "ABC1D23" and result.source == "binary"
    assert result.needs_review is False and result.reasons == []


def test_low_confidence_requires_review_but_keeps_text():
    result = choose_reading([interpret_fragments([frag("ABC1D23", 0, 0, conf=0.4)], "b")])
    assert result.needs_review and "Low OCR confidence" in result.reasons[0]
    assert result.plate == "ABC1D23"   # shown to the operator, not hidden


def test_disagreeing_readings_require_review():
    r1 = interpret_fragments([frag("ABC1D23", 0, 0, conf=0.9)], "binary")
    r2 = interpret_fragments([frag("ABC1D28", 0, 0, conf=0.9)], "enhanced")
    result = choose_reading([r1, r2])
    assert result.needs_review and "different plates" in result.reasons[0]


def test_invalid_text_requires_review_and_no_plate_is_invented():
    result = choose_reading([interpret_fragments([frag("HELLO", 0, 0, conf=0.99)], "b")])
    assert result.needs_review and result.plate is None
    assert result.raw_text == "HELLO"


def test_no_readings():
    assert choose_reading([]).needs_review


# ------------------------------------------------------------ real EasyOCR
@pytest.mark.slow
@pytest.mark.parametrize("text", ["ABC1D23", "XYZ9876"])
def test_easyocr_reads_synthetic_plate(text):
    from modules.image_processing import preprocess_plate
    from tests.conftest import make_synthetic_plate, rotate_ccw

    ocr = PlateOCR()
    if not ocr.available:
        pytest.skip(ocr.load_error)
    processed = preprocess_plate(rotate_ccw(make_synthetic_plate(text), 5))
    result = ocr.recognize({"enhanced": processed.enhanced, "binary": processed.binary})
    assert result.plate == text
    assert 0.0 < result.confidence <= 1.0


# ------------------------------------------------- observed vs guessed chars
def _reading(text, source, conf=0.5):
    box = [[0, 0], [100, 0], [100, 40], [0, 40]]
    return interpret_fragments([(box, text, conf)], source)


def test_directly_read_character_wins_over_a_guessed_one():
    # real case (Mercosul plate ABC1D23): one image read "T" at a digit position,
    # the correction guessed 7; the other image read "1" directly
    guessed, observed = _reading("ABCTD23", "enhanced", 0.486), _reading("ABC1D23", "binary", 0.494)
    assert guessed.plate == "ABC7D23" and observed.plate == "ABC1D23"
    result = choose_reading([guessed, observed], OCRConfig(min_confidence=0.45))
    assert result.plate == "ABC1D23" and not result.needs_review


def test_observed_character_wins_even_with_lower_confidence():
    result = choose_reading([_reading("ABCTD23", "enhanced", 0.9), _reading("ABC1D23", "binary", 0.7)])
    assert result.plate == "ABC1D23" and not result.needs_review


def test_two_different_observations_still_go_to_review():
    result = choose_reading([_reading("ABC1D23", "enhanced", 0.9), _reading("ABC7D23", "binary", 0.9)])
    assert result.needs_review and any("different plates" in r for r in result.reasons)


def test_two_different_guesses_still_go_to_review():
    # "T" -> 7 and "I" -> 1 are both substitutions: nothing was observed at that position
    result = choose_reading([_reading("ABCTD23", "enhanced", 0.9), _reading("ABCID23", "binary", 0.9)])
    assert result.needs_review


# ------------------------------------------------------- plate model engine
def test_plate_model_reading_uses_the_lowest_character_probability():
    reading = plate_model_reading("ABC1D23", [0.99, 0.98, 0.97, 0.62, 0.99, 0.99, 0.99, 1.0, 1.0, 1.0], "Brazil")
    assert reading.plate == "ABC1D23" and reading.confidence == pytest.approx(0.62)
    assert reading.source == "plate_model" and "Brazil" in reading.notes[-1]


def test_plate_model_reading_goes_through_position_correction():
    reading = plate_model_reading("A8C1D23", [1.0] * 10)            # 8 at a letter position
    assert reading.plate == "ABC1D23" and [str(c) for c in reading.corrections] == ["pos 2: 8->B"]
    assert plate_model_reading("", [1.0] * 10).plate is None


class _FakePrediction:
    def __init__(self, plate):
        self.plate, self.char_probs, self.region = plate, [0.99] * 10, "Brazil"


class _FakePlateModel:
    def __init__(self, plate):
        self.plate = plate

    def run(self, image, return_confidence=False):
        assert image.ndim == 3 and image.flags["C_CONTIGUOUS"]
        return [_FakePrediction(self.plate)]


def _ocr_with(plate_model_text, easy_fragments=None):
    ocr = PlateOCR(OCRConfig())
    ocr._plate_model = _FakePlateModel(plate_model_text)
    if easy_fragments is None:
        ocr.load_error = "EasyOCR not installed (test)"
    else:
        ocr._reader = object()
        ocr.read_fragments = lambda image: easy_fragments
    return ocr


IMAGES = {"crop": np.zeros((40, 120, 3), np.uint8), "enhanced": np.zeros((40, 120), np.uint8),
          "binary": np.zeros((40, 120), np.uint8)}


def test_valid_plate_model_read_does_not_call_easyocr():
    ocr = _ocr_with("ABC1D23", easy_fragments=[])
    ocr.read_fragments = lambda image: pytest.fail("EasyOCR should not run")
    result = ocr.recognize(IMAGES)
    assert result.plate == "ABC1D23" and result.source == "plate_model" and not result.needs_review


def test_invalid_plate_model_read_falls_back_to_easyocr():
    box = [[0, 0], [100, 0], [100, 40], [0, 40]]
    ocr = _ocr_with("ABC12", easy_fragments=[(box, "ABC1D23", 0.9)])
    result = ocr.recognize(IMAGES)
    assert result.plate == "ABC1D23" and result.source in {"enhanced", "binary"}
    assert [r.source for r in result.readings] == ["plate_model", "enhanced", "binary"]


def test_no_engine_available_raises():
    ocr = PlateOCR(OCRConfig())
    ocr.plate_model_error, ocr.load_error = "no plate model", "no easyocr"
    with pytest.raises(RuntimeError, match="no plate model"):
        ocr.recognize(IMAGES)


def test_easyocr_engine_ignores_the_plate_model():
    ocr = PlateOCR(OCRConfig(engine="easyocr"))
    box = [[0, 0], [100, 0], [100, 40], [0, 40]]
    ocr._reader = object()
    ocr.read_fragments = lambda image: [(box, "XYZ9876", 0.9)]
    assert ocr.recognize(IMAGES).plate == "XYZ9876" and ocr.engine_name == "EasyOCR"
