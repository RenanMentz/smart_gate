"""Tests for Brazilian plate cleaning/validation."""

import pytest

from modules.plate_format import (
    PlateFormat,
    clean_plate_text,
    detect_format,
    format_for_display,
    is_valid_plate,
    normalize_plate,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("abc-1234", "ABC1234"),
        (" ABC 1D23 ", "ABC1D23"),
        ("abc.1d.23", "ABC1D23"),
        ("", ""),
        (None, ""),
    ],
)
def test_clean_plate_text(raw, expected):
    assert clean_plate_text(raw) == expected


@pytest.mark.parametrize(
    ("plate", "fmt"),
    [
        ("ABC1234", PlateFormat.OLD),
        ("ABC1D23", PlateFormat.MERCOSUL),
        ("AB1234", PlateFormat.INVALID),     # too short
        ("ABC12345", PlateFormat.INVALID),   # too long
        ("1BC1234", PlateFormat.INVALID),    # digit in a letter position
        ("ABC1DD3", PlateFormat.INVALID),    # letter in a digit position
    ],
)
def test_detect_format(plate, fmt):
    assert detect_format(plate) is fmt


def test_normalize_plate_valid_and_invalid():
    assert normalize_plate("bra-2e19") == "BRA2E19"
    assert is_valid_plate("xyz-9876")
    assert not is_valid_plate("hello")
    with pytest.raises(ValueError):
        normalize_plate("ABCD123")


def test_format_for_display():
    assert format_for_display("ABC1234") == "ABC-1234"
    assert format_for_display("ABC1D23") == "ABC1D23"
