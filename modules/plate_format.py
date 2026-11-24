"""Brazilian license plate format helpers.

Two layouts are valid in Brazil:

* Old (pre-2018) layout ........ ``LLL NNNN``  e.g. ``ABC1234``
* Mercosul (2018+) layout ...... ``LLL N L NN`` e.g. ``ABC1D23``

where ``L`` is a letter A-Z and ``N`` is a digit 0-9. Both layouts have
7 characters and share positions 0-2 (letters), 3 (digit) and 5-6 (digits);
they differ only in position 4 (digit in the old layout, letter in Mercosul).

This module only *cleans and validates* text. OCR-specific error correction
(e.g. ``O`` -> ``0``) lives in :mod:`modules.ocr`, because it must take the
recognition confidence into account.
"""

from __future__ import annotations

import re
from enum import Enum

PLATE_LENGTH = 7

OLD_PATTERN = re.compile(r"^[A-Z]{3}[0-9]{4}$")
MERCOSUL_PATTERN = re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$")


class PlateFormat(str, Enum):
    """Possible layouts of a plate string."""

    OLD = "OLD"
    MERCOSUL = "MERCOSUL"
    INVALID = "INVALID"


def clean_plate_text(text: str | None) -> str:
    """Uppercase the text and drop every character that is not A-Z or 0-9.

    Spaces, hyphens and dots that people (or the OCR) put between the
    characters are removed: ``"abc-1d23"`` -> ``"ABC1D23"``.
    """
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def detect_format(plate: str) -> PlateFormat:
    """Return the layout of an already cleaned plate string."""
    if OLD_PATTERN.match(plate):
        return PlateFormat.OLD
    if MERCOSUL_PATTERN.match(plate):
        return PlateFormat.MERCOSUL
    return PlateFormat.INVALID


def is_valid_plate(text: str | None) -> bool:
    """True if ``text`` (after cleaning) is a valid old or Mercosul plate."""
    return detect_format(clean_plate_text(text)) is not PlateFormat.INVALID


def normalize_plate(text: str | None) -> str:
    """Clean ``text`` and ensure it is a valid plate.

    Raises:
        ValueError: if the cleaned text matches neither layout.
    """
    plate = clean_plate_text(text)
    if detect_format(plate) is PlateFormat.INVALID:
        raise ValueError(
            f"Invalid plate '{text}'. Expected ABC1234 (old) or ABC1D23 (Mercosul)."
        )
    return plate


def format_for_display(plate: str) -> str:
    """Return a human friendly representation (``ABC-1234`` / ``ABC1D23``)."""
    if detect_format(plate) is PlateFormat.OLD:
        return f"{plate[:3]}-{plate[3:]}"
    return plate
