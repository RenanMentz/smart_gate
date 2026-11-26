"""Tests for the runtime overrides stored in the settings table."""

import pytest

import config
from config import AppConfig, apply_settings


def test_valid_settings_override_defaults_without_touching_the_base():
    base = AppConfig()
    cfg = apply_settings(base, {
        "ocr_min_confidence": "0.75", "yolo_confidence": "0.4",
        "duplicate_window_seconds": "30", "gate_open_seconds": "8",
    })
    assert (cfg.ocr.min_confidence, cfg.detection.confidence_threshold) == (0.75, 0.4)
    assert (cfg.duplicate_window_seconds, cfg.gate_open_seconds) == (30, 8)
    assert base.ocr.min_confidence == 0.60 and base.gate_open_seconds == 5     # base unchanged


def test_empty_settings_keep_the_defaults():
    cfg = apply_settings(AppConfig(), {})
    assert cfg == AppConfig()


@pytest.mark.parametrize(
    ("key", "value"),
    [("ocr_min_confidence", "abc"), ("ocr_min_confidence", "1.5"), ("yolo_confidence", "0"),
     ("duplicate_window_seconds", "-1"), ("gate_open_seconds", "0"), ("gate_open_seconds", "")],
)
def test_invalid_or_out_of_range_values_are_ignored(key, value):
    assert apply_settings(AppConfig(), {key: value}) == AppConfig()


def test_integer_settings_accept_values_stored_as_floats():
    assert apply_settings(AppConfig(), {"gate_open_seconds": "7.0"}).gate_open_seconds == 7


def test_unknown_keys_are_ignored():
    assert apply_settings(AppConfig(), {"something_else": "1"}) == AppConfig()


def test_every_limited_setting_has_a_constant():
    assert set(config.SETTING_LIMITS) == {
        config.SETTING_OCR_MIN_CONFIDENCE, config.SETTING_YOLO_CONFIDENCE,
        config.SETTING_DUPLICATE_WINDOW, config.SETTING_GATE_OPEN_SECONDS,
    }
