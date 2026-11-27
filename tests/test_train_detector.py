"""Tests for the fine-tuning report (no training: metrics are given)."""

from scripts import train_detector as td


def test_comparison_report_lists_both_models_and_the_caveats():
    before = {s: {"precision": 0.9, "recall": 0.5, "map50": 0.6, "map50_95": 0.3} for s in td.EVAL_SPLITS}
    after = {s: {"precision": 0.92, "recall": 0.8, "map50": 0.85, "map50_95": 0.45} for s in td.EVAL_SPLITS}
    meta = {"date": "d", "images": {"train": 294, "valid": 64, "test": 12}, "epochs_run": 30, "epochs": 50,
            "patience": 15, "imgsz": 640, "batch": 16, "device": "GPU", "minutes": 3.0, "best": "runs/best.pt"}
    report = td.comparison_markdown(before, after, meta)
    assert "| valid | mAP@0.5 | 0.600 | 0.850 | +0.250 |" in report
    assert "| test | Recall | 0.500 | 0.800 | +0.300 |" in report
    assert "contamination" in report and "inflated" in report and "12 images" in report
