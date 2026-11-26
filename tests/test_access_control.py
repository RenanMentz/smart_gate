"""Tests for the access decision rules, the history and the manual review.

Every test uses a temporary database. The OCR results are built by hand (no
model is loaded), so each rule is checked deterministically.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from database.database import AccessResult, Database
from modules.access_control import AccessController, decide_access
from modules.history import AccessHistory, resolve_image_path
from tests.conftest import make_ocr_result

NOW = datetime(2026, 1, 15, 9, 30, 0)


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def history(db, tmp_path):
    return AccessHistory(db, captures_dir=tmp_path / "captures", duplicate_window_seconds=10)


@pytest.fixture
def controller(db, history):
    return AccessController(db, history)


@pytest.fixture
def image():
    return np.full((80, 120, 3), 127, dtype=np.uint8)


# ------------------------------------------------------------------ decision
def test_authorized_vehicle_is_authorized(db):
    vehicle = db.add_vehicle("ABC1D23", owner="Ana")
    decision = decide_access(db, "ABC1D23")
    assert decision.result is AccessResult.AUTHORIZED
    assert decision.vehicle.id == vehicle.id and decision.opens_gate


def test_blocked_vehicle_is_denied(db):
    db.add_vehicle("ABC1D23", authorized=False)
    decision = decide_access(db, "ABC1D23")
    assert decision.result is AccessResult.DENIED
    assert decision.vehicle is not None and not decision.opens_gate


def test_unknown_plate_is_denied(db):
    db.add_vehicle("XYZ9876")
    decision = decide_access(db, "ABC1234")
    assert decision.result is AccessResult.DENIED and decision.vehicle is None


def test_ocr_review_flag_wins_over_a_registered_plate(db):
    db.add_vehicle("ABC1D23")
    decision = decide_access(db, "ABC1D23", needs_review=True, review_reasons=["Low OCR confidence."])
    assert decision.result is AccessResult.MANUAL_REVIEW
    assert "Low OCR confidence." in decision.reason
    assert not decision.opens_gate


def test_missing_plate_goes_to_manual_review(db):
    decision = decide_access(db, None)
    assert decision.result is AccessResult.MANUAL_REVIEW and decision.plate is None


@pytest.mark.parametrize(
    ("registered", "read"),
    [("ABC1823", "ABC1B23"),    # registered old plate read as Mercosul (8 -> B)
     ("ABC1B23", "ABC1823")],   # registered Mercosul plate read as old (B -> 8)
)
def test_unregistered_plate_with_registered_twin_goes_to_review(db, registered, read):
    db.add_vehicle(registered)
    decision = decide_access(db, read)
    assert decision.result is AccessResult.MANUAL_REVIEW
    assert [v.plate for v in decision.similar_vehicles] == [registered]


def test_twin_exception_does_not_apply_when_the_plate_itself_is_registered(db):
    db.add_vehicle("ABC1B23", authorized=False)
    db.add_vehicle("ABC1823")
    assert decide_access(db, "ABC1B23").result is AccessResult.DENIED


def test_unregistered_plate_without_twin_is_denied(db):
    db.add_vehicle("ABC1823")
    assert decide_access(db, "ABC1C23").result is AccessResult.DENIED   # C has no digit twin


def test_confirmed_decision_ignores_review_rules(db):
    db.add_vehicle("ABC1823")
    decision = decide_access(db, "ABC1B23", needs_review=True, allow_review=False)
    assert decision.result is AccessResult.DENIED


def test_confirmed_decision_requires_a_plate(db):
    with pytest.raises(ValueError):
        decide_access(db, None, allow_review=False)


def test_controller_evaluate_uses_ocr_result(db, controller):
    db.add_vehicle("ABC1D23")
    assert controller.evaluate(make_ocr_result("ABC1D23")).result is AccessResult.AUTHORIZED
    assert controller.evaluate(make_ocr_result("ABC1D23", True, ["x"])).result is AccessResult.MANUAL_REVIEW
    assert controller.evaluate(make_ocr_result(None, True, ["not a plate"])).result is AccessResult.MANUAL_REVIEW


def test_controller_evaluate_without_ocr_goes_to_review_with_the_error(controller):
    decision = controller.evaluate(None, error="EasyOCR unavailable.")
    assert decision.result is AccessResult.MANUAL_REVIEW
    assert "EasyOCR unavailable." in decision.reason


# ------------------------------------------------------------------- history
def test_record_saves_image_and_all_fields(db, history, image):
    vehicle = db.add_vehicle("ABC1D23")
    decision = decide_access(db, "ABC1D23")
    outcome = history.record(decision, raw_text="ABC 1D23", confidence=0.93,
                             detection_method="yolo", image=image, now=NOW)
    log = outcome.log
    assert not outcome.duplicate
    assert (log.plate, log.raw_text, log.result) == ("ABC1D23", "ABC 1D23", AccessResult.AUTHORIZED)
    assert (log.confidence, log.detection_method, log.source) == (0.93, "yolo", "AUTO")
    assert log.vehicle_id == vehicle.id and log.timestamp == "2026-01-15 09:30:00"
    saved = resolve_image_path(log.image_path)
    assert saved is not None and saved.name.startswith("capture_20260115_093000")
    assert saved.suffix == ".jpg" and saved.parent == history.captures_dir.resolve()


def test_record_without_image_leaves_path_empty(db, history):
    outcome = history.record(decide_access(db, None), now=NOW)
    assert outcome.log.image_path is None and outcome.log.plate is None


def test_same_plate_and_result_inside_window_is_not_logged_again(db, history, image):
    db.add_vehicle("ABC1D23")
    decision = decide_access(db, "ABC1D23")
    first = history.record(decision, image=image, now=NOW)
    second = history.record(decision, image=image, now=NOW + timedelta(seconds=10))
    assert second.duplicate and second.log.id == first.log.id
    assert len(db.list_access_logs()) == 1
    assert len(list(history.captures_dir.iterdir())) == 1    # no second image either


def test_same_plate_after_the_window_is_logged_again(db, history):
    decision = decide_access(db, "ABC1234")
    history.record(decision, now=NOW)
    again = history.record(decision, now=NOW + timedelta(seconds=11))
    assert not again.duplicate and len(db.list_access_logs()) == 2


def test_same_plate_with_a_different_result_is_logged(db, history):
    first = history.record(decide_access(db, "ABC1234"), now=NOW)               # DENIED
    db.add_vehicle("ABC1234")
    second = history.record(decide_access(db, "ABC1234"), now=NOW + timedelta(seconds=2))   # AUTHORIZED
    assert not second.duplicate and second.log.id != first.log.id


def test_window_zero_disables_suppression(db, tmp_path):
    history = AccessHistory(db, captures_dir=tmp_path / "c", duplicate_window_seconds=0)
    decision = decide_access(db, "ABC1234")
    history.record(decision, now=NOW)
    assert not history.record(decision, now=NOW).duplicate


def test_unreadable_reads_are_always_logged(db, history):
    decision = decide_access(db, None)
    history.record(decision, now=NOW)
    assert not history.record(decision, now=NOW + timedelta(seconds=1)).duplicate
    assert len(db.list_access_logs()) == 2


def test_failure_to_save_the_image_does_not_lose_the_record(db, history):
    outcome = history.record(decide_access(db, "ABC1234"), image=np.zeros((0, 0, 3), np.uint8), now=NOW)
    assert outcome.log.image_path is None and "could not be saved" in outcome.log.notes


def test_resolve_image_path_handles_missing_values(tmp_path):
    assert resolve_image_path(None) is None
    assert resolve_image_path(str(tmp_path / "gone.jpg")) is None


# ------------------------------------------------------------- manual review
def test_manual_review_authorizes_a_registered_plate(db, controller, image):
    db.add_vehicle("ABC1D23")
    auto = controller.history.record(decide_access(db, None), raw_text="AB?1D23", confidence=0.4,
                                     detection_method="yolo", image=image, now=NOW)
    outcome = controller.review_manually(
        "abc-1d23", image_path=auto.log.image_path, raw_text="AB?1D23", confidence=0.4,
        detection_method="yolo", original_log=auto.log,
    )
    assert outcome.decision.result is AccessResult.AUTHORIZED
    log = outcome.record.log
    assert log.source == "MANUAL" and log.plate == "ABC1D23"
    assert log.image_path == auto.log.image_path           # image is reused, not duplicated
    assert f"#{auto.log.id}" in log.notes
    assert [entry.source for entry in db.list_access_logs()] == ["MANUAL", "AUTO"]


def test_manual_review_never_returns_to_the_review_queue(db, controller):
    db.add_vehicle("ABC1823")
    outcome = controller.review_manually("ABC1B23")      # twin of a registered plate
    assert outcome.decision.result is AccessResult.DENIED
    assert outcome.record.log.detection_method == "manual"


def test_manual_review_rejects_invalid_plate_and_logs_nothing(db, controller):
    with pytest.raises(ValueError):
        controller.review_manually("XX-12")
    assert db.list_access_logs() == []


def test_manual_record_is_not_hidden_by_a_recent_automatic_one(db, controller):
    controller.history.record(decide_access(db, "ABC1234"))                        # AUTO / DENIED, just now
    outcome = controller.review_manually("ABC1234")                              # MANUAL / DENIED
    assert not outcome.record.duplicate
    assert len(db.list_access_logs()) == 2
