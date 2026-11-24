"""Tests for the SQLite persistence layer (each test uses a temporary database)."""

import pytest

from database.database import (
    AccessResult,
    Database,
    DuplicatePlateError,
    VehicleNotFoundError,
)


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


def test_add_and_find_vehicle_normalizes_plate(db):
    v = db.add_vehicle("abc-1d23", model="Onix", color="Prata", owner="Ana")
    assert v.plate == "ABC1D23"
    assert v.authorized is True
    assert v.created_at and v.updated_at
    found = db.find_vehicle_by_plate("ABC 1D23")
    assert found is not None and found.id == v.id


def test_duplicate_plate_is_rejected_even_with_different_formatting(db):
    db.add_vehicle("ABC1234")
    with pytest.raises(DuplicatePlateError):
        db.add_vehicle("abc-1234")


def test_invalid_plate_is_rejected(db):
    with pytest.raises(ValueError):
        db.add_vehicle("XX12")


def test_find_unknown_or_invalid_plate_returns_none(db):
    assert db.find_vehicle_by_plate("ZZZ9999") is None
    assert db.find_vehicle_by_plate("not a plate") is None


def test_update_vehicle_fields(db):
    v = db.add_vehicle("ABC1234", model="Gol")
    updated = db.update_vehicle(v.id, model="Polo", color="Azul")
    assert (updated.model, updated.color, updated.plate) == ("Polo", "Azul", "ABC1234")


def test_update_to_existing_plate_raises(db):
    db.add_vehicle("ABC1234")
    v2 = db.add_vehicle("DEF5678")
    with pytest.raises(DuplicatePlateError):
        db.update_vehicle(v2.id, plate="ABC1234")


def test_set_authorization(db):
    v = db.add_vehicle("ABC1234")
    assert db.set_authorization(v.id, False).authorized is False
    assert db.set_authorization(v.id, True).authorized is True


def test_delete_vehicle_keeps_history(db):
    v = db.add_vehicle("ABC1234")
    log = db.add_access_log(AccessResult.AUTHORIZED, plate=v.plate, vehicle_id=v.id, confidence=0.9)
    db.delete_vehicle(v.id)
    with pytest.raises(VehicleNotFoundError):
        db.get_vehicle(v.id)
    logs = db.list_access_logs()
    assert len(logs) == 1
    assert logs[0].id == log.id
    assert logs[0].plate == "ABC1234"
    assert logs[0].vehicle_id is None


def test_delete_missing_vehicle_raises(db):
    with pytest.raises(VehicleNotFoundError):
        db.delete_vehicle(999)


def test_list_vehicles_search_and_filter(db):
    db.add_vehicle("ABC1234", model="Gol", owner="Carlos")
    db.add_vehicle("XYZ1A23", model="Onix", owner="Ana", authorized=False)
    assert [v.plate for v in db.list_vehicles(search="onix")] == ["XYZ1A23"]
    assert [v.plate for v in db.list_vehicles(search="abc-12")] == ["ABC1234"]
    assert [v.plate for v in db.list_vehicles(authorized=False)] == ["XYZ1A23"]
    assert len(db.list_vehicles()) == 2


def test_access_log_filters_and_ordering(db):
    db.add_access_log(AccessResult.AUTHORIZED, plate="ABC1234", timestamp="2026-01-01 08:00:00")
    db.add_access_log(AccessResult.DENIED, plate="XYZ9999", timestamp="2026-01-02 09:00:00")
    db.add_access_log(AccessResult.MANUAL_REVIEW, plate=None, timestamp="2026-01-03 10:00:00")

    all_logs = db.list_access_logs()
    assert [l.timestamp[:10] for l in all_logs] == ["2026-01-03", "2026-01-02", "2026-01-01"]
    assert len(db.list_access_logs(result=AccessResult.DENIED)) == 1
    assert len(db.list_access_logs(plate="abc")) == 1
    assert len(db.list_access_logs(start="2026-01-02", end="2026-01-02")) == 1
    assert db.last_log_for_plate("ABC1234").result is AccessResult.AUTHORIZED


def test_count_logs_by_result(db):
    db.add_access_log(AccessResult.AUTHORIZED, plate="ABC1234")
    db.add_access_log(AccessResult.AUTHORIZED, plate="ABC1234")
    db.add_access_log(AccessResult.DENIED, plate="XYZ9999")
    counts = db.count_logs_by_result()
    assert counts[AccessResult.AUTHORIZED] == 2
    assert counts[AccessResult.DENIED] == 1
    assert counts[AccessResult.MANUAL_REVIEW] == 0


def test_invalid_result_is_rejected(db):
    with pytest.raises(ValueError):
        db.add_access_log("MAYBE", plate="ABC1234")


def test_settings_roundtrip(db):
    assert db.get_setting("ocr_min_confidence", "0.6") == "0.6"
    db.set_setting("ocr_min_confidence", 0.75)
    db.set_setting("ocr_min_confidence", 0.8)
    assert db.get_setting("ocr_min_confidence") == "0.8"
    assert db.get_all_settings() == {"ocr_min_confidence": "0.8"}
