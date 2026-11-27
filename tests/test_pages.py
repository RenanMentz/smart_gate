"""Tests for the management screens: dashboard, vehicles, authorizations, history, settings.

Pure helpers are tested directly. Each page runs in Streamlit's ``AppTest``
with the database replaced by a temporary one, and the tests check that every
control really changes the data (no decorative buttons).
"""

from datetime import date, datetime, timedelta

import pytest
from streamlit.testing.v1 import AppTest

import config
from database.database import TIMESTAMP_FORMAT, AccessResult, Database
from interface import authorizations, common, dashboard, history, settings, vehicles
from modules import system_status
from modules.detection import PlateDetector
from modules.history import AccessHistory
from modules.ocr import PlateOCR
from modules.pipeline import RecognitionPipeline
from tests.test_pipeline import SCENE

PAGE_TIMEOUT = 20
PAGE_MODULES = (dashboard, vehicles, authorizations, history, settings)


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "pages.db")


@pytest.fixture
def pipeline(db, tmp_path):
    """Real detector / OCR objects (models are lazy and never loaded here)."""
    return RecognitionPipeline(db, detector=PlateDetector(model_path=tmp_path / "missing.pt"), ocr=PlateOCR(),
                               captures_dir=tmp_path / "captures")


@pytest.fixture(autouse=True)
def use_temp_resources(monkeypatch, db, pipeline):
    for module in (*PAGE_MODULES, common):
        if hasattr(module, "get_database"):
            monkeypatch.setattr(module, "get_database", lambda: db)
        if hasattr(module, "get_pipeline"):
            monkeypatch.setattr(module, "get_pipeline", lambda: pipeline)


def _dashboard_page():
    from interface import dashboard
    dashboard.render()


def _vehicles_page():
    from interface import vehicles
    vehicles.render()


def _authorizations_page():
    from interface import authorizations
    authorizations.render()


def _history_page():
    from interface import history
    history.render()


def _settings_page():
    from interface import settings
    settings.render()


def run_page(script):
    at = AppTest.from_function(script, default_timeout=PAGE_TIMEOUT).run()
    assert not at.exception, at.exception
    return at


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def add_log(db, result, plate="ABC1D23", when=None, image_path=None, source="AUTO"):
    when = when or datetime.now()
    return db.add_access_log(result, plate=plate, raw_text=plate or "", confidence=0.8, detection_method="yolo",
                             source=source, image_path=image_path, timestamp=when.strftime(TIMESTAMP_FORMAT))


# --------------------------------------------------------------- common
def test_logs_frame_uses_portuguese_labels(db):
    add_log(db, AccessResult.MANUAL_REVIEW, plate=None, source="MANUAL")
    add_log(db, AccessResult.AUTHORIZED, plate="ABC1234")
    frame = common.logs_frame(db.list_access_logs())
    assert list(frame["Placa"]) == ["ABC-1234", "—"]
    assert list(frame["Resultado"]) == ["Autorizado", "Revisão manual"]
    assert list(frame["Origem"]) == ["Automática", "Operador"]
    assert frame["Confiança"].iloc[0] == 80.0


def test_thumbnail_of_saved_capture_and_missing_file(db, tmp_path):
    stored = AccessHistory(db, tmp_path / "captures").save_capture(SCENE)
    uri = common.thumbnail_data_uri(stored)
    assert uri.startswith("data:image/jpeg;base64,")
    assert common.thumbnail_data_uri(str(tmp_path / "gone.jpg")) is None
    assert common.thumbnail_data_uri(None) is None
    frame = common.logs_frame([add_log(db, AccessResult.DENIED, image_path=stored)], thumbnails=True)
    assert frame.columns[1] == "Imagem" and frame["Imagem"].iloc[0] == uri


def test_csv_export_is_excel_friendly(db):
    add_log(db, AccessResult.DENIED, image_path="data/captures/x.jpg")
    data = common.csv_bytes(db.list_access_logs())
    assert data.startswith("﻿".encode())                     # BOM: Excel reads the accents
    text = data.decode("utf-8-sig")
    header, row = text.splitlines()[:2]
    assert header.split(";")[:4] == ["ID", "Data e hora", "Placa", "Resultado"]
    assert "Negado" in row and "data/captures/x.jpg" in row and "80,0" in row


# ------------------------------------------------------------- dashboard
def test_activity_frame_counts_per_hour_with_zeros(db):
    today = date(2026, 3, 10)
    add_log(db, AccessResult.AUTHORIZED, when=datetime(2026, 3, 10, 8, 5))
    add_log(db, AccessResult.AUTHORIZED, when=datetime(2026, 3, 10, 8, 50))
    add_log(db, AccessResult.DENIED, when=datetime(2026, 3, 10, 17, 0))
    add_log(db, AccessResult.DENIED, when=datetime(2026, 3, 9, 17, 0))        # yesterday: not in "today"
    frame = dashboard.activity_frame(db.list_access_logs(limit=None), "today", today)
    assert len(frame) == 24
    assert frame.loc[8, "Autorizado"] == 2 and frame.loc[17, "Negado"] == 1
    assert frame.to_numpy().sum() == 3


def test_activity_frame_per_day_covers_the_whole_period(db):
    today = date(2026, 3, 10)
    add_log(db, AccessResult.MANUAL_REVIEW, when=datetime(2026, 3, 4, 9))
    add_log(db, AccessResult.MANUAL_REVIEW, when=datetime(2026, 3, 3, 9))     # 8 days ago: outside 7d
    frame = dashboard.activity_frame(db.list_access_logs(limit=None), "7d", today)
    assert len(frame) == 7 and frame.index[0].date() == date(2026, 3, 4)
    assert frame["Revisão manual"].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert len(dashboard.activity_frame([], "30d", today)) == 30


def test_dashboard_shows_real_counts(db):
    add_log(db, AccessResult.AUTHORIZED)
    add_log(db, AccessResult.DENIED)
    add_log(db, AccessResult.DENIED, when=datetime.now() - timedelta(days=3))
    db.add_vehicle("ABC1D23")
    at = run_page(_dashboard_page)
    metrics = [(m.label, m.value) for m in at.metric]
    assert metrics[:4] == [("Tentativas", "2"), ("Autorizado", "1"), ("Negado", "1"), ("Revisão manual", "0")]
    assert metrics[4:8] == [("Tentativas", "3"), ("Autorizado", "1"), ("Negado", "2"), ("Revisão manual", "0")]
    assert any("Veículos cadastrados: 1" in c.value for c in at.caption)
    assert len(at.dataframe) >= 2                                  # table view + latest attempts


def test_dashboard_without_logs_shows_a_hint():
    at = run_page(_dashboard_page)
    assert any("Nenhuma tentativa" in i.value for i in at.info)


# --------------------------------------------------------------- vehicles
def submit_new_vehicle(at, plate, owner=""):
    widget(at.text_input, "Placa *").set_value(plate)
    widget(at.text_input, "Proprietário").set_value(owner)
    return widget(at.button, "Cadastrar").click().run()


def test_register_vehicle_normalizes_the_plate(db):
    at = submit_new_vehicle(run_page(_vehicles_page), "abc-1d23", "Ana")
    assert not at.exception
    vehicle = db.find_vehicle_by_plate("ABC1D23")
    assert vehicle is not None and vehicle.owner == "Ana" and vehicle.authorized
    assert any("cadastrado" in s.value for s in at.success)


@pytest.mark.parametrize(("plate", "message"), [("12", "Placa inválida"), ("ABC1D23", "já está cadastrada")])
def test_register_rejects_invalid_and_duplicate_plates(db, plate, message):
    db.add_vehicle("ABC1D23")
    at = submit_new_vehicle(run_page(_vehicles_page), plate)
    assert any(message in e.value for e in at.error)
    assert len(db.list_vehicles()) == 1


def test_edit_vehicle(db):
    vehicle = db.add_vehicle("ABC1D23", owner="Ana")
    at = run_page(_vehicles_page)
    edit_plate = [w for w in at.text_input if w.label == "Placa *"][1]   # [0] is the registration form
    edit_plate.set_value("XYZ9876")
    [w for w in at.checkbox if w.label == "Autorizado a entrar"][1].uncheck()   # [0]: registration form
    widget(at.button, "Salvar alterações").click().run()
    updated = db.get_vehicle(vehicle.id)
    assert updated.plate == "XYZ9876" and not updated.authorized and updated.owner == "Ana"


def test_delete_requires_confirmation(db):
    db.add_vehicle("ABC1D23")
    at = run_page(_vehicles_page)
    widget(at.button, "Remover veículo").click().run()
    assert any("Remover o veículo ABC1D23" in w.value for w in at.warning)
    assert len(db.list_vehicles()) == 1                                  # nothing removed yet
    widget(at.button, "Cancelar").click().run()
    assert not at.warning and len(db.list_vehicles()) == 1
    widget(at.button, "Remover veículo").click().run()
    widget(at.button, "Confirmar remoção").click().run()
    assert db.list_vehicles() == []
    assert any("removido" in s.value for s in at.success)


def test_vehicle_search(db):
    db.add_vehicle("ABC1D23", owner="Ana")
    db.add_vehicle("XYZ9876", owner="Bruno")
    at = run_page(_vehicles_page)
    at.text_input(key="sg_vehicle_search").set_value("bru").run()
    assert at.dataframe[0].value["Placa"].tolist() == ["XYZ-9876"]
    at.text_input(key="sg_vehicle_search").set_value("nada").run()
    assert any("Nenhum veículo encontrado" in i.value for i in at.info)


# --------------------------------------------------------- authorizations
def test_set_status_by_plate_cases(db):
    db.add_vehicle("ABC1D23")
    assert authorizations.set_status_by_plate("abc1d23", False, False)[0] == "success"
    assert not db.find_vehicle_by_plate("ABC1D23").authorized
    assert authorizations.set_status_by_plate("ABC1D23", False, False)[0] == "info"     # already blocked
    assert authorizations.set_status_by_plate("ZZZ", True, False)[0] == "error"
    kind, message = authorizations.set_status_by_plate("QQQ1Q11", True, False)
    assert kind == "warning" and "não está cadastrada" in message and db.find_vehicle_by_plate("QQQ1Q11") is None
    assert authorizations.set_status_by_plate("QQQ1Q11", True, True)[0] == "success"
    assert db.find_vehicle_by_plate("QQQ1Q11").authorized


def test_authorization_buttons_change_the_database(db):
    vehicle = db.add_vehicle("ABC1D23")
    at = run_page(_authorizations_page)
    at.button(key=f"sg_auth_toggle_{vehicle.id}").click().run()
    assert not db.get_vehicle(vehicle.id).authorized
    assert any("bloqueada" in s.value for s in at.success)
    at.button(key=f"sg_auth_toggle_{vehicle.id}").click().run()
    assert db.get_vehicle(vehicle.id).authorized


def test_authorize_by_plate_form(db):
    db.add_vehicle("ABC1D23", authorized=False)
    at = run_page(_authorizations_page)
    widget(at.text_input, "Placa").set_value("abc-1d23")
    widget(at.button, "Autorizar").click().run()
    assert db.find_vehicle_by_plate("ABC1D23").authorized


# ---------------------------------------------------------------- history
def test_period_bounds():
    today = date(2026, 3, 10)
    assert history.period_bounds((date(2026, 3, 1), date(2026, 3, 5)), today) == ("2026-03-01", "2026-03-05")
    assert history.period_bounds((date(2026, 3, 1),), today) == ("2026-03-01", "2026-03-01")   # still picking
    assert history.period_bounds(date(2026, 3, 2), today) == ("2026-03-02", "2026-03-02")
    assert history.period_bounds((), today) == (None, None)


def test_history_filters_and_export(db, tmp_path):
    stored = AccessHistory(db, tmp_path / "captures").save_capture(SCENE)
    add_log(db, AccessResult.AUTHORIZED, plate="ABC1D23", image_path=stored)
    add_log(db, AccessResult.DENIED, plate="XYZ9876")
    add_log(db, AccessResult.DENIED, plate="XYZ9876", when=datetime.now() - timedelta(days=60))  # outside 30 days
    at = run_page(_history_page)
    table = at.dataframe[0].value
    assert table["Placa"].tolist() == ["XYZ-9876", "ABC1D23"]
    assert table["Imagem"].iloc[1].startswith("data:image/jpeg")
    assert any("2 registro(s)" in c.value for c in at.caption)

    at.selectbox(key="sg_hist_result").set_value(AccessResult.DENIED).run()
    assert at.dataframe[0].value["Resultado"].tolist() == ["Negado"]
    at.selectbox(key="sg_hist_result").set_value(history.ALL_RESULTS).run()
    at.text_input(key="sg_hist_plate").set_value("abc").run()
    assert at.dataframe[0].value["Placa"].tolist() == ["ABC1D23"]
    assert at.get("download_button")                                   # export of the filtered rows


def test_history_capture_viewer(db, tmp_path):
    stored = AccessHistory(db, tmp_path / "captures").save_capture(SCENE)
    log = add_log(db, AccessResult.AUTHORIZED, image_path=stored)
    add_log(db, AccessResult.DENIED, plate="XYZ9876", image_path=None)
    at = run_page(_history_page)
    assert any("não está disponível" in i.value for i in at.info)      # newest (selected) has no image
    at.selectbox(key="sg_hist_selected").set_value(log.id).run()
    assert at.get("image") and any(f"#{log.id}" in m.value for m in at.markdown)


def test_history_empty():
    at = run_page(_history_page)
    assert any("Nenhum registro" in i.value for i in at.info)


# --------------------------------------------------------------- settings
def test_settings_from_form():
    assert settings.settings_from_form(75, 30, 15, 8) == {
        config.SETTING_OCR_MIN_CONFIDENCE: "0.75", config.SETTING_YOLO_CONFIDENCE: "0.30",
        config.SETTING_DUPLICATE_WINDOW: "15", config.SETTING_GATE_OPEN_SECONDS: "8",
    }


def test_saved_settings_are_used_by_the_pipeline(db, pipeline):
    at = run_page(_settings_page)
    widget(at.slider, "Confiança mínima do OCR (%)").set_value(80)
    widget(at.slider, "Confiança mínima do YOLO (%)").set_value(40)
    widget(at.number_input, "Janela de duplicidade (s)").set_value(30)
    widget(at.number_input, "Tempo da cancela aberta (s)").set_value(12)
    widget(at.button, "Salvar").click().run()
    assert any("Configurações salvas" in s.value for s in at.success)

    cfg = pipeline.refresh_config()                           # what the next reading will use
    assert cfg.ocr.min_confidence == 0.80 and pipeline.ocr.cfg.min_confidence == 0.80
    assert cfg.detection.confidence_threshold == 0.40 and pipeline.detector.cfg.confidence_threshold == 0.40
    assert pipeline.history.duplicate_window_seconds == 30 and cfg.gate_open_seconds == 12
    assert widget(at.slider, "Confiança mínima do OCR (%)").value == 80   # form shows the stored value

    widget(at.button, "Restaurar padrões").click().run()
    assert db.get_all_settings() == {}
    assert pipeline.refresh_config().ocr.min_confidence == config.DEFAULT_CONFIG.ocr.min_confidence


def test_settings_page_shows_model_status_without_loading_models(pipeline):
    at = run_page(_settings_page)
    texts = [e.value for e in (*at.success, *at.info, *at.error)]
    assert any(t.startswith("**GPU:**") for t in texts)
    assert any("Detector YOLO11" in t and "not found" in t for t in at.error.values)
    assert any("Leitor de caracteres" in t and "not loaded yet" in t for t in texts)
    assert widget(at.button, "Carregar modelos agora")
    assert not pipeline.detector.loaded and not pipeline.ocr.loaded


# ----------------------------------------------------------- system status
def test_yolo_status_recognizes_the_published_weights(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    detector = PlateDetector(model_path=weights)
    digest = system_status.sha256_of(weights)
    assert "SHA-256 verified" in system_status.yolo_status(detector, digest).detail
    assert "differ" in system_status.yolo_status(detector, "0" * 64).detail
    missing = system_status.yolo_status(PlateDetector(model_path=tmp_path / "none.pt"))
    assert not missing.loaded and "not found" in missing.detail


def test_gpu_status_never_raises():
    status = system_status.gpu_status()
    assert status.torch_version and status.message
