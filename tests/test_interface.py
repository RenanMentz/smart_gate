"""Tests for the reading screen.

The pure helpers are tested directly. The page itself runs inside Streamlit's
``AppTest`` harness with a pipeline whose detector and OCR are stubs (the
models are not loaded), so what is checked is the behaviour of the screen:
what it shows for each verdict, that images are processed only once, that the
review form really decides again, and that the gate runs only when authorized.
"""

from datetime import datetime

import cv2
import pytest
from streamlit.testing.v1 import AppTest

import config
from database.database import AccessResult, Database
from interface import access, styles
from modules.pipeline import RecognitionPipeline
from modules.ocr import interpret_fragments
from tests.conftest import make_ocr_result
from tests.test_pipeline import SCENE, StubDetector, StubOCR

PAGE_TIMEOUT = 20


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "ui.db")


def make_pipeline(db, tmp_path, plate="ABC1D23", detections=None, **ocr_kwargs):
    detector = StubDetector(detections=detections)
    ocr = StubOCR(make_ocr_result(plate, **ocr_kwargs))
    return RecognitionPipeline(db, detector=detector, ocr=ocr, captures_dir=tmp_path / "captures")


@pytest.fixture
def use_pipeline(monkeypatch, db):
    """Make the page use the given pipeline and database instead of the cached real ones."""
    def install(pipeline):
        monkeypatch.setattr(access, "get_pipeline", lambda: pipeline)
        monkeypatch.setattr(access, "get_database", lambda: db)
    return install


def seeded_page(pipeline, number=1, name="foto.jpg"):
    """AppTest of app.py with one reading already processed."""
    result = pipeline.process(SCENE)
    reading = access.Reading(key=f"upload:{number}", number=number, name=name,
                             created=datetime.now(), result=result)
    at = AppTest.from_file(str(config.BASE_DIR / "app.py"), default_timeout=PAGE_TIMEOUT)
    at.session_state[access.STATE_READINGS] = [reading]
    at.session_state[access.STATE_SEEN] = {reading.key}
    at.session_state[access.STATE_ERRORS] = {}
    at.session_state[access.STATE_COUNTER] = number
    at.session_state[access.STATE_SELECTED] = reading.key
    return at, reading


def page_html(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def png_bytes(image) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, file_id, name, data):
        self.file_id, self.name, self._data = file_id, name, data

    def getvalue(self):
        return self._data


# ------------------------------------------------------------- pure helpers
def test_styles_escape_dynamic_text():
    html = styles.verdict_html(AccessResult.DENIED, "<script>alert(1)</script>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<script>" not in styles.chip_html("<b>x</b>")


def test_html_builders_have_no_blank_or_indented_lines():
    # Streamlit's markdown parser would turn those into code blocks
    pieces = [styles.brand_bar_html(), styles.verdict_html(AccessResult.AUTHORIZED, "ok"),
              styles.plate_html("ABC1D23", "legenda"), styles.plate_html(None),
              styles.gate_html(True, AccessResult.AUTHORIZED, 5, 1), styles.gate_html(False, AccessResult.DENIED, 5)]
    for html in pieces:
        assert "\n" not in html


@pytest.mark.parametrize(("plate", "shown", "css"), [("ABC1D23", "ABC1D23", "sg-plate-old"),
                                                       ("ABC1234", "ABC-1234", "sg-plate-old")])
def test_plate_html_shows_the_display_format(plate, shown, css):
    html = styles.plate_html(plate)
    assert shown in html
    assert (css in html) == (plate == "ABC1234")      # only the old layout is drawn grey


def test_plate_html_placeholder_when_no_plate():
    assert "Sem leitura" in styles.plate_html(None)


def test_gate_animation_lasts_gate_open_seconds_and_restarts_per_run():
    first = styles.gate_html(True, AccessResult.AUTHORIZED, 7, run_id=1)
    second = styles.gate_html(True, AccessResult.AUTHORIZED, 7, run_id=2)
    assert "sg-arm-1 7s" in first and "sg-arm-2 7s" in second and first != second
    assert "animation" not in styles.gate_html(False, AccessResult.AUTHORIZED, 7)     # closed: static


def test_describe_decision_texts(db, tmp_path):
    db.add_vehicle("ABC1D23")
    db.add_vehicle("XYZ9876", authorized=False)
    db.add_vehicle("ABC1823")

    def describe(plate, **kwargs):
        pipeline = make_pipeline(db, tmp_path, plate, **kwargs)
        result = pipeline.process(SCENE)
        return result, access.describe_decision(result, result.decision)

    assert describe("ABC1D23")[1] == "Veículo cadastrado e autorizado."
    assert describe("XYZ9876")[1] == "Veículo cadastrado, mas bloqueado."
    assert describe("QQQ1Q11")[1] == "Placa não cadastrada."
    assert "parecida (ABC-1823)" in describe("ABC1B23")[1]
    assert "está cadastrada (autorizada)" in describe("ABC1D23", needs_review=True, reasons=["x"])[1]
    assert "Confira a placa" in describe("QQQ1Q11", needs_review=True, reasons=["x"])[1]
    result, text = describe("ABC1D23", detections=None)
    manual = access.describe_decision(result, result.decision, manual=True)
    assert manual.startswith("Placa confirmada pelo operador.")


def test_describe_decision_for_pipeline_failures(db, tmp_path):
    pipeline = RecognitionPipeline(db, detector=StubDetector(detections=[]), ocr=StubOCR(),
                                   captures_dir=tmp_path / "c")
    result = pipeline.process(SCENE)
    assert "Nenhuma placa foi detectada" in access.describe_decision(result, result.decision)
    assert access.histogram_frame(result) is None and access.readings_table(result).empty


def test_histogram_frame_compares_before_and_after_equalization(db, tmp_path):
    result = make_pipeline(db, tmp_path).process(SCENE)
    frame = access.histogram_frame(result)
    assert list(frame.columns) == ["Antes da equalização", "Depois da equalização"]
    assert len(frame) == 256 and frame.sum().nunique() == 1        # same number of pixels in both


def test_readings_table_lists_every_ocr_reading(db, tmp_path):
    result = make_pipeline(db, tmp_path).process(SCENE)
    box = [[0, 0], [100, 0], [100, 40], [0, 40]]
    result.ocr.readings = [interpret_fragments([(box, "ABC1D23", 0.91)], "enhanced"),
                           interpret_fragments([(box, "ABC1D2", 0.50)], "binary")]
    table = access.readings_table(result)
    assert list(table["Imagem lida"]) == ["Tons de cinza melhorado", "Binarizada"]
    assert list(table["Placa interpretada"]) == ["ABC1D23", "inválida"]
    assert list(table["Confiança"]) == ["91%", "50%"]


# ------------------------------------------------------------------ the page
def test_empty_page_renders_with_a_hint_and_no_exception(use_pipeline, db, tmp_path):
    use_pipeline(make_pipeline(db, tmp_path))
    at = AppTest.from_file(str(config.BASE_DIR / "app.py"), default_timeout=PAGE_TIMEOUT).run()
    assert not at.exception
    assert "Leitura de placas" in [h.value for h in at.header]
    assert any("primeira placa" in i.value for i in at.info)
    assert any("Nenhum veículo está cadastrado" in w.value for w in at.warning)


def test_authorized_reading_shows_green_verdict_and_runs_the_gate(use_pipeline, db, tmp_path):
    db.add_vehicle("ABC1D23", owner="Ana Souza", model="Onix", color="Prata")
    pipeline = make_pipeline(db, tmp_path)
    use_pipeline(pipeline)
    at, reading = seeded_page(pipeline)
    at.run()
    assert not at.exception
    html = page_html(at)
    assert "sg-verdict-ok" in html and "Acesso autorizado" in html
    assert "ABC1D23" in html
    assert "Cancela aberta" in html and f"sg-arm-{reading.number} {config.DEFAULT_CONFIG.gate_open_seconds}s" in html
    assert any("Ana Souza" in m.value for m in at.markdown)           # database lookup is shown
    assert len(at.tabs) >= 4 and not at.get("form")                    # no review form when authorized


def test_gate_stays_closed_for_denied_access(use_pipeline, db, tmp_path):
    pipeline = make_pipeline(db, tmp_path)
    use_pipeline(pipeline)
    at, _ = seeded_page(pipeline)
    at.run()
    html = page_html(at)
    assert "sg-verdict-denied" in html and "Acesso negado" in html
    assert "Cancela fechada" in html and "Cancela aberta" not in html


def test_gate_open_time_follows_the_setting(use_pipeline, db, tmp_path):
    db.add_vehicle("ABC1D23")
    db.set_setting(config.SETTING_GATE_OPEN_SECONDS, "9")
    pipeline = make_pipeline(db, tmp_path)
    use_pipeline(pipeline)
    at, reading = seeded_page(pipeline)
    at.run()
    assert f"sg-arm-{reading.number} 9s" in page_html(at)


def test_review_form_decides_again_and_opens_the_gate(use_pipeline, db, tmp_path):
    db.add_vehicle("ABC1D23")
    pipeline = make_pipeline(db, tmp_path, needs_review=True, reasons=["Low OCR confidence."])
    use_pipeline(pipeline)
    at, reading = seeded_page(pipeline)
    at.run()
    assert "sg-verdict-review" in page_html(at) and "Cancela aberta" not in page_html(at)
    assert at.text_input[0].value == "ABC1D23"                          # pre-filled with the OCR plate
    assert "não encontrada" not in page_html(at)                         # the registry is shown even in review
    assert any("ainda não foi confirmada" in c.value for c in at.caption)

    at.text_input[0].set_value("abc-1d23")
    at.button[0].click().run()
    assert not at.exception
    html = page_html(at)
    assert "sg-verdict-ok" in html and "Placa confirmada pelo operador" in html and "Cancela aberta" in html
    assert not at.text_input                                             # form disappears once resolved
    assert [log.source for log in db.list_access_logs()] == ["MANUAL", "AUTO"]


def test_review_form_rejects_an_invalid_plate_without_logging(use_pipeline, db, tmp_path):
    pipeline = make_pipeline(db, tmp_path, plate=None, needs_review=True, reasons=["not a plate"])
    use_pipeline(pipeline)
    at, _ = seeded_page(pipeline)
    at.run()
    assert at.text_input[0].value == ""
    at.text_input[0].set_value("123")
    at.button[0].click().run()
    assert any("Placa inválida" in e.value for e in at.error)
    assert "sg-verdict-review" in page_html(at)
    assert len(db.list_access_logs()) == 1                                # only the automatic attempt


def test_selector_switches_between_readings(use_pipeline, db, tmp_path):
    db.add_vehicle("ABC1D23")
    pipeline = make_pipeline(db, tmp_path)
    use_pipeline(pipeline)
    at, first = seeded_page(pipeline, number=1, name="primeira.jpg")
    second = access.Reading(key="upload:2", number=2, name="segunda.jpg", created=first.created,
                            result=make_pipeline(Database(tmp_path / "other.db"), tmp_path).process(SCENE))
    at.session_state[access.STATE_READINGS] = [second, first]
    at.session_state[access.STATE_SELECTED] = second.key
    at.run()
    assert "sg-verdict-denied" in page_html(at)                           # second reading: plate unknown here
    at.selectbox[0].set_value(first.key).run()
    assert "sg-verdict-ok" in page_html(at) and "Cancela fechada" not in page_html(at)


# ----------------------------------------------------------- image intake
def intake_script():
    import streamlit as st

    from interface import access

    pipeline = access.get_pipeline()
    present = access._ingest(st.session_state["files"], "upload", pipeline)
    st.session_state["present"] = present
    st.write(f"readings={len(st.session_state[access.STATE_READINGS])}")


def run_intake(files, pipeline):
    at = AppTest.from_function(intake_script, default_timeout=PAGE_TIMEOUT)
    at.session_state["files"] = files
    at.session_state[access.STATE_READINGS] = []
    at.session_state[access.STATE_SEEN] = set()
    at.session_state[access.STATE_ERRORS] = {}
    at.session_state[access.STATE_COUNTER] = 0
    return at.run()


def test_each_image_is_processed_once_across_reruns(use_pipeline, db, tmp_path):
    pipeline = make_pipeline(db, tmp_path)
    use_pipeline(pipeline)
    files = [FakeUpload("a", "um.png", png_bytes(SCENE)), FakeUpload("b", "dois.png", png_bytes(SCENE))]
    at = run_intake(files, pipeline)
    assert not at.exception
    readings = at.session_state[access.STATE_READINGS]
    assert [r.name for r in readings] == ["dois.png", "um.png"]            # newest first
    assert [r.number for r in readings] == [2, 1]
    assert at.session_state[access.STATE_SELECTED] == readings[0].key
    at.run()                                                               # a rerun must not reprocess
    assert len(at.session_state[access.STATE_READINGS]) == 2
    assert len(pipeline.ocr.calls) == 2


def test_invalid_files_are_reported_and_do_not_stop_the_others(use_pipeline, db, tmp_path):
    pipeline = make_pipeline(db, tmp_path)
    use_pipeline(pipeline)
    files = [FakeUpload("x", "doc.pdf", b"%PDF"), FakeUpload("y", "quebrada.png", b"not an image"),
             FakeUpload("z", "boa.png", png_bytes(SCENE))]
    at = run_intake(files, pipeline)
    assert not at.exception
    errors = at.session_state[access.STATE_ERRORS]
    assert set(errors) == {"upload:x", "upload:y"}
    assert "Unsupported file type" in errors["upload:x"][1]
    assert [r.name for r in at.session_state[access.STATE_READINGS]] == ["boa.png"]


def test_session_keeps_only_the_most_recent_readings(use_pipeline, db, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSION_READINGS_LIMIT", 2)
    pipeline = make_pipeline(db, tmp_path)
    use_pipeline(pipeline)
    files = [FakeUpload(str(i), f"{i}.png", png_bytes(SCENE)) for i in range(4)]
    at = run_intake(files, pipeline)
    assert [r.name for r in at.session_state[access.STATE_READINGS]] == ["3.png", "2.png"]
