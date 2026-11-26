"""Plate reading screen: capture, recognition, verdict and simulated gate.

Flow of one reading
-------------------
1. The operator uploads one or more images or takes a webcam snapshot.
2. Every new image is validated (:mod:`modules.image_io`) and sent through the
   shared :class:`~modules.pipeline.RecognitionPipeline`.
3. The result is kept in the session (newest first) so several vehicles can be
   processed one after another without restarting anything.
4. The screen shows the verdict (green / red / amber), the plate readout, the
   database lookup, the simulated gate and, below, the original image, the
   detected plate, the OpenCV steps, the histograms before / after
   equalization and the OCR details.
5. If the verdict is *manual review*, the operator confirms or corrects the
   plate and the system decides again (recorded with ``source="MANUAL"``).

Streamlit re-runs this script on every interaction, so everything that must
survive a click (readings, processed file ids, errors) lives in
``st.session_state``, and an image is processed only the first time it is seen.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from database.database import AccessResult
from interface import styles
from interface.resources import get_database, get_pipeline
from modules.access_control import AccessDecision, ManualReviewOutcome
from modules.image_io import ImageValidationError, bgr_to_rgb, load_image_from_bytes
from modules.image_processing import compute_histogram
from modules.pipeline import PipelineResult, PipelineStatus, RecognitionPipeline
from modules.plate_format import PlateFormat, format_for_display

UPLOAD_TYPES = [ext.lstrip(".") for ext in sorted(config.ALLOWED_EXTENSIONS)]

STATE_READINGS = "sg_readings"      # list[Reading], newest first
STATE_SEEN = "sg_seen"              # keys of the images already processed
STATE_ERRORS = "sg_errors"          # key -> (file name, message)
STATE_SELECTED = "sg_selected"      # key of the reading on display
STATE_COUNTER = "sg_counter"        # increasing id used to number readings

STATUS_TEXT = {
    PipelineStatus.NO_PLATE: "Nenhuma placa foi detectada na imagem.",
    PipelineStatus.DETECTION_FAILED: "A detecção da placa falhou.",
    PipelineStatus.PREPROCESS_FAILED: "O tratamento da imagem da placa falhou.",
    PipelineStatus.OCR_FAILED: "O leitor de caracteres (OCR) não está disponível.",
}
METHOD_TEXT = {"yolo": "YOLO11 (rede neural)", "classical": "OpenCV clássico (heurístico)", "none": "Nenhum", "manual": "Manual"}
FORMAT_TEXT = {PlateFormat.OLD: "Antigo", PlateFormat.MERCOSUL: "Mercosul", PlateFormat.INVALID: "Inválido"}
SOURCE_TEXT = {"enhanced": "Tons de cinza melhorado", "binary": "Binarizada"}


@dataclass
class Reading:
    """One processed image kept in the session."""

    key: str
    number: int
    name: str
    created: datetime
    result: PipelineResult
    resolution: ManualReviewOutcome | None = None   # set when the operator resolves a review
    opened_at: float | None = None                  # monotonic time when the gate started opening

    @property
    def decision(self) -> AccessDecision:
        return self.resolution.decision if self.resolution else self.result.decision

    @property
    def label(self) -> str:
        plate = format_for_display(self.decision.plate) if self.decision.plate else "sem leitura"
        verdict = styles.VERDICTS[self.decision.result][1]
        return f"{self.number}. {self.name} – {plate} – {verdict}"


# --------------------------------------------------------------------------
# Pure helpers (no Streamlit calls, unit tested)
# --------------------------------------------------------------------------
def describe_decision(result: PipelineResult, decision: AccessDecision, manual: bool = False) -> str:
    """One sentence in Portuguese explaining the verdict to the operator."""
    if decision.result is AccessResult.AUTHORIZED:
        text = "Veículo cadastrado e autorizado."
    elif decision.result is AccessResult.DENIED:
        text = "Veículo cadastrado, mas bloqueado." if decision.vehicle is not None else "Placa não cadastrada."
    else:
        text = ""
    if text:
        return f"Placa confirmada pelo operador. {text}" if manual else text
    # manual review
    if result.status is not PipelineStatus.OK:
        return STATUS_TEXT[result.status] + " Digite a placa para continuar."
    if decision.similar_vehicles:
        twins = ", ".join(format_for_display(v.plate) for v in decision.similar_vehicles)
        return f"Placa não cadastrada, mas existe uma parecida ({twins}). O 5º caractere pode ter sido lido errado."
    if result.ocr is not None and result.ocr.plate:
        return "Leitura incerta. Confira a placa na imagem antes de liberar."
    return "Nenhuma placa válida foi lida. Digite a placa para continuar."


def find_step(result: PipelineResult, name: str):
    """Image of the processing step called ``name`` (``None`` if it does not exist)."""
    for step in result.steps:
        if step.name == name:
            return step
    return None


def histogram_frame(result: PipelineResult) -> pd.DataFrame | None:
    """Intensity histograms before and after the equalization step (``None`` without a crop)."""
    before, after = find_step(result, "Denoise"), find_step(result, "Equalize")
    if before is None or after is None:
        return None
    return pd.DataFrame({
        "Antes da equalização": compute_histogram(before.image),
        "Depois da equalização": compute_histogram(after.image),
    })


def readings_table(result: PipelineResult) -> pd.DataFrame:
    """One row per image the OCR read (grayscale and binary)."""
    rows = []
    if result.ocr is not None:
        for r in result.ocr.readings:
            rows.append({
                "Imagem lida": SOURCE_TEXT.get(r.source, r.source),
                "Texto bruto": r.raw_text or "—",
                "Placa interpretada": format_for_display(r.plate) if r.plate else "inválida",
                "Confiança": f"{r.confidence:.0%}",
            })
    return pd.DataFrame(rows, columns=["Imagem lida", "Texto bruto", "Placa interpretada", "Confiança"])


# --------------------------------------------------------------------------
# Session state and image intake
# --------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault(STATE_READINGS, [])
    st.session_state.setdefault(STATE_SEEN, set())
    st.session_state.setdefault(STATE_ERRORS, {})
    st.session_state.setdefault(STATE_COUNTER, 0)


def _file_key(source: str, file) -> str:
    """Identity of an uploaded file: Streamlit's id, or a hash of the bytes."""
    file_id = getattr(file, "file_id", None)
    return f"{source}:{file_id}" if file_id else f"{source}:{hashlib.sha1(file.getvalue()).hexdigest()}"


def _ingest(files: list, source: str, pipeline: RecognitionPipeline) -> list[str]:
    """Process the files not seen before. Returns the keys of every file currently present."""
    present: list[str] = []
    new_files = []
    for file in files:
        key = _file_key(source, file)
        present.append(key)
        if key not in st.session_state[STATE_SEEN]:
            new_files.append((key, file))

    for index, (key, file) in enumerate(new_files, start=1):
        st.session_state[STATE_SEEN].add(key)
        name = file.name if source == "upload" else "Webcam"
        try:
            image = load_image_from_bytes(file.getvalue(), file.name if source == "upload" else None)
        except ImageValidationError as exc:
            st.session_state[STATE_ERRORS][key] = (name, str(exc))
            continue
        try:
            with st.spinner(f"Lendo a placa de {name} ({index}/{len(new_files)})…"):
                result = pipeline.process(image)
        except Exception as exc:  # the pipeline contains its own errors; this is the last safety net
            st.session_state[STATE_ERRORS][key] = (name, f"Erro inesperado ao processar a imagem: {exc}")
            continue
        st.session_state[STATE_COUNTER] += 1
        reading = Reading(key=key, number=st.session_state[STATE_COUNTER], name=name,
                          created=datetime.now(), result=result)
        readings: list[Reading] = st.session_state[STATE_READINGS]
        readings.insert(0, reading)
        del readings[config.SESSION_READINGS_LIMIT:]
        st.session_state[STATE_SELECTED] = key        # show the newest reading
    return present


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _show_image(image, caption: str) -> None:
    st.image(bgr_to_rgb(image), caption=caption, width="stretch")


def _render_vehicle(decision: AccessDecision) -> None:
    with st.container(border=True):
        st.markdown("**Consulta ao cadastro**")
        vehicle = decision.vehicle
        if vehicle is not None:
            lines = [
                f"**Placa:** {format_for_display(vehicle.plate)}",
                f"**Proprietário:** {vehicle.owner or '—'}",
                f"**Veículo:** {vehicle.model or '—'}, {vehicle.color or 'cor não informada'}",
                f"**Situação:** {'Autorizado' if vehicle.authorized else 'Bloqueado'}",
            ]
            st.markdown("  \n".join(lines))
        elif decision.similar_vehicles:
            st.write("Placa não cadastrada. Placas parecidas no cadastro:")
            for v in decision.similar_vehicles:
                st.write(f"- {format_for_display(v.plate)} ({v.owner or 'sem proprietário'})")
        elif decision.plate:
            st.write("Placa não encontrada no cadastro.")
        else:
            st.write("Sem placa para consultar.")


def _render_gate(reading: Reading) -> None:
    decision, result = reading.decision, reading.result
    seconds = result.gate_open_seconds
    opening = False
    if decision.opens_gate:
        if reading.opened_at is None:
            reading.opened_at = time.monotonic()
        # while the animation is running, reruns emit identical HTML, so it is not restarted
        opening = time.monotonic() - reading.opened_at < seconds
    with st.container(border=True):
        st.markdown("**Cancela**")
        st.markdown(styles.gate_html(opening, decision.result, seconds, run_id=reading.number),
                    unsafe_allow_html=True)


def _render_review_form(reading: Reading, pipeline: RecognitionPipeline) -> None:
    ocr = reading.result.ocr
    prefill = format_for_display(ocr.plate) if ocr and ocr.plate else ""
    with st.container(border=True):
        st.markdown("**Revisão do operador**")
        st.caption("Confira a placa na imagem, digite ou confirme e o sistema decide de novo. "
                   "A ação fica registrada como manual no histórico.")
        with st.form(f"review_{reading.key}", border=False):
            value = st.text_input("Placa", value=prefill, max_chars=8, placeholder="ABC-1234 ou ABC1D23")
            submitted = st.form_submit_button("Confirmar placa", type="primary")
        if submitted:
            try:
                reading.resolution = pipeline.review_manually(reading.result, value)
            except ValueError:
                st.error("Placa inválida. Use o formato antigo (ABC-1234) ou Mercosul (ABC1D23).")
            else:
                st.rerun()


def _render_verdict_area(reading: Reading, pipeline: RecognitionPipeline) -> None:
    result, decision = reading.result, reading.decision
    manual = reading.resolution is not None
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(styles.verdict_html(decision.result, describe_decision(result, decision, manual)),
                    unsafe_allow_html=True)
        if result.duplicate and not manual:
            st.info(f"Leitura repetida: a mesma placa já foi registrada nos últimos "
                    f"{pipeline.history.duplicate_window_seconds} s, então nenhum registro novo foi criado.")
        if result.history_error:
            st.warning("Este acesso não foi gravado no histórico. Detalhes na aba Detalhes técnicos.")
        plate_col, vehicle_col = st.columns(2, gap="medium")
        with plate_col:
            caption = "Placa confirmada pelo operador" if manual else (
                "Placa lida pelo OCR" if decision.plate else "")
            st.markdown(styles.plate_html(decision.plate, caption), unsafe_allow_html=True)
            chips = [styles.chip_html(METHOD_TEXT.get(result.detection_method, result.detection_method))]
            if result.ocr is not None:
                chips.append(styles.chip_html(f"OCR {result.ocr.confidence:.0%}"))
            if manual:
                chips.append(styles.chip_html("Revisão manual", manual=True))
            st.markdown("".join(chips), unsafe_allow_html=True)
        with vehicle_col:
            _render_vehicle(decision)
        if decision.result is AccessResult.MANUAL_REVIEW:
            _render_review_form(reading, pipeline)
    with right:
        _render_gate(reading)


def _render_images(result: PipelineResult) -> None:
    cols = st.columns(3, gap="medium")
    with cols[0]:
        _show_image(result.original, "Imagem original")
    with cols[1]:
        _show_image(result.annotated, "Placa detectada")
    with cols[2]:
        if result.crop is not None:
            _show_image(result.crop, "Recorte da placa")
        else:
            st.info("Sem recorte: nenhuma placa foi detectada.")


def _render_ocr_tab(result: PipelineResult, pipeline: RecognitionPipeline) -> None:
    ocr = result.ocr
    if ocr is None:
        st.info("O OCR não foi executado nesta leitura. " + STATUS_TEXT.get(result.status, ""))
        return
    minimum = pipeline.config.ocr.min_confidence
    a, b, c = st.columns(3)
    a.metric("Texto bruto do OCR", ocr.raw_text or "—")
    b.metric("Placa interpretada", format_for_display(ocr.plate) if ocr.plate else "inválida")
    c.metric("Formato", FORMAT_TEXT[ocr.format])
    st.progress(min(max(ocr.confidence, 0.0), 1.0),
                text=f"Confiança do OCR: {ocr.confidence:.0%} (mínimo exigido: {minimum:.0%})")
    st.markdown("**Correções aplicadas por posição**")
    if ocr.corrections:
        for fix in ocr.corrections:
            st.write(f"- Posição {fix.position + 1}: `{fix.original}` → `{fix.corrected}`")
    else:
        st.write("Nenhuma correção foi necessária.")
    if ocr.reasons:
        st.warning("Motivos para revisão (informados pelo OCR):\n\n" + "\n".join(f"- {r}" for r in ocr.reasons))
    table = readings_table(result)
    if not table.empty:
        st.markdown("**Leitura de cada imagem**")
        st.dataframe(table, hide_index=True, width="stretch")


def _render_steps_tab(result: PipelineResult) -> None:
    if not result.steps:
        st.info("Sem etapas: o tratamento da imagem só roda quando uma placa é detectada.")
        return
    st.caption(f"Ângulo corrigido: {result.processing.skew_angle:+.1f}°. "
               + ("Placa com caracteres claros: imagem invertida." if result.processing.inverted else ""))
    for start in range(0, len(result.steps), 4):
        cols = st.columns(4, gap="small")
        for col, step in zip(cols, result.steps[start:start + 4], strict=False):
            with col:
                _show_image(step.image, step.name)
                st.caption(step.description)


def _render_histogram_tab(result: PipelineResult) -> None:
    frame = histogram_frame(result)
    if frame is None:
        st.info("Sem histograma: nenhuma placa foi tratada nesta leitura.")
        return
    before, after = find_step(result, "Denoise"), find_step(result, "Equalize")
    a, b = st.columns(2)
    with a:
        _show_image(before.image, "Antes da equalização")
    with b:
        _show_image(after.image, "Depois da equalização")
        st.caption(after.description)
    st.line_chart(frame, x_label="Intensidade do pixel (0 = preto, 255 = branco)", y_label="Quantidade de pixels")


def _render_details_tab(reading: Reading) -> None:
    result = reading.result
    det = result.detection.best if result.detection else None
    if det is None:
        det_confidence = "—"
    elif det.method == "classical":
        det_confidence = f"{det.confidence:.2f} (pontuação heurística, não é probabilidade)"
    else:
        det_confidence = f"{det.confidence:.2f} (YOLO)"
    if result.log is None:
        logged = "não gravado"
    else:
        logged = f"#{result.log.id}" + (" (já existia)" if result.duplicate else "")
    rows = {
        "Etapa final do pipeline": "Concluído" if result.status is PipelineStatus.OK else STATUS_TEXT[result.status],
        "Método de detecção": METHOD_TEXT.get(result.detection_method, result.detection_method),
        "Confiança da detecção": det_confidence,
        "Caixa da placa (x1, y1, x2, y2)": str(det.bbox) if det else "—",
        "Iluminação corrigida (CLAHE)": "Sim" if result.detection and result.detection.illumination_fixed else "Não",
        "Tempo de processamento": f"{result.elapsed_ms:.0f} ms",
        "Registro no histórico": logged,
        "Imagem salva": result.log.image_path if result.log and result.log.image_path else "—",
        "Motivo técnico da decisão": result.decision.reason,
    }
    st.table(pd.DataFrame({"Informação": list(rows), "Valor": list(rows.values())}).set_index("Informação"))
    if result.messages:
        st.markdown("**Mensagens do sistema**")
        for message in result.messages:
            st.write(f"- {message}")


def _render_reading(reading: Reading, pipeline: RecognitionPipeline) -> None:
    _render_verdict_area(reading, pipeline)
    _render_images(reading.result)
    tabs = st.tabs(["Reconhecimento (OCR)", "Etapas do OpenCV", "Histograma", "Detalhes técnicos"])
    with tabs[0]:
        _render_ocr_tab(reading.result, pipeline)
    with tabs[1]:
        _render_steps_tab(reading.result)
    with tabs[2]:
        _render_histogram_tab(reading.result)
    with tabs[3]:
        _render_details_tab(reading)


def render() -> None:
    """Page function registered in ``app.py``."""
    _init_state()
    pipeline = get_pipeline()
    st.header("Leitura de placas")

    upload_tab, camera_tab = st.tabs(["Enviar imagens", "Webcam"])
    with upload_tab:
        uploads = st.file_uploader(
            f"Fotos de veículos (JPG, JPEG ou PNG, até {config.MAX_UPLOAD_SIZE_MB} MB cada). "
            "Você pode enviar várias de uma vez.",
            type=UPLOAD_TYPES, accept_multiple_files=True, max_upload_size=config.MAX_UPLOAD_SIZE_MB,
        )
    with camera_tab:
        snapshot = st.camera_input("Posicione a placa no centro e tire a foto")

    present = _ingest(list(uploads or []), "upload", pipeline)
    present += _ingest([snapshot] if snapshot is not None else [], "camera", pipeline)
    for key in present:
        if key in st.session_state[STATE_ERRORS]:
            name, message = st.session_state[STATE_ERRORS][key]
            st.error(f"**{name}:** {message}")

    if not get_database().list_vehicles():
        st.warning("Nenhum veículo está cadastrado: toda placa lida será negada até que exista um cadastro.")

    readings: list[Reading] = st.session_state[STATE_READINGS]
    if not readings:
        st.info("Envie uma foto ou use a webcam para ler a primeira placa.")
        return

    keys = [r.key for r in readings]
    if st.session_state.get(STATE_SELECTED) not in keys:
        st.session_state[STATE_SELECTED] = keys[0]
    labels = {r.key: r.label for r in readings}
    selected = st.selectbox("Leitura em exibição", keys, key=STATE_SELECTED,
                            format_func=labels.__getitem__,
                            help="As leituras desta sessão, da mais recente para a mais antiga.")
    _render_reading(next(r for r in readings if r.key == selected), pipeline)
