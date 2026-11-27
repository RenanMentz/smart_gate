"""Settings: runtime thresholds and the status of models and GPU.

The thresholds are stored in the ``settings`` table. The pipeline re-reads that
table before every image (:meth:`modules.pipeline.RecognitionPipeline.refresh_config`),
so a saved value takes effect on the next reading without restarting.
Allowed ranges come from :data:`config.SETTING_LIMITS`; removing a row
("restore defaults") brings back the value in ``config.py``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

import config
from interface.common import flash, show_flash
from interface.resources import get_database, get_pipeline
from modules import system_status
from scripts.download_model import SHA256 as PUBLISHED_SHA256


def current_config() -> config.AppConfig:
    return config.apply_settings(config.DEFAULT_CONFIG, get_database().get_all_settings())


def settings_from_form(ocr_percent: int, yolo_percent: int, duplicate_seconds: int, gate_seconds: int) -> dict:
    """Form values -> strings stored in the ``settings`` table."""
    return {
        config.SETTING_OCR_MIN_CONFIDENCE: f"{ocr_percent / 100:.2f}",
        config.SETTING_YOLO_CONFIDENCE: f"{yolo_percent / 100:.2f}",
        config.SETTING_DUPLICATE_WINDOW: str(int(duplicate_seconds)),
        config.SETTING_GATE_OPEN_SECONDS: str(int(gate_seconds)),
    }


def _limits(key: str) -> tuple[float, float]:
    _, low, high = config.SETTING_LIMITS[key]
    return low, high


def _render_thresholds() -> None:
    cfg, default = current_config(), config.DEFAULT_CONFIG
    st.subheader("Limiares")
    with st.form("sg_settings"):
        low, high = _limits(config.SETTING_OCR_MIN_CONFIDENCE)
        ocr = st.slider("Confiança mínima do OCR (%)", int(low * 100), int(high * 100),
                        round(cfg.ocr.min_confidence * 100), step=1,
                        help=f"Leituras abaixo deste valor vão para revisão manual. "
                             f"Padrão: {default.ocr.min_confidence:.0%}.")
        low, high = _limits(config.SETTING_YOLO_CONFIDENCE)
        yolo = st.slider("Confiança mínima do YOLO (%)", int(low * 100), int(high * 100),
                         round(cfg.detection.confidence_threshold * 100), step=1,
                         help=f"Caixas com confiança menor são descartadas. Valores altos perdem placas "
                              f"difíceis; baixos aceitam falsos positivos. "
                              f"Padrão: {default.detection.confidence_threshold:.0%}.")
        a, b = st.columns(2)
        low, high = _limits(config.SETTING_DUPLICATE_WINDOW)
        window = a.number_input("Janela de duplicidade (s)", int(low), int(high), cfg.duplicate_window_seconds,
                                step=1, help="A mesma placa com o mesmo resultado dentro desta janela não gera "
                                             f"registro novo. 0 desativa. Padrão: {default.duplicate_window_seconds} s.")
        low, high = _limits(config.SETTING_GATE_OPEN_SECONDS)
        gate = b.number_input("Tempo da cancela aberta (s)", int(low), int(high), cfg.gate_open_seconds, step=1,
                              help=f"Duração da animação da cancela. Padrão: {default.gate_open_seconds} s.")
        save, reset = st.columns(2)
        saved = save.form_submit_button("Salvar", type="primary", icon=":material/save:")
        restored = reset.form_submit_button("Restaurar padrões", icon=":material/restart_alt:")
    db = get_database()
    if saved:
        for key, value in settings_from_form(ocr, yolo, window, gate).items():
            db.set_setting(key, value)
        flash("Configurações salvas. Valem a partir da próxima leitura.")
        st.rerun()
    if restored:
        for key in config.SETTING_LIMITS:
            db.delete_setting(key)
        flash("Valores padrão restaurados.")
        st.rerun()


@st.cache_data(show_spinner=False, max_entries=8)
def _file_sha256(path: str, mtime: float) -> str:       # mtime is part of the cache key
    return system_status.sha256_of(Path(path))


def _status_line(ok: bool | None, title: str, detail: str) -> None:
    if ok:
        st.success(f"**{title}:** {detail}", icon=":material/check_circle:")
    elif ok is None:
        st.info(f"**{title}:** {detail}", icon=":material/schedule:")
    else:
        st.error(f"**{title}:** {detail}", icon=":material/error:")


def _render_status() -> None:
    st.subheader("Modelos e hardware")
    pipeline = get_pipeline()
    detector, ocr = pipeline.detector, pipeline.ocr

    gpu = system_status.gpu_status()
    gpu_detail = gpu.message
    if gpu.cuda_available:
        gpu_detail += f" Capacidade {gpu.capability}, PyTorch {gpu.torch_version} (CUDA {gpu.cuda_version})."
    elif gpu.torch_version:
        gpu_detail += f" PyTorch {gpu.torch_version}."
    # no GPU is a valid setup (CPU works, only slower): shown as information, not as an error
    _status_line(gpu.usable if gpu.cuda_available else None, "GPU", gpu_detail)

    file_hash = None
    if detector.model_path.is_file():
        file_hash = _file_sha256(str(detector.model_path), detector.model_path.stat().st_mtime)
    yolo = system_status.yolo_status(detector, PUBLISHED_SHA256, file_hash)
    _status_line(False if yolo.error or not detector.model_path.is_file() else (True if yolo.loaded else None),
                 "Detector YOLO11", yolo.detail + (f" Erro: {yolo.error}" if yolo.error else ""))

    easy = system_status.ocr_status(ocr)
    _status_line(False if easy.error else (True if easy.loaded else None),
                 "Leitor de caracteres (OCR)", easy.detail + (f" Erro: {easy.error}" if easy.error else ""))

    if not (detector.loaded and ocr.loaded):
        if st.button("Carregar modelos agora", icon=":material/memory:",
                     help="Carrega o YOLO e o leitor de caracteres já, para a primeira leitura não demorar. "
                          "Se um carregamento falhou, tenta de novo."):
            detector.load_error = None          # allow a new attempt after fixing the problem
            ocr.load_error = None
            ocr.plate_model_error = None
            with st.spinner("Carregando modelos (pode levar alguns segundos)…"):
                ok_yolo, ok_ocr = detector.yolo_available, ocr.available
            flash("Modelos carregados." if ok_yolo and ok_ocr else "Um dos modelos não carregou; veja o erro acima.",
                  "success" if ok_yolo and ok_ocr else "warning")
            st.rerun()


def render() -> None:
    """Page function registered in ``app.py``."""
    st.header("Configurações")
    show_flash()
    _render_thresholds()
    _render_status()
