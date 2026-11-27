"""Helpers shared by the management screens (dashboard, vehicles, history...).

Pure functions (no Streamlit calls) are unit tested; the two Streamlit helpers
implement the "flash message" pattern: an action stores a message, calls
``st.rerun()`` so the page redraws with the new data, and the message is shown
once on the next run.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from database.database import AccessLog, AccessResult, Vehicle
from modules.history import resolve_image_path
from modules.plate_format import format_for_display

RESULT_LABELS = {
    AccessResult.AUTHORIZED: "Autorizado",
    AccessResult.DENIED: "Negado",
    AccessResult.MANUAL_REVIEW: "Revisão manual",
}
SOURCE_LABELS = {"AUTO": "Automática", "MANUAL": "Operador"}
METHOD_LABELS = {"yolo": "YOLO11", "classical": "OpenCV clássico", "none": "Nenhum", "manual": "Manual", "": "—"}

THUMBNAIL_HEIGHT = 72
THUMBNAIL_QUALITY = 70

STATE_FLASH = "sg_flash"


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def display_plate(plate: str | None) -> str:
    return format_for_display(plate) if plate else "—"


def logs_frame(logs: list[AccessLog], thumbnails: bool = False) -> pd.DataFrame:
    """Access logs as a table with Portuguese column names (newest first, as given).

    ``thumbnails=True`` adds an ``Imagem`` column with JPEG data URIs for
    ``st.column_config.ImageColumn``.
    """
    columns = ["ID", "Data e hora", "Placa", "Resultado", "Confiança", "Origem", "Detecção", "Texto do OCR",
               "Observações"]
    rows = []
    for log in logs:
        row = {
            "ID": log.id,
            "Data e hora": log.timestamp,
            "Placa": display_plate(log.plate),
            "Resultado": RESULT_LABELS[log.result],
            "Confiança": None if log.confidence is None else round(log.confidence * 100, 1),
            "Origem": SOURCE_LABELS.get(log.source, log.source),
            "Detecção": METHOD_LABELS.get(log.detection_method, log.detection_method),
            "Texto do OCR": log.raw_text,
            "Observações": log.notes,
        }
        if thumbnails:
            row["Imagem"] = thumbnail_data_uri(log.image_path)
        rows.append(row)
    if thumbnails:
        columns.insert(1, "Imagem")
    return pd.DataFrame(rows, columns=columns)


def csv_bytes(logs: list[AccessLog]) -> bytes:
    """CSV export of the logs (UTF-8 with BOM so Excel shows accents correctly)."""
    frame = logs_frame(logs)
    frame["Caminho da imagem"] = [log.image_path or "" for log in logs]
    return frame.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def vehicles_frame(vehicles: list[Vehicle]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Placa": display_plate(v.plate), "Proprietário": v.owner, "Modelo": v.model, "Cor": v.color,
          "Situação": "Autorizado" if v.authorized else "Bloqueado", "Atualizado em": v.updated_at}
         for v in vehicles],
        columns=["Placa", "Proprietário", "Modelo", "Cor", "Situação", "Atualizado em"],
    )


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def read_image(path: Path) -> np.ndarray | None:
    """Read an image from disk (``np.fromfile`` works with non-ASCII Windows paths)."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


@st.cache_data(show_spinner=False, max_entries=2000)
def _thumbnail(path: str, mtime: float) -> str | None:   # mtime is part of the cache key
    image = read_image(Path(path))
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = THUMBNAIL_HEIGHT / h
    small = cv2.resize(image, (max(1, round(w * scale)), THUMBNAIL_HEIGHT), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, THUMBNAIL_QUALITY])
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode()}" if ok else None


def thumbnail_data_uri(stored_path: str | None) -> str | None:
    """Small JPEG of a stored capture as a data URI (``None`` if the file is missing)."""
    path = resolve_image_path(stored_path)
    if path is None:
        return None
    return _thumbnail(str(path), path.stat().st_mtime)


# --------------------------------------------------------------------------
# Flash messages
# --------------------------------------------------------------------------
def flash(message: str, kind: str = "success") -> None:
    """Store a message to show after the next rerun (``kind``: success, info, warning, error)."""
    st.session_state[STATE_FLASH] = (kind, message)


def show_flash() -> None:
    item = st.session_state.pop(STATE_FLASH, None)
    if item:
        kind, message = item
        getattr(st, kind)(message)
