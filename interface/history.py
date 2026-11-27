"""Access history: filters, thumbnails, capture viewer and CSV export.

Filters (plate, result, period) are applied in SQL by
:meth:`database.database.Database.list_access_logs`. The table shows at most
``DISPLAY_LIMIT`` rows (each one carries a thumbnail), while the CSV export
contains *every* row that matches the filters.
"""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from database.database import AccessLog, AccessResult
from interface.common import RESULT_LABELS, SOURCE_LABELS, csv_bytes, display_plate, logs_frame, read_image
from interface.resources import get_database
from modules.history import resolve_image_path
from modules.image_io import bgr_to_rgb

DISPLAY_LIMIT = 200
DEFAULT_DAYS = 30
ALL_RESULTS = "all"


def period_bounds(value, today: date) -> tuple[str | None, str | None]:
    """Convert the ``st.date_input`` range to ``(start, end)`` strings.

    While the operator is picking the range the widget returns a single date;
    it is then used as both bounds.
    """
    if not value:
        return None, None
    if isinstance(value, date):
        value = (value,)
    start = value[0]
    end = value[1] if len(value) > 1 else value[0]
    return start.isoformat(), min(end, today).isoformat() if end else None


def _filters(today: date) -> dict:
    a, b, c = st.columns([2, 2, 3])
    plate = a.text_input("Placa", placeholder="Parte da placa", key="sg_hist_plate")
    options = [ALL_RESULTS, *AccessResult]
    result = b.selectbox("Resultado", options, key="sg_hist_result",
                         format_func=lambda r: "Todos" if r == ALL_RESULTS else RESULT_LABELS[r])
    period = c.date_input("Período", value=(today - timedelta(days=DEFAULT_DAYS - 1), today),
                          max_value=today, format="DD/MM/YYYY", key="sg_hist_period")
    start, end = period_bounds(period, today)
    return {"plate": plate, "result": None if result == ALL_RESULTS else AccessResult(result),
            "start": start, "end": end}


def _render_capture(log: AccessLog) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        path = resolve_image_path(log.image_path)
        image = read_image(path) if path else None
        if image is not None:
            st.image(bgr_to_rgb(image), caption=f"Captura do registro #{log.id}", width="stretch")
        else:
            st.info("A imagem deste registro não está disponível (não foi salva ou foi apagada).")
    with right:
        rows = [
            f"**Registro:** #{log.id}",
            f"**Data e hora:** {log.timestamp}",
            f"**Placa:** {display_plate(log.plate)}",
            f"**Resultado:** {RESULT_LABELS[log.result]}",
            f"**Origem:** {SOURCE_LABELS.get(log.source, log.source)}",
            f"**Texto do OCR:** {log.raw_text or '—'}",
            f"**Confiança:** {'—' if log.confidence is None else f'{log.confidence:.0%}'}",
            f"**Observações:** {log.notes or '—'}",
        ]
        st.markdown("  \n".join(rows))


def render() -> None:
    """Page function registered in ``app.py``."""
    st.header("Histórico de acessos")
    today = date.today()
    filters = _filters(today)
    db = get_database()
    logs = db.list_access_logs(**filters, limit=None)
    if not logs:
        st.info("Nenhum registro encontrado com esses filtros.")
        return

    shown = logs[:DISPLAY_LIMIT]
    info, export = st.columns([4, 1], vertical_alignment="center")
    info.caption(f"{len(logs)} registro(s) encontrado(s)"
                 + (f"; mostrando os {DISPLAY_LIMIT} mais recentes." if len(logs) > DISPLAY_LIMIT else "."))
    export.download_button("Exportar CSV", data=csv_bytes(logs), file_name=f"historico_{today.isoformat()}.csv",
                           mime="text/csv", icon=":material/download:", width="stretch")

    st.dataframe(logs_frame(shown, thumbnails=True), hide_index=True, width="stretch", row_height=56,
                 column_config={
                     "Imagem": st.column_config.ImageColumn("Imagem", width="small"),
                     "Confiança": st.column_config.NumberColumn("Confiança do OCR", format="%.0f%%"),
                 })

    st.subheader("Ver captura")
    by_id = {log.id: log for log in shown}
    selected = st.selectbox("Registro", list(by_id), key="sg_hist_selected",
                            format_func=lambda i: f"#{i} – {by_id[i].timestamp} – {display_plate(by_id[i].plate)} – "
                                                  f"{RESULT_LABELS[by_id[i].result]}")
    _render_capture(by_id[selected])
