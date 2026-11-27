"""Dashboard: today's activity, totals per result and the latest attempts.

The activity chart is drawn as *small multiples*: one bar chart per result
(authorized / denied / manual review) instead of one stacked chart. Green and
red are hard to tell apart for people with red-green colour blindness, so each
chart is identified by its title and the colour only reinforces it.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from database.database import TIMESTAMP_FORMAT, AccessLog, AccessResult
from interface import styles
from interface.common import RESULT_LABELS, logs_frame
from interface.resources import get_database

PERIODS = {"today": "Hoje, por hora", "7d": "Últimos 7 dias", "30d": "Últimos 30 dias"}
RESULT_COLORS = {AccessResult.AUTHORIZED: styles.GREEN, AccessResult.DENIED: styles.RED,
                 AccessResult.MANUAL_REVIEW: styles.AMBER}
LATEST_ATTEMPTS = 10


def period_start(period: str, today: date) -> date:
    """First day included in the period."""
    days = {"today": 0, "7d": 6, "30d": 29}[period]
    return today - timedelta(days=days)


def activity_frame(logs: list[AccessLog], period: str, today: date) -> pd.DataFrame:
    """Number of attempts per result in each hour (``today``) or each day (``7d`` / ``30d``).

    Every hour / day of the period is present, with zeros when there was no
    attempt, so the bars are spaced evenly. Columns are the result labels.
    """
    start = period_start(period, today)
    if period == "today":
        index = pd.Index(range(24), name="Hora")
    else:
        index = pd.DatetimeIndex(pd.date_range(start, today, freq="D"), name="Dia")
    frame = pd.DataFrame(0, index=index, columns=[RESULT_LABELS[r] for r in AccessResult])
    for log in logs:
        when = datetime.strptime(log.timestamp, TIMESTAMP_FORMAT)
        if not start <= when.date() <= today:
            continue
        key = when.hour if period == "today" else pd.Timestamp(when.date())
        frame.loc[key, RESULT_LABELS[log.result]] += 1
    return frame


def _render_metrics(today: date) -> None:
    db = get_database()
    today_counts = db.count_logs_by_result(start=today.isoformat())
    total_counts = db.count_logs_by_result()
    vehicles = db.list_vehicles()
    authorized = sum(v.authorized for v in vehicles)

    st.subheader("Acessos de hoje")
    cols = st.columns(4)
    cols[0].metric("Tentativas", sum(today_counts.values()))
    for col, result in zip(cols[1:], AccessResult, strict=True):
        col.metric(RESULT_LABELS[result], today_counts[result])

    st.subheader("Desde o início")
    cols = st.columns(4)
    cols[0].metric("Tentativas", sum(total_counts.values()))
    for col, result in zip(cols[1:], AccessResult, strict=True):
        col.metric(RESULT_LABELS[result], total_counts[result])
    st.caption(f"Veículos cadastrados: {len(vehicles)} ({authorized} autorizados, "
               f"{len(vehicles) - authorized} bloqueados).")


def _render_activity(today: date) -> None:
    st.subheader("Atividade")
    period = st.segmented_control("Período", list(PERIODS), default="today", format_func=PERIODS.__getitem__,
                                  key="sg_dashboard_period") or "today"
    start = period_start(period, today)
    logs = get_database().list_access_logs(start=start.isoformat(), end=today.isoformat(), limit=None)
    frame = activity_frame(logs, period, today)
    x_label = "Hora do dia" if period == "today" else "Dia"
    cols = st.columns(3, gap="medium")
    for col, result in zip(cols, AccessResult, strict=True):
        label = RESULT_LABELS[result]
        with col, st.container(border=True):
            st.markdown(f"**{label}** · {int(frame[label].sum())} no período")
            st.bar_chart(frame[[label]], color=RESULT_COLORS[result], height=220,
                         x_label=x_label, y_label="Tentativas")
    with st.expander("Ver os números em tabela"):
        table = frame.copy()
        if period != "today":
            table.index = table.index.strftime("%d/%m/%Y")
        st.dataframe(table, width="stretch")


def _render_latest() -> None:
    st.subheader("Últimas tentativas")
    logs = get_database().list_access_logs(limit=LATEST_ATTEMPTS)
    if not logs:
        st.info("Nenhuma tentativa de acesso registrada ainda. Use a tela Leitura de placas.")
        return
    frame = logs_frame(logs, thumbnails=True)[["Imagem", "Data e hora", "Placa", "Resultado", "Confiança", "Origem"]]
    st.dataframe(frame, hide_index=True, width="stretch", row_height=56, column_config={
        "Imagem": st.column_config.ImageColumn("Imagem", width="small"),
        "Confiança": st.column_config.NumberColumn("Confiança do OCR", format="%.0f%%"),
    })


def render() -> None:
    """Page function registered in ``app.py``."""
    header, button = st.columns([5, 1], vertical_alignment="bottom")
    header.header("Dashboard")
    if button.button("Atualizar", icon=":material/refresh:", width="stretch"):
        st.rerun()
    today = date.today()
    _render_metrics(today)
    _render_activity(today)
    _render_latest()
