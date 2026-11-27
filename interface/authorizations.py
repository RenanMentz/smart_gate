"""Authorizations: allow or block registered plates.

Two ways to act:

* **By plate** - type a plate and authorize or block it at once (the usual
  situation at the gate: "block ABC1D23 from now on"). A plate that is not
  registered can be registered on the spot as authorized.
* **From the list** - every registered vehicle with a button that switches its
  status, with filters by status and text.

The change is effective on the next reading: the access decision reads the
database every time.
"""

from __future__ import annotations

import streamlit as st

from database.database import DuplicatePlateError
from interface.common import display_plate, flash, show_flash
from interface.resources import get_database
from modules.plate_format import normalize_plate

STATUS_FILTERS = {"all": "Todos", "authorized": "Autorizados", "blocked": "Bloqueados"}
LIST_LIMIT = 200


def set_status_by_plate(plate_text: str, authorized: bool, register_missing: bool) -> tuple[str, str]:
    """Authorize or block a plate. Returns ``(kind, message)`` for the operator.

    ``register_missing`` registers an unknown plate (only when authorizing).
    """
    db = get_database()
    try:
        plate = normalize_plate(plate_text)
    except ValueError:
        return "error", "Placa inválida. Use o formato antigo (ABC-1234) ou Mercosul (ABC1D23)."
    vehicle = db.find_vehicle_by_plate(plate)
    action = "autorizada" if authorized else "bloqueada"
    if vehicle is None:
        if authorized and register_missing:
            try:
                db.add_vehicle(plate, authorized=True)
            except DuplicatePlateError:      # registered by someone else meanwhile
                vehicle = db.find_vehicle_by_plate(plate)
            else:
                return "success", f"Placa {display_plate(plate)} cadastrada e autorizada."
        else:
            hint = (" Marque “Cadastrar se não existir” para cadastrar e autorizar." if authorized
                    else " Uma placa sem cadastro já é negada na cancela.")
            return "warning", f"A placa {display_plate(plate)} não está cadastrada.{hint}"
    if vehicle.authorized == authorized:
        return "info", f"A placa {display_plate(plate)} já estava {action}."
    db.set_authorization(vehicle.id, authorized)
    return "success", f"Placa {display_plate(plate)} {action}."


def _render_by_plate() -> None:
    with st.form("sg_auth_plate", border=True):
        st.markdown("**Autorizar ou bloquear por placa**")
        plate = st.text_input("Placa", max_chars=8, placeholder="ABC-1234 ou ABC1D23")
        register = st.checkbox("Cadastrar se não existir (somente ao autorizar)")
        a, b = st.columns(2)
        allow = a.form_submit_button("Autorizar", type="primary", icon=":material/check_circle:")
        block = b.form_submit_button("Bloquear", icon=":material/block:")
    if allow or block:
        kind, message = set_status_by_plate(plate, authorized=allow, register_missing=register)
        if kind == "error":
            st.error(message)
        else:
            flash(message, kind)
            st.rerun()


def _render_list() -> None:
    st.subheader("Situação dos veículos")
    a, b = st.columns([1, 2])
    status = a.segmented_control("Mostrar", list(STATUS_FILTERS), default="all",
                                 format_func=STATUS_FILTERS.__getitem__, key="sg_auth_filter") or "all"
    search = b.text_input("Pesquisar", placeholder="Placa ou proprietário", key="sg_auth_search")
    authorized = {"all": None, "authorized": True, "blocked": False}[status]
    vehicles = get_database().list_vehicles(search=search, authorized=authorized)
    if not vehicles:
        st.info("Nenhum veículo encontrado.")
        return
    if len(vehicles) > LIST_LIMIT:
        st.caption(f"Mostrando {LIST_LIMIT} de {len(vehicles)} veículos. Refine a pesquisa.")
    for vehicle in vehicles[:LIST_LIMIT]:
        with st.container(border=True):
            plate_col, info_col, status_col, action_col = st.columns([2, 4, 2, 2], vertical_alignment="center")
            plate_col.markdown(f"**{display_plate(vehicle.plate)}**")
            details = " · ".join(x for x in (vehicle.owner, vehicle.model, vehicle.color) if x)
            info_col.write(details or "Sem dados adicionais")
            if vehicle.authorized:
                status_col.markdown(":green-badge[:material/check: Autorizado]")
                label, icon = "Bloquear", ":material/block:"
            else:
                status_col.markdown(":red-badge[:material/block: Bloqueado]")
                label, icon = "Autorizar", ":material/check_circle:"
            if action_col.button(label, icon=icon, key=f"sg_auth_toggle_{vehicle.id}", width="stretch"):
                get_database().set_authorization(vehicle.id, not vehicle.authorized)
                flash(f"Placa {display_plate(vehicle.plate)} {'bloqueada' if vehicle.authorized else 'autorizada'}.")
                st.rerun()


def render() -> None:
    """Page function registered in ``app.py``."""
    st.header("Autorizações")
    show_flash()
    _render_by_plate()
    _render_list()
