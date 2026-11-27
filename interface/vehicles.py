"""Vehicle registry: register, search, edit and remove vehicles.

Plates are normalized by the database (``abc-1d23`` -> ``ABC1D23``), so the
operator can type them with or without hyphen and in any case. Invalid and
duplicate plates are rejected with a message. Removing a vehicle asks for a
confirmation first; its access history is kept.
"""

from __future__ import annotations

import streamlit as st

from database.database import DuplicatePlateError, Vehicle, VehicleNotFoundError
from interface.common import display_plate, flash, show_flash, vehicles_frame
from interface.resources import get_database

STATE_EDIT = "sg_vehicle_edit"            # id of the vehicle selected for editing
STATE_CONFIRM_DELETE = "sg_vehicle_delete"  # id of the vehicle waiting for delete confirmation

INVALID_PLATE = "Placa inválida. Use o formato antigo (ABC-1234) ou Mercosul (ABC1D23)."


def _save_new(plate: str, owner: str, model: str, color: str, authorized: bool) -> None:
    try:
        vehicle = get_database().add_vehicle(plate, model=model, color=color, owner=owner, authorized=authorized)
    except ValueError:
        st.error(INVALID_PLATE)
    except DuplicatePlateError:
        st.error(f"A placa {display_plate(plate.upper())} já está cadastrada. Edite o cadastro existente.")
    else:
        flash(f"Veículo {display_plate(vehicle.plate)} cadastrado.")
        st.rerun()


def _render_register() -> None:
    with st.form("sg_vehicle_new", clear_on_submit=False):
        st.markdown("**Cadastrar veículo**")
        a, b = st.columns(2)
        plate = a.text_input("Placa *", max_chars=8, placeholder="ABC-1234 ou ABC1D23")
        owner = b.text_input("Proprietário", max_chars=80)
        c, d = st.columns(2)
        model = c.text_input("Modelo", max_chars=60)
        color = d.text_input("Cor", max_chars=30)
        authorized = st.checkbox("Autorizado a entrar", value=True)
        if st.form_submit_button("Cadastrar", type="primary", icon=":material/add:"):
            _save_new(plate, owner, model, color, authorized)


def _render_edit(vehicle: Vehicle) -> None:
    db = get_database()
    with st.form(f"sg_vehicle_edit_{vehicle.id}", border=True):
        st.markdown(f"**Editar {display_plate(vehicle.plate)}**")
        a, b = st.columns(2)
        plate = a.text_input("Placa *", value=display_plate(vehicle.plate), max_chars=8)
        owner = b.text_input("Proprietário", value=vehicle.owner, max_chars=80)
        c, d = st.columns(2)
        model = c.text_input("Modelo", value=vehicle.model, max_chars=60)
        color = d.text_input("Cor", value=vehicle.color, max_chars=30)
        authorized = st.checkbox("Autorizado a entrar", value=vehicle.authorized)
        save, remove = st.columns(2)
        saved = save.form_submit_button("Salvar alterações", type="primary", icon=":material/save:")
        removed = remove.form_submit_button("Remover veículo", icon=":material/delete:")
    if saved:
        try:
            updated = db.update_vehicle(vehicle.id, plate=plate, owner=owner, model=model, color=color,
                                        authorized=authorized)
        except ValueError:
            st.error(INVALID_PLATE)
        except DuplicatePlateError:
            st.error("Já existe outro veículo com essa placa.")
        except VehicleNotFoundError:
            st.error("Este veículo não existe mais.")
        else:
            flash(f"Cadastro de {display_plate(updated.plate)} atualizado.")
            st.rerun()
    if removed:
        st.session_state[STATE_CONFIRM_DELETE] = vehicle.id
    if st.session_state.get(STATE_CONFIRM_DELETE) == vehicle.id:
        _render_delete_confirmation(vehicle)


def _render_delete_confirmation(vehicle: Vehicle) -> None:
    with st.container(border=True):
        st.warning(f"Remover o veículo {display_plate(vehicle.plate)} ({vehicle.owner or 'sem proprietário'})? "
                   "Ele deixa de ser reconhecido na cancela. O histórico de acessos é mantido.")
        yes, no = st.columns(2)
        if yes.button("Confirmar remoção", type="primary", key="sg_vehicle_delete_yes"):
            try:
                get_database().delete_vehicle(vehicle.id)
            except VehicleNotFoundError:
                pass                                  # already removed: same final state
            st.session_state.pop(STATE_CONFIRM_DELETE, None)
            st.session_state.pop(STATE_EDIT, None)
            flash(f"Veículo {display_plate(vehicle.plate)} removido.")
            st.rerun()
        if no.button("Cancelar", key="sg_vehicle_delete_no"):
            st.session_state.pop(STATE_CONFIRM_DELETE, None)
            st.rerun()


def render() -> None:
    """Page function registered in ``app.py``."""
    st.header("Veículos")
    show_flash()
    _render_register()

    st.subheader("Veículos cadastrados")
    search = st.text_input("Pesquisar", placeholder="Placa, proprietário, modelo ou cor",
                           key="sg_vehicle_search")
    vehicles = get_database().list_vehicles(search=search)
    if not vehicles:
        st.info("Nenhum veículo encontrado." if search.strip() else "Nenhum veículo cadastrado ainda.")
        return
    st.caption(f"{len(vehicles)} veículo(s).")
    st.dataframe(vehicles_frame(vehicles), hide_index=True, width="stretch")

    by_id = {v.id: v for v in vehicles}
    if st.session_state.get(STATE_EDIT) not in by_id:
        st.session_state[STATE_EDIT] = vehicles[0].id
    selected = st.selectbox("Veículo para editar ou remover", list(by_id), key=STATE_EDIT,
                            format_func=lambda i: f"{display_plate(by_id[i].plate)} – {by_id[i].owner or 'sem proprietário'}")
    if st.session_state.get(STATE_CONFIRM_DELETE) not in (None, selected):
        st.session_state.pop(STATE_CONFIRM_DELETE)   # confirmation belongs to another vehicle
    _render_edit(by_id[selected])
