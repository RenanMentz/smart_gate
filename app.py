"""Smart Gate - Streamlit entry point.

Run with::

    streamlit run app.py

``st.navigation`` builds the multipage app; every page is a function in the
``interface`` package. The stylesheet and the brand bar are drawn here, so
they appear on every page.
"""

import streamlit as st

import config
from interface import access, authorizations, dashboard, history, settings, styles, vehicles

config.ensure_directories()
st.set_page_config(page_title="Smart Gate", page_icon=":material/garage_door:", layout="wide")

PAGES = {
    "Operação": [
        st.Page(access.render, title="Leitura de placas", icon=":material/photo_camera:",
                url_path="leitura", default=True),
        st.Page(dashboard.render, title="Dashboard", icon=":material/monitoring:", url_path="dashboard"),
    ],
    "Cadastro": [
        st.Page(vehicles.render, title="Veículos", icon=":material/directions_car:", url_path="veiculos"),
        st.Page(authorizations.render, title="Autorizações", icon=":material/verified_user:",
                url_path="autorizacoes"),
    ],
    "Registros": [
        st.Page(history.render, title="Histórico", icon=":material/history:", url_path="historico"),
    ],
    "Sistema": [
        st.Page(settings.render, title="Configurações", icon=":material/settings:", url_path="configuracoes"),
    ],
}

styles.inject_css()
st.markdown(styles.brand_bar_html(), unsafe_allow_html=True)
st.navigation(PAGES).run()
