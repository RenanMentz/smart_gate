"""Smart Gate - Streamlit entry point.

Run with::

    streamlit run app.py

``st.navigation`` builds the multipage app; every page is a function in the
``interface`` package. The stylesheet and the brand bar are drawn here, so
they appear on every page.
"""

import streamlit as st

import config
from interface import access, styles

config.ensure_directories()
st.set_page_config(page_title="Smart Gate", page_icon=":material/garage_door:", layout="wide")

PAGES = [
    st.Page(access.render, title="Leitura de placas", icon=":material/photo_camera:",
            url_path="leitura", default=True),
]

styles.inject_css()
st.markdown(styles.brand_bar_html(), unsafe_allow_html=True)
st.navigation(PAGES).run()
