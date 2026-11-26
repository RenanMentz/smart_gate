"""Shared, cached objects of the interface.

Streamlit re-runs the script on every click. ``st.cache_resource`` keeps one
instance for the whole server, so the YOLO and EasyOCR models (loaded lazily by
the detector and the OCR on the first reading) are never reloaded and the
database object is created once.
"""

from __future__ import annotations

import streamlit as st

import config
from database.database import Database
from modules.pipeline import RecognitionPipeline


@st.cache_resource(show_spinner=False)
def get_database() -> Database:
    config.ensure_directories()
    return Database(config.DATABASE_PATH)


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RecognitionPipeline:
    return RecognitionPipeline(get_database(), captures_dir=config.CAPTURES_DIR)
