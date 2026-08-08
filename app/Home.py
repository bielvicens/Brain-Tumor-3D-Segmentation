from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importem les funcions de la interfície gràfica
from src.ui.theme import (
    inject_theme,
    render_hero,
    render_feature_grid,
    render_sidebar_brand
)

# Configuració inicial de la pàgina
st.set_page_config(
    page_title="Brain Tumor 3D Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Injecció del Tema -----------------------------------------------
# Això elimina la necessitat de tenir blocs <style> llargs al codi. 
# Aplica directament el fons fosc, tipografies i colors globals.
inject_theme()

# --- 2. Hero Section (Capçalera) ----------------------------------------
# Utilitzem la funció del tema perquè el títol quedi integrat visualment
render_hero(
    eyebrow="OVERVIEW",
    title="Brain Tumor 3D Segmentation",
    subtitle="Inference and analysis toolkit for a 3D U-Net trained on BraTS multi-modal MRI volumes."
)

# --- 3. Funcionalitats (Targetes) ---------------------------------------
# La teva funció `render_feature_grid` de theme.py requereix 4 valors:
# (etiqueta petita superior, icona, títol, descripció)
features = [
    ("INFERENCE", "🔍", "Predict", "Run the trained model on a patient and view the segmentation overlay."),
    ("METRICS", "📊", "Evaluate", "Score model checkpoints against ground-truth masks with Dice/IoU."),
    ("DATASET", "🗂️", "Explore", "Browse the dataset: patients, modalities, and volume statistics."),
    ("ANALYSIS", "📈", "Training results", "Inspect loss curves and metrics across training runs."),
]

# Donem una mica d'aire abans de les targetes i les renderitzem
st.markdown("<br>", unsafe_allow_html=True)
render_feature_grid(features)

# --- 4. Peu de pàgina / Instruccions ------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider() # Estilitzat automàticament pel tema (color de vora fosc)
st.markdown(
    "Use the pages in the sidebar to get started — the **Predict** page is "
    "the fastest way to see the model in action on a sample patient."
)

# --- 5. Barra Lateral ----------------------------------------------------
with st.sidebar:
    render_sidebar_brand(subtitle="Home")
    st.caption("Brain Tumor 3D Segmentation & Clinical Report Assistant")