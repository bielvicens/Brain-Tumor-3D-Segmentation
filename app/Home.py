from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuració inicial de la pàgina
st.set_page_config(
    page_title="Brain Tumor 3D Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estils CSS avançats ------------------------------------------------
st.markdown(
    """
    <style>
        /* Ajustos generals d'espaiat */
        .main .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 1150px;
        }
        
        /* Títol principal en Blanc */
        .title-white {
            color: #ffffff; /* Color blanc */
            font-weight: 800;
            font-size: 3.2rem;
            margin-bottom: 0.5rem;
            letter-spacing: -1.5px;
        }

        /* Subtítol */
        .hero-subtitle {
            color: #64748b;
            font-size: 1.15rem;
            line-height: 1.6;
            margin-bottom: 3rem;
            font-weight: 400;
        }

        /* Targetes de funcionalitats */
        .feature-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.75rem 1.5rem;
            height: 100%;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            position: relative;
            overflow: hidden;
        }

        /* Efecte en passar el ratolí per sobre (Hover) */
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            border-color: #cbd5e1;
        }

        /* Línia de color superior en fer hover */
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .feature-card:hover::before {
            opacity: 1;
        }

        /* Contenidor de la icona */
        .icon-box {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 50px;
            height: 50px;
            background: #f1f5f9;
            border-radius: 12px;
            font-size: 1.6rem;
            margin-bottom: 1.25rem;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }

        /* Títol de la targeta (Negre) */
        .feature-card h4 {
            color: #000000; 
            font-weight: 700;
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
            letter-spacing: -0.25px;
            margin-top: 0;
        }

        /* Text descriptiu de la targeta */
        .feature-card p {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 0;
        }

        /* Ocultar elements per defecte de Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero Section (Capçalera) ---------------------------------------------
st.markdown('<h1 class="title-white">Brain Tumor 3D Segmentation</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Inference and analysis toolkit for a 3D U-Net '
    "trained on BraTS multi-modal MRI volumes.</p>",
    unsafe_allow_html=True,
)

# --- Funcionalitats (Targetes) ----------------------------------------
features = [
    ("🔍", "Predict", "Run the trained model on a patient and view the segmentation overlay."),
    ("📊", "Evaluate", "Score model checkpoints against ground-truth masks with Dice/IoU."),
    ("🗂️", "Explore", "Browse the dataset: patients, modalities, and volume statistics."),
    # S'ha afegit un <br> (salt de línia HTML) al final d'aquesta descripció
    ("📈", "Training results", "Inspect loss curves and metrics across training runs.<br><br>"),
]

columns = st.columns(4)
for column, (icon, name, description) in zip(columns, features):
    with column:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="icon-box">{icon}</div>
                <h4>{name}</h4>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Peu de pàgina / Instruccions -------------------------------------
st.markdown("<br>", unsafe_allow_html=True) # Un altre salt de línia per separar les targetes de la línia divisòria
st.divider()
st.markdown(
    "Use the pages in the sidebar to get started — the **Predict** page is "
    "the fastest way to see the model in action on a sample patient."
)

with st.sidebar:
    st.caption("Brain Tumor 3D Segmentation & Clinical Report Assistant")