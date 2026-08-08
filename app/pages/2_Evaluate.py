from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.builders import (
    build_dataloader,
    build_datasets,
    build_model,
    build_pipeline,
)
# --- NOU: Importem totes les funcions del tema ---
from src.ui.theme import (
    inject_theme,
    render_page_header,
    render_sidebar_brand,
    render_section_label,
    render_section_heading,
    render_status_banner,
    render_metric_row,
    render_stat_grid,
    render_ruler
)
from src.inference import Predictor
from src.utils import ProjectConfig, load_checkpoint
from src.utils.metrics import mean_dice, mean_iou


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Evaluate · Brain Tumor 3D Segmentation",
    page_icon="📊",
    layout="wide",
)

# Injecció del tema global (substitueix el bloc CSS manual)
inject_theme()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NOU: Afegim els colors del tema ("#ef4444", "#eab308", "#a78bfa") 
# perquè es mostrin a les targetes de mètriques.
CLASS_INFO = {
    1: ("NCR", "Necrotic core", "#ef4444"),
    2: ("ED", "Edema", "#eab308"),
    3: ("ET", "Enhancing tumor", "#a78bfa"),
}


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
# (Les funcions load_validation_dataset, load_predictor i 
# list_available_checkpoints es mantenen exactament igual)

@st.cache_resource
def load_validation_dataset():
    config = ProjectConfig()
    train_pipeline = build_pipeline(config, training=True)
    validation_pipeline = build_pipeline(config, training=False)
    _, validation_dataset = build_datasets(config, train_pipeline, validation_pipeline)
    validation_loader = build_dataloader(validation_dataset, config)
    return validation_dataset, validation_loader


@st.cache_resource
def load_predictor(checkpoint_name: str) -> Predictor:
    config = ProjectConfig()
    model = build_model(config)
    checkpoint_path = Path(config.checkpoint.directory) / config.experiment.name / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    load_checkpoint(path=checkpoint_path, model=model, map_location=config.training.device)
    return Predictor(model=model, device=config.training.device)


def list_available_checkpoints() -> list[str]:
    config = ProjectConfig()
    checkpoint_dir = Path(config.checkpoint.directory) / config.experiment.name

    if not checkpoint_dir.is_dir():
        return []

    return sorted(path.name for path in checkpoint_dir.glob("*.pt"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    predictor: Predictor,
    dataloader,
) -> tuple[float, float, dict[int, float], dict[int, float]]:
    
    predictor.model.eval()

    total_dice = 0.0
    total_iou = 0.0

    dice_per_class = {class_id: 0.0 for class_id in CLASS_INFO}
    iou_per_class = {class_id: 0.0 for class_id in CLASS_INFO}

    num_batches = 0

    for images, masks in dataloader:
        images = images.to(predictor.device, dtype=torch.float32)
        masks = masks.to(predictor.device, dtype=torch.long)

        logits = predictor.model(images)
        predictions = torch.argmax(logits, dim=1)

        dice = mean_dice(prediction=predictions, target=masks, num_classes=logits.shape[1], include_background=False)
        iou = mean_iou(prediction=predictions, target=masks, num_classes=logits.shape[1], include_background=False)

        total_dice += float(dice.item())
        total_iou += float(iou.item())

        for class_id in CLASS_INFO:
            class_prediction = (predictions == class_id).long()
            class_target = (masks == class_id).long()

            class_dice = mean_dice(prediction=class_prediction, target=class_target, num_classes=2, include_background=False)
            class_iou = mean_iou(prediction=class_prediction, target=class_target, num_classes=2, include_background=False)

            dice_per_class[class_id] += float(class_dice.item())
            iou_per_class[class_id] += float(class_iou.item())

        num_batches += 1

    if num_batches == 0:
        raise ValueError("Cannot evaluate an empty validation dataset.")

    mean_dice_value = total_dice / num_batches
    mean_iou_value = total_iou / num_batches
    dice_per_class = {class_id: value / num_batches for class_id, value in dice_per_class.items()}
    iou_per_class = {class_id: value / num_batches for class_id, value in iou_per_class.items()}

    return mean_dice_value, mean_iou_value, dice_per_class, iou_per_class


# ---------------------------------------------------------------------------
# Page header (Canviat per utilitzar render_page_header)
# ---------------------------------------------------------------------------

render_page_header(
    eyebrow="METRICS",
    icon="",
    title="Evaluate",
    subtitle="Evaluate a trained 3D U-Net checkpoint on the validation dataset using Dice and IoU segmentation metrics."
)


# ---------------------------------------------------------------------------
# Sidebar (Estilitzat)
# ---------------------------------------------------------------------------

available_checkpoints = list_available_checkpoints()

with st.sidebar:
    render_sidebar_brand(subtitle="Evaluate")
    render_section_label("Evaluation settings")

    if not available_checkpoints:
        st.error("No checkpoints found. Train a model first.")
        st.stop()

    checkpoint = st.selectbox(
        "Checkpoint",
        available_checkpoints,
        index=(available_checkpoints.index("best.pt") if "best.pt" in available_checkpoints else 0),
    )

    evaluate_button = st.button("Run evaluation", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Dataset information (Estilitzat amb render_metric_row)
# ---------------------------------------------------------------------------

try:
    validation_dataset, validation_loader = load_validation_dataset()
except Exception as exc:
    st.error(f"Could not load the validation dataset: {exc}")
    st.stop()

render_section_heading("Validation dataset", meta="Configuration")

render_metric_row([
    ("Validation patients", str(len(validation_dataset))),
    ("Validation batches", str(len(validation_loader))),
    ("Batch size", str(validation_loader.batch_size))
])

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

if evaluate_button:
    try:
        with st.spinner(f"Evaluating {checkpoint} on the validation dataset..."):
            predictor = load_predictor(checkpoint)
            (
                mean_dice_value,
                mean_iou_value,
                dice_per_class,
                iou_per_class,
            ) = evaluate_model(predictor=predictor, dataloader=validation_loader)
            
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        st.stop()

    st.session_state["evaluation_checkpoint"] = checkpoint
    st.session_state["mean_dice"] = mean_dice_value
    st.session_state["mean_iou"] = mean_iou_value
    st.session_state["dice_per_class"] = dice_per_class
    st.session_state["iou_per_class"] = iou_per_class


# ---------------------------------------------------------------------------
# Results (Molt estilitzat per la Workstation)
# ---------------------------------------------------------------------------

if "mean_dice" in st.session_state:
    
    evaluated_checkpoint = st.session_state["evaluation_checkpoint"]
    dice_per_class = st.session_state["dice_per_class"]
    iou_per_class = st.session_state["iou_per_class"]

    render_ruler()
    
    # Banner d'èxit del tema
    render_status_banner(
        "success", 
        f"Evaluation completed using <strong>{evaluated_checkpoint}</strong>."
    )

    # Rendiment general
    render_section_heading("Overall performance", meta="Averages")
    render_stat_grid([
        ("Mean Dice", f"{st.session_state['mean_dice']:.4f}", None, "Across all regions"),
        ("Mean IoU", f"{st.session_state['mean_iou']:.4f}", None, "Across all regions")
    ])
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Rendiment per classe (generant llistes pel Grid)
    render_section_heading("Performance by tumor region")
    
    # Construïm les dades de les targetes passant-hi el color per pintar la barra lateral
    dice_cards = []
    iou_cards = []
    
    for class_id, (code, desc, color) in CLASS_INFO.items():
        dice_cards.append((f"{code} Dice", f"{dice_per_class[class_id]:.4f}", color, desc))
        iou_cards.append((f"{code} IoU", f"{iou_per_class[class_id]:.4f}", color, desc))

    st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)
    render_section_label("DICE SCORES")
    render_stat_grid(dice_cards)

    st.markdown("<div style='margin-bottom:0.8rem; margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    render_section_label("INTERSECTION OVER UNION (IOU)")
    render_stat_grid(iou_cards)

    render_ruler(compact=True)

    # Taula resum final
    render_section_heading("Metrics summary")
    
    summary_data = {
        "Region": [CLASS_INFO[c][0] for c in CLASS_INFO],
        "Description": [CLASS_INFO[c][1] for c in CLASS_INFO],
        "Dice": [dice_per_class[c] for c in CLASS_INFO],
        "IoU": [iou_per_class[c] for c in CLASS_INFO],
    }

    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True,
    )

else:
    # Banner informatiu si no s'ha premut el botó
    st.markdown("<br>", unsafe_allow_html=True)
    render_status_banner(
        "info", 
        "Select a checkpoint and click <strong>Run evaluation</strong> to evaluate the model."
    )