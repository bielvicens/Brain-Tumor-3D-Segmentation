from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict import predict
from src.builders import build_model, build_pipeline
from src.data import BraTSReader, Modality
from src.inference import Predictor
from src.utils import ProjectConfig, load_checkpoint

# --- BraTS class legend (labels + a fixed, professional discrete palette) ---
# Class 0 (background) is intentionally excluded: it's masked out of the
# overlay entirely rather than tinted, so only the tumor sub-regions show.
CLASS_INFO = {
    1: ("NCR", "Necrotic core", "#e74c3c"),
    2: ("ED", "Edema", "#f1c40f"),
    3: ("ET", "Enhancing tumor", "#2ecc71"),
}
OVERLAY_CMAP = ListedColormap([CLASS_INFO[c][2] for c in sorted(CLASS_INFO)])

st.set_page_config(
    page_title="Predict · Brain Tumor 3D Segmentation",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; max-width: 1200px; }
        .legend-swatch {
            display: inline-block; width: 0.85rem; height: 0.85rem;
            border-radius: 3px; margin-right: 0.4rem; vertical-align: middle;
        }
        footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Cached resources -----------------------------------------------------
@st.cache_resource
def load_reader() -> tuple[BraTSReader, object]:
    """Build the BraTSReader and inference-time preprocessing pipeline once."""
    config = ProjectConfig()
    pipeline = build_pipeline(config, training=False)
    reader = BraTSReader(config.data.dataset_root)
    return reader, pipeline


@st.cache_data
def list_patient_ids(_reader: BraTSReader) -> list[str]:
    """List every patient available for prediction (cached; `_reader` is
    excluded from Streamlit's cache key since BraTSReader isn't hashable)."""
    return [patient.patient_id for patient in _reader.get_patients()]


@st.cache_resource
def load_predictor(checkpoint_name: str) -> Predictor:
    """Load the trained model for `checkpoint_name` only once per checkpoint."""
    config = ProjectConfig()
    model = build_model(config)
    checkpoint_path = Path(config.checkpoint.directory) / config.experiment.name / checkpoint_name
    load_checkpoint(path=checkpoint_path, model=model, map_location=config.training.device)
    return Predictor(model=model, device=config.training.device)


def list_available_checkpoints() -> list[str]:
    """Checkpoint filenames that actually exist on disk right now.

    Not cached: unlike the model/reader, this should always reflect the
    live filesystem (e.g. a training run may still be writing "best.pt").
    """
    config = ProjectConfig()
    checkpoint_dir = Path(config.checkpoint.directory) / config.experiment.name
    if not checkpoint_dir.is_dir():
        return []
    return sorted(path.name for path in checkpoint_dir.glob("*.pt"))


def _window_for_display(volume_slice: np.ndarray) -> np.ndarray:
    """Rescale a (possibly zero-mean, negative-valued) normalized MRI slice
    to [0, 1] for `st.image`/`imshow`, using percentile windowing so a few
    extreme outlier voxels don't wash out the whole image."""
    low, high = np.percentile(volume_slice, [1, 99])
    if high <= low:
        return np.zeros_like(volume_slice, dtype=np.float32)
    windowed = np.clip(volume_slice, low, high)
    return ((windowed - low) / (high - low)).astype(np.float32)


# --- Page header ------------------------------------------------------
st.title("🔍 Predict")
st.write("Run the trained 3D U-Net on a patient and inspect the segmentation.")

reader, pipeline = load_reader()
patient_ids = list_patient_ids(reader)
available_checkpoints = list_available_checkpoints()

with st.sidebar:
    st.header("Settings")

    if not patient_ids:
        st.error("No patients found under the configured dataset root.")
        st.stop()
    patient_id = st.selectbox("Patient", patient_ids)

    if not available_checkpoints:
        st.error("No checkpoints found yet. Train a model first.")
        st.stop()
    checkpoint = st.selectbox(
        "Checkpoint",
        available_checkpoints,
        index=available_checkpoints.index("best.pt") if "best.pt" in available_checkpoints else 0,
    )

    predict_button = st.button("Run prediction", type="primary", use_container_width=True)

if predict_button:
    try:
        with st.spinner("Running inference..."):
            predictor = load_predictor(checkpoint)
            prediction, probabilities, sample = predict(
                predictor=predictor,
                reader=reader,
                pipeline=pipeline,
                patient_id=patient_id,
            )
    except Exception as exc:  # surfaced as a clear message, not a raw traceback
        st.error(f"Prediction failed for patient '{patient_id}': {exc}")
        st.stop()

    st.session_state["prediction"] = prediction
    st.session_state["probabilities"] = probabilities
    st.session_state["sample"] = sample
    st.session_state["predicted_patient_id"] = patient_id

# --- Results ------------------------------------------------------------
if "prediction" in st.session_state and st.session_state.get("predicted_patient_id") == patient_id:
    prediction = st.session_state["prediction"]
    probabilities = st.session_state["probabilities"]
    sample = st.session_state["sample"]

    st.success(f"Inference completed for **{patient_id}**.")

    metric_columns = st.columns(3)
    metric_columns[0].metric("Prediction shape", " × ".join(str(d) for d in prediction.shape))
    metric_columns[1].metric("Probabilities shape", " × ".join(str(d) for d in probabilities.shape))
    tumor_voxels = int(np.sum(prediction > 0))
    metric_columns[2].metric("Predicted tumor voxels", f"{tumor_voxels:,}")

    st.divider()

    control_columns = st.columns([2, 1])
    with control_columns[0]:
        selected_modality = st.selectbox(
            "MRI modality",
            [Modality.T1N, Modality.T1C, Modality.T2W, Modality.T2F],
            format_func=lambda m: m.value.upper(),
        )
    with control_columns[1]:
        depth = prediction.shape[-1]
        slice_index = st.slider("Slice", min_value=0, max_value=depth - 1, value=depth // 2)

    image = sample.modalities[selected_modality]
    image_slice = _window_for_display(image[:, :, slice_index])
    prediction_slice = prediction[:, :, slice_index]

    display_columns = st.columns(3)

    with display_columns[0]:
        st.subheader(selected_modality.value.upper())
        st.image(image_slice, clamp=True, use_container_width=True)

    with display_columns[1]:
        st.subheader("Segmentation overlay")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image_slice, cmap="gray")
        masked_prediction = np.ma.masked_where(prediction_slice == 0, prediction_slice)
        ax.imshow(masked_prediction, cmap=OVERLAY_CMAP, vmin=1, vmax=3, alpha=0.55)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        legend_html = " &nbsp; ".join(
            f'<span class="legend-swatch" style="background:{color}"></span>{code} ({label})'
            for code, label, color in CLASS_INFO.values()
        )
        st.markdown(legend_html, unsafe_allow_html=True)

    with display_columns[2]:
        class_options = list(CLASS_INFO.keys())
        probability_class = st.selectbox(
            "Probability map for class",
            class_options,
            format_func=lambda c: f"{CLASS_INFO[c][0]} ({CLASS_INFO[c][1]})",
        )
        st.subheader(f"P({CLASS_INFO[probability_class][0]})")
        fig, ax = plt.subplots(figsize=(5, 5))
        prob_map = ax.imshow(
            probabilities[probability_class, :, :, slice_index], cmap="viridis", vmin=0, vmax=1
        )
        ax.axis("off")
        fig.colorbar(prob_map, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
else:
    st.info("Select a patient and checkpoint in the sidebar, then run a prediction.")