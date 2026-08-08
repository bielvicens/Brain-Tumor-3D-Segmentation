from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data import BraTSReader, Modality
from src.utils import ProjectConfig

# --- NOU: Importem les funcions del tema ---
from src.ui.theme import (
    inject_theme,
    render_page_header,
    render_sidebar_brand,
    render_section_heading,
    render_section_label,
    render_metric_row,
    render_stat_grid,
    render_ruler
)


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Explore · Brain Tumor 3D Segmentation",
    page_icon="🗂️",
    layout="wide",
)

# Injecció del tema (elimina la necessitat del bloc CSS)
inject_theme()


# ---------------------------------------------------------------------------
# Matplotlib Dark Theme Configuration
# ---------------------------------------------------------------------------
# Ajustem matplotlib perquè els gràfics s'integrin al fons de l'aplicació
plt.rcParams['figure.facecolor'] = '#0a0f1a'
plt.rcParams['axes.facecolor'] = '#0a0f1a'
plt.rcParams['text.color'] = '#94a3b8'
plt.rcParams['axes.labelcolor'] = '#94a3b8'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['axes.edgecolor'] = '#334155'


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
# (Les funcions load_reader, get_patient_ids, etc., es mantenen igual)

@st.cache_resource
def load_reader() -> BraTSReader:
    config = ProjectConfig()
    return BraTSReader(config.data.dataset_root)

@st.cache_data
def get_patient_ids() -> list[str]:
    reader = load_reader()
    return [patient.patient_id for patient in reader.get_patients()]

@st.cache_data
def load_patient_modalities(patient_id: str) -> dict[Modality, np.ndarray]:
    reader = load_reader()
    patient = reader.get_patient(patient_id)
    return reader.load_modalities(patient)

@st.cache_data
def load_patient_segmentation(patient_id: str) -> np.ndarray:
    reader = load_reader()
    patient = reader.get_patient(patient_id)
    return reader.load_segmentation(patient)

@st.cache_data
def get_patient_metadata(patient_id: str) -> dict[Modality, object]:
    reader = load_reader()
    patient = reader.get_patient(patient_id)
    metadata = {}
    for modality in patient.available_modalities:
        metadata[modality] = reader.get_metadata(patient, modality)
    return metadata


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def window_for_display(volume_slice: np.ndarray) -> np.ndarray:
    low, high = np.percentile(volume_slice, [1, 99])
    if high <= low:
        return np.zeros_like(volume_slice, dtype=np.float32)
    windowed = np.clip(volume_slice, low, high)
    return ((windowed - low) / (high - low)).astype(np.float32)

def format_shape(shape: tuple[int, ...]) -> str:
    return " × ".join(str(int(value)) for value in shape)

def format_spacing(spacing: tuple[float, ...]) -> str:
    return " × ".join(f"{float(value):.2f}" for value in spacing) + " mm"

def modality_label(modality: Modality) -> str:
    labels = {
        Modality.T1N: "T1",
        Modality.T1C: "T1c",
        Modality.T2W: "T2",
        Modality.T2F: "FLAIR",
        Modality.SEG: "Segmentation",
    }
    return labels.get(modality, modality.value.upper())

def calculate_volume_statistics(volume: np.ndarray) -> dict[str, float]:
    finite_values = volume[np.isfinite(volume)]
    if finite_values.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
    }


# ---------------------------------------------------------------------------
# Header (Estilitzat)
# ---------------------------------------------------------------------------

render_page_header(
    eyebrow="DATASET",
    icon="",
    title="Explore",
    subtitle="Explore BraTS patients, MRI modalities, volume metadata, voxel statistics and ground-truth segmentations."
)


# ---------------------------------------------------------------------------
# Load patients
# ---------------------------------------------------------------------------

reader = load_reader()
patient_ids = get_patient_ids()

if not patient_ids:
    st.error("No valid patients were found under the configured BraTS dataset root.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar (Estilitzat)
# ---------------------------------------------------------------------------

with st.sidebar:
    render_sidebar_brand(subtitle="Explore")
    render_section_label("Dataset selection")

    patient_id = st.selectbox("Patient", patient_ids)
    patient = reader.get_patient(patient_id)

    available_modalities = [
        modality for modality in patient.available_modalities if modality != Modality.SEG
    ]

    if not available_modalities:
        st.error("No MRI modalities are available for this patient.")
        st.stop()

    selected_modality = st.selectbox(
        "MRI modality",
        available_modalities,
        format_func=modality_label,
    )

    show_segmentation = st.checkbox("Show ground-truth segmentation", value=True)


# ---------------------------------------------------------------------------
# Patient overview (Estilitzat)
# ---------------------------------------------------------------------------

render_section_heading("Patient overview", meta=patient_id)

render_metric_row([
    ("Patient ID", patient.patient_id),
    ("MRI modalities", str(len(available_modalities))),
    ("Complete case", "Yes" if patient.is_complete else "No"),
    ("Missing modalities", str(len(patient.missing_modalities)))
])

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Patient file information
# ---------------------------------------------------------------------------

with st.expander("View patient source files", expanded=False):
    st.markdown(f"<span style='color: #94a3b8; font-size: 0.9rem;'><strong>Directory:</strong> {patient.case_dir}</span>", unsafe_allow_html=True)
    file_rows = []
    for modality in patient.available_modalities:
        file_rows.append({
            "Modality": modality_label(modality),
            "File": patient.path_for(modality).name,
        })
    if file_rows:
        st.dataframe(file_rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Metadata (Estilitzat)
# ---------------------------------------------------------------------------

render_ruler()
render_section_heading("Volume metadata")

metadata = get_patient_metadata(patient_id)
metadata_rows = []

for modality in available_modalities:
    if modality not in metadata:
        continue
    info = metadata[modality]
    metadata_rows.append({
        "Modality": modality_label(modality),
        "Shape": format_shape(info.shape),
        "Voxel spacing": format_spacing(info.voxel_spacing),
        "Data type": str(info.dtype),
    })

if metadata_rows:
    st.dataframe(metadata_rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Load selected modality & Volume information
# ---------------------------------------------------------------------------

modalities = load_patient_modalities(patient_id)
volume = modalities[selected_modality]

render_ruler()
render_section_heading("Volume information", meta=modality_label(selected_modality))

stats = calculate_volume_statistics(volume)

# Dividim en dues línies perquè 6 mètriques no quedin apretades en pantalles petites
render_metric_row([
    ("Dimensions", format_shape(volume.shape)),
    ("Voxels", f"{volume.size:,}"),
    ("Mean", f"{stats['mean']:.3f}")
])

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

render_metric_row([
    ("Minimum", f"{stats['min']:.3f}"),
    ("Maximum", f"{stats['max']:.3f}"),
    ("Std Dev", f"{stats['std']:.3f}")
])


# ---------------------------------------------------------------------------
# Slice viewer
# ---------------------------------------------------------------------------

render_ruler()
render_section_heading(f"{modality_label(selected_modality)} viewer", meta="Axial Plane")

depth = volume.shape[-1]
slice_index = st.slider(
    "Axial slice",
    min_value=0,
    max_value=depth - 1,
    value=depth // 2,
)

image_slice = volume[:, :, slice_index]
display_image = window_for_display(image_slice)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

segmentation = None
if show_segmentation:
    try:
        segmentation = load_patient_segmentation(patient_id)
    except Exception as exc:
        st.warning(f"Could not load ground-truth segmentation: {exc}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

if segmentation is not None:
    display_columns = st.columns(2)

    with display_columns[0]:
        render_section_label(f"MRI — {modality_label(selected_modality)}")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(display_image, cmap="gray")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with display_columns[1]:
        render_section_label("GROUND-TRUTH SEGMENTATION")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(display_image, cmap="gray")
        
        mask = np.ma.masked_where(
            segmentation[:, :, slice_index] == 0,
            segmentation[:, :, slice_index],
        )
        
        ax.imshow(
            mask,
            cmap="viridis",
            vmin=1,
            vmax=3,
            alpha=0.55,
        )
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

else:
    render_section_label(f"MRI — {modality_label(selected_modality)}")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(display_image, cmap="gray")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Segmentation statistics
# ---------------------------------------------------------------------------

if segmentation is not None:
    render_ruler()
    render_section_heading("Ground-truth tumor statistics")

    segmentation_values, segmentation_counts = np.unique(
        segmentation,
        return_counts=True,
    )
    total_voxels = segmentation.size
    tumor_rows = []

    # NOU: Afegim els colors de les classes per si vols estilitzar-ho en el futur
    class_names = {
        0: ("Background", None),
        1: ("NCR", "#ef4444"),
        2: ("Edema", "#eab308"),
        3: ("Enhancing tumor", "#a78bfa"),
    }

    for value, count in zip(segmentation_values, segmentation_counts):
        value = int(value)
        count = int(count)
        name, _ = class_names.get(value, (f"Class {value}", None))
        
        tumor_rows.append({
            "Class": name,
            "Label": value,
            "Voxels": f"{count:,}",
            "Percentage": f"{100.0 * count / total_voxels:.2f}%",
        })

    st.dataframe(tumor_rows, use_container_width=True, hide_index=True)

    tumor_voxels = int(np.sum(segmentation > 0))
    tumor_percentage = 100.0 * tumor_voxels / total_voxels

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Utilitzem targetes d'estat per donar més importància a aquestes dades
    render_stat_grid([
        ("Total Tumor Voxels", f"{tumor_voxels:,}", "#10b981", None),
        ("Tumor Percentage", f"{tumor_percentage:.3f}%", "#10b981", "Relative to entire scan volume")
    ])


# ---------------------------------------------------------------------------
# Intensity distribution
# ---------------------------------------------------------------------------

render_ruler()
render_section_heading("Intensity distribution", meta=modality_label(selected_modality))

hist_column, stats_column = st.columns([2, 1], gap="large")

with hist_column:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    finite_values = volume[np.isfinite(volume)]
    max_samples = 250_000

    if finite_values.size > max_samples:
        rng = np.random.default_rng(42)
        finite_values = rng.choice(finite_values, size=max_samples, replace=False)

    # Color de l'histograma adaptat al tema
    ax.hist(finite_values, bins=100, color="#3b82f6", alpha=0.8, edgecolor="#2563eb")

    ax.set_xlabel("Voxel intensity")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.1, color="#94a3b8")
    
    # Amagem les vores superior i dreta per un disseny més net
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with stats_column:
    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    render_section_label("INTENSITY STATS")
    
    # Utilitzem el Grid compacte vertical
    render_stat_grid([
        ("Mean", f"{stats['mean']:.4f}", None, None),
        ("Standard dev.", f"{stats['std']:.4f}", None, None),
        ("Minimum", f"{stats['min']:.4f}", None, None),
        ("Maximum", f"{stats['max']:.4f}", None, None)
    ])