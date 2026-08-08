from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Training Results · Brain Tumor 3D Segmentation",
    page_icon="📈",
    layout="wide",
)

# Injecció del tema (elimina l'estil CSS manual antic)
inject_theme()


# ---------------------------------------------------------------------
# Matplotlib Dark Theme Configuration
# ---------------------------------------------------------------------
# Configuració perquè els gràfics s'integrin al fons fosc de l'aplicació
plt.rcParams['figure.facecolor'] = '#0a0f1a'
plt.rcParams['axes.facecolor'] = '#0a0f1a'
plt.rcParams['text.color'] = '#94a3b8'
plt.rcParams['axes.labelcolor'] = '#94a3b8'
plt.rcParams['xtick.color'] = '#94a3b8'
plt.rcParams['ytick.color'] = '#94a3b8'
plt.rcParams['axes.edgecolor'] = '#334155'


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_experiment_directories(checkpoint_root: Path) -> list[Path]:
    """Return experiment directories containing training artifacts."""

    if not checkpoint_root.is_dir():
        return []

    return sorted(
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir()
    )


def load_history(history_path: Path) -> dict:
    """Load a training history JSON file."""

    with history_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_best_loss_epoch(val_loss: list[float]) -> int | None:
    """Return the 1-based epoch with the lowest validation loss."""

    if not val_loss:
        return None

    return min(
        range(len(val_loss)),
        key=lambda index: val_loss[index],
    ) + 1


def find_best_dice_epoch(val_dice: list[float]) -> int | None:
    """Return the 1-based epoch with the highest validation Dice."""

    if not val_dice:
        return None

    return max(
        range(len(val_dice)),
        key=lambda index: val_dice[index],
    ) + 1


def plot_training_history(history: dict) -> tuple[plt.Figure, plt.Figure]:
    """Create loss and Dice plots from the training history."""

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    train_dice = history.get("train_dice", [])
    val_dice = history.get("val_dice", [])

    epochs_loss = range(1, len(train_loss) + 1)
    epochs_val_loss = range(1, len(val_loss) + 1)

    epochs_dice = range(1, len(train_dice) + 1)
    epochs_val_dice = range(1, len(val_dice) + 1)

    # --------------------------------------------------------------
    # Loss
    # --------------------------------------------------------------

    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))

    loss_ax.plot(
        epochs_loss,
        train_loss,
        label="Train",
        linewidth=2,
        color="#3b82f6",
    )

    if val_loss:
        loss_ax.plot(
            epochs_val_loss,
            val_loss,
            label="Validation",
            linewidth=2,
            color="#a78bfa",
        )

    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training and Validation Loss", color="#f8fafc", pad=12)
    loss_ax.grid(True, alpha=0.1, color="#94a3b8")
    loss_ax.legend(facecolor="#0a0f1a", edgecolor="#334155")
    
    loss_ax.spines['top'].set_visible(False)
    loss_ax.spines['right'].set_visible(False)

    loss_fig.tight_layout()

    # --------------------------------------------------------------
    # Dice
    # --------------------------------------------------------------

    dice_fig, dice_ax = plt.subplots(figsize=(8, 5))

    dice_ax.plot(
        epochs_dice,
        train_dice,
        label="Train",
        linewidth=2,
        color="#3b82f6",
    )

    if val_dice:
        dice_ax.plot(
            epochs_val_dice,
            val_dice,
            label="Validation",
            linewidth=2,
            color="#a78bfa",
        )

    dice_ax.set_xlabel("Epoch")
    dice_ax.set_ylabel("Dice")
    dice_ax.set_title("Training and Validation Dice", color="#f8fafc", pad=12)
    dice_ax.set_ylim(0, 1)
    dice_ax.grid(True, alpha=0.1, color="#94a3b8")
    dice_ax.legend(facecolor="#0a0f1a", edgecolor="#334155")

    dice_ax.spines['top'].set_visible(False)
    dice_ax.spines['right'].set_visible(False)

    dice_fig.tight_layout()

    return loss_fig, dice_fig


# ---------------------------------------------------------------------
# Header (Estilitzat)
# ---------------------------------------------------------------------

render_page_header(
    eyebrow="MODEL TRAINING",
    icon="",
    title="Training Results",
    subtitle="Inspect the training history, validation performance and saved checkpoints of the 3D U-Net."
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

config = ProjectConfig()

checkpoint_root = (
    Path(config.checkpoint.directory)
)

experiments = get_experiment_directories(checkpoint_root)


# ---------------------------------------------------------------------
# Sidebar (Estilitzat)
# ---------------------------------------------------------------------

with st.sidebar:
    render_sidebar_brand(subtitle="Training Results")
    render_section_label("Training run")

    if not experiments:
        st.error(
            "No training experiments were found under "
            f"`{checkpoint_root}`."
        )
        st.stop()

    experiment_names = [path.name for path in experiments]

    default_experiment = config.experiment.name

    if default_experiment in experiment_names:
        default_index = experiment_names.index(default_experiment)
    else:
        default_index = 0

    selected_experiment_name = st.selectbox(
        "Experiment",
        experiment_names,
        index=default_index,
    )


experiment_dir = checkpoint_root / selected_experiment_name

history_path = experiment_dir / "history.json"
best_checkpoint = experiment_dir / "best.pt"
last_checkpoint = experiment_dir / "last.pt"

available_checkpoint_files = sorted(
    path.name
    for path in experiment_dir.glob("*.pt")
)


# ---------------------------------------------------------------------
# Validate history
# ---------------------------------------------------------------------

if not history_path.exists():
    st.warning(
        f"No `history.json` was found for experiment "
        f"`{selected_experiment_name}`."
    )

    if available_checkpoint_files:
        st.info(
            "Checkpoints were found, but the training history is not "
            "available yet."
        )

        st.write("Available checkpoints:")

        for checkpoint in available_checkpoint_files:
            st.code(checkpoint)

    st.stop()


history = load_history(history_path)

train_loss = history.get("train_loss", [])
val_loss = history.get("val_loss", [])
train_dice = history.get("train_dice", [])
val_dice = history.get("val_dice", [])


if not train_loss:
    st.warning("The training history does not contain any epochs.")
    st.stop()


# ---------------------------------------------------------------------
# Basic information
# ---------------------------------------------------------------------

num_epochs = len(train_loss)

best_loss_epoch = find_best_loss_epoch(val_loss)
best_dice_epoch = find_best_dice_epoch(val_dice)

best_val_loss = (
    val_loss[best_loss_epoch - 1]
    if best_loss_epoch is not None
    else None
)

best_val_dice = (
    val_dice[best_dice_epoch - 1]
    if best_dice_epoch is not None
    else None
)

final_train_loss = train_loss[-1]
final_train_dice = train_dice[-1]

final_val_loss = val_loss[-1] if val_loss else None
final_val_dice = val_dice[-1] if val_dice else None


# ---------------------------------------------------------------------
# Run status (Estilitzat amb render_metric_row)
# ---------------------------------------------------------------------

render_section_heading("Run overview")

render_metric_row([
    ("Epochs completed", str(num_epochs)),
    ("Best validation loss", f"{best_val_loss:.4f}" if best_val_loss is not None else "—"),
    ("Best validation Dice", f"{best_val_dice:.4f}" if best_val_dice is not None else "—"),
    ("Best Dice epoch", str(best_dice_epoch) if best_dice_epoch is not None else "—")
])

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Checkpoint status
# ---------------------------------------------------------------------

render_section_heading("Checkpoints")

checkpoint_columns = st.columns(3)

with checkpoint_columns[0]:
    if best_checkpoint.exists():
        st.success("✓ best.pt available")
    else:
        st.warning("best.pt not found")

with checkpoint_columns[1]:
    if last_checkpoint.exists():
        st.success("✓ last.pt available")
    else:
        st.warning("last.pt not found")

with checkpoint_columns[2]:
    st.info(
        f"{len(available_checkpoint_files)} checkpoint file(s)"
    )


# ---------------------------------------------------------------------
# Final metrics (Estilitzat amb render_metric_row)
# ---------------------------------------------------------------------

render_ruler()
render_section_heading("Final metrics")

render_metric_row([
    ("Train loss", f"{final_train_loss:.4f}"),
    ("Validation loss", f"{final_val_loss:.4f}" if final_val_loss is not None else "—"),
    ("Train Dice", f"{final_train_dice:.4f}"),
    ("Validation Dice", f"{final_val_dice:.4f}" if final_val_dice is not None else "—")
])

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Best epoch analysis (Estilitzat amb render_stat_grid)
# ---------------------------------------------------------------------

render_ruler()
render_section_heading("Best validation performance")

best_columns = st.columns(2)

with best_columns[0]:
    if best_loss_epoch is not None:
        render_stat_grid([
            ("Lowest validation loss", f"{best_val_loss:.4f}", "#ef4444", f"Epoch {best_loss_epoch}")
        ])
    else:
        st.info("Validation loss is not available.")

with best_columns[1]:
    if best_dice_epoch is not None:
        render_stat_grid([
            ("Highest validation Dice", f"{best_val_dice:.4f}", "#10b981", f"Epoch {best_dice_epoch}")
        ])
    else:
        st.info("Validation Dice is not available.")


# ---------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------

render_ruler()
render_section_heading("Training curves")

loss_fig, dice_fig = plot_training_history(history)

curve_columns = st.columns(2)

with curve_columns[0]:
    st.pyplot(
        loss_fig,
        use_container_width=True,
    )

with curve_columns[1]:
    st.pyplot(
        dice_fig,
        use_container_width=True,
    )

plt.close(loss_fig)
plt.close(dice_fig)


# ---------------------------------------------------------------------
# Epoch-by-epoch data
# ---------------------------------------------------------------------

render_ruler()
render_section_heading("Epoch-by-epoch metrics")

rows = []

for index in range(num_epochs):

    row = {
        "Epoch": index + 1,
        "Train Loss": train_loss[index],
        "Train Dice": train_dice[index],
    }

    if index < len(val_loss):
        row["Validation Loss"] = val_loss[index]

    if index < len(val_dice):
        row["Validation Dice"] = val_dice[index]

    rows.append(row)


st.dataframe(
    rows,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# Raw history
# ---------------------------------------------------------------------

render_ruler()

with st.expander("View raw history.json"):
    st.json(history)


# ---------------------------------------------------------------------
# Sidebar information
# ---------------------------------------------------------------------

with st.sidebar:

    st.divider()

    render_section_label("Experiment directory")

    st.code(
        str(experiment_dir),
        language="text",
    )

    render_section_label("Saved files")

    files = sorted(
        path.name
        for path in experiment_dir.iterdir()
        if path.is_file()
    )

    if files:
        for file_name in files:
            st.write(f"• {file_name}")
    else:
        st.write("No files found.")