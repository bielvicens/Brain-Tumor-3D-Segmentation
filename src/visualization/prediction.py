"""Visualization utilities for segmentation predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.info(
    "Prediction figure saved to '%s'.",
    output_path,
)


def plot_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray | None = None,
    *,
    slice_index: int | None = None,
    output_path: str | Path | None = None,
) -> Path | None:
    """Visualize a prediction on a single axial slice.

    Args:
        image:
            3D MRI volume (D,H,W) or (H,W,D).
        prediction:
            Predicted segmentation.
        ground_truth:
            Optional ground-truth segmentation.
        slice_index:
            Slice to visualize. If None, uses the middle slice.
        output_path:
            Optional path where the figure will be saved.
    """

    if image.ndim != 3:
        raise ValueError("image must be 3-dimensional.")

    if prediction.shape != image.shape:
        raise ValueError("prediction shape must match image.")

    if (
        ground_truth is not None
        and ground_truth.shape != image.shape
    ):
        raise ValueError(
            "ground_truth shape must match image."
        )

    if slice_index is None:
        slice_index = image.shape[2] // 2
    if not 0 <= slice_index < image.shape[2]:
        raise ValueError(
            f"slice_index must be between 0 and {image.shape[2]-1}."
        )

    image_slice = image[:, :, slice_index]
    pred_slice = prediction[:, :, slice_index]

    if ground_truth is None:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5),
        )

        axes[0].imshow(
            image_slice,
            cmap="gray",
        )
        axes[0].set_title("MRI")

        axes[1].imshow(
            image_slice,
            cmap="gray",
        )
        axes[1].imshow(
            pred_slice,
            alpha=0.5,
            cmap="jet",
        )
        axes[1].set_title("Prediction")

    else:

        gt_slice = ground_truth[:, :, slice_index]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )

        axes[0].imshow(
            image_slice,
            cmap="gray",
        )
        axes[0].set_title("MRI")

        axes[1].imshow(
            image_slice,
            cmap="gray",
        )
        axes[1].imshow(
            gt_slice,
            alpha=0.5,
            cmap="jet",
        )
        axes[1].set_title("Ground Truth")

        axes[2].imshow(
            image_slice,
            cmap="gray",
        )
        axes[2].imshow(
            pred_slice,
            alpha=0.5,
            cmap="jet",
        )
        axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        return output_path

    plt.close(fig)
    return None