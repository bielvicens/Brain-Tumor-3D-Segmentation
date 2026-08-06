"""Prediction entry point for the BraTS 3D U-Net project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.builders import (
    build_model,
    build_pipeline,
)
from src.data import (
    BraTSReader,
    MRI_MODALITIES,
)
from src.inference import Predictor
from src.preprocessing import PreprocessingSample
from src.utils import (
    ProjectConfig,
    load_checkpoint,
)
from src.visualization.prediction import plot_prediction, plot_probabilities


def load_patient_sample(
    reader: BraTSReader,
    patient_id: str,
    pipeline,
) -> PreprocessingSample:
    """Load and preprocess one BraTS patient."""

    patient = reader.get_patient(patient_id)

    modalities = reader.load_modalities(patient)
    segmentation = None

    metadata = reader.get_metadata(
        patient,
        MRI_MODALITIES[0],
    )

    sample = PreprocessingSample(
        patient_id=patient.patient_id,
        modalities=modalities,
        segmentation=segmentation,
        voxel_spacing=metadata.voxel_spacing,
        affine=metadata.affine,
    )

    return pipeline.run(sample)


def sample_to_tensor(
    sample: PreprocessingSample,
) -> torch.Tensor:
    """Convert a preprocessing sample into a model input tensor."""

    image = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    sample.modalities[modality],
                    dtype=np.float32,
                )
            )
            for modality in MRI_MODALITIES
        ],
        dim=0,
    )

    return image.unsqueeze(0)


def predict(
    predictor: Predictor,
    reader: BraTSReader,
    pipeline,
    patient_id: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    PreprocessingSample,
]:
    """Predict the segmentation mask for one patient."""

    sample = load_patient_sample(
        reader=reader,
        patient_id=patient_id,
        pipeline=pipeline,
    )

    image = sample_to_tensor(sample)

    prediction, probabilities = (
        predictor.predict_with_probabilities(
            image,
        )
    )

    return (
        prediction.squeeze(0).cpu().numpy(),
        probabilities.squeeze(0).cpu().numpy(),
        sample,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run inference with a trained BraTS 3D U-Net.",
    )

    parser.add_argument(
        "--patient",
        required=True,
        help="BraTS patient ID.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to the project's best checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions"),
        help="Directory where predictions are saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Run inference."""

    args = parse_args()

    config = ProjectConfig()

    pipeline = build_pipeline(
        config,
        training=False,
    )

    reader = BraTSReader(
        config.data.dataset_root,
    )

    model = build_model(config)

    checkpoint_path = (
        args.checkpoint
        if args.checkpoint is not None
        else Path(config.checkpoint.directory)
        / config.experiment.name
        / config.checkpoint.best_model_name
    )

    load_checkpoint(
        path=checkpoint_path,
        model=model,
        map_location=config.training.device,
    )

    predictor = Predictor(
        model=model,
        device=config.training.device,
    )

    prediction, probabilities, sample = predict(
        predictor=predictor,
        reader=reader,
        pipeline=pipeline,
        patient_id=args.patient,
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output
        / f"{args.patient}_prediction.npy"
    )

    np.save(
        output_path,
        prediction,
    )

    plot_prediction(
        image=sample.modalities[MRI_MODALITIES[3]],
        prediction=prediction,
        ground_truth=None,
        output_path=args.output / f"{args.patient}_prediction.png",
    )

    probability_path = (
    args.output
    / f"{args.patient}_probabilities.npy"
)

    np.save(
        probability_path,
        probabilities,
    )

    plot_probabilities(
        image=sample.modalities[MRI_MODALITIES[0]],
        probabilities=probabilities,
        output_path=args.output /
            f"{args.patient}_probabilities.png",
    )

    print(f"Prediction saved to: {output_path}")
    print(f"Probabilities saved to: {probability_path}")


if __name__ == "__main__":
    main()