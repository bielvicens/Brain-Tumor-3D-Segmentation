"""Configuration objects for the BraTS segmentation project.

This module centralises all configurable hyperparameters used throughout the
project. Dataclasses provide type safety, sensible defaults and make future
migration to YAML or JSON configuration straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# Data
# ============================================================================

@dataclass(slots=True)
class DataConfig:
    """Configuration for dataset loading."""

    dataset_root: Path = Path("data/raw/BraTS/TrainingData")
    train_split: float = 0.8
    validation_split: float = 0.2
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False


# ============================================================================
# Model
# ============================================================================

@dataclass(slots=True)
class ModelConfig:
    """Configuration of the segmentation network."""

    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 32


# ============================================================================
# Training
# ============================================================================

@dataclass(slots=True)
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: str = "cpu"


# ============================================================================
# Checkpoints
# ============================================================================

@dataclass(slots=True)
class CheckpointConfig:
    """Checkpoint configuration."""

    directory: Path = Path("checkpoints")
    best_model_name: str = "best_model.pt"
    last_model_name: str = "last_model.pt"
    save_best_only: bool = False


# ============================================================================
# Early stopping
# ============================================================================

@dataclass(slots=True)
class EarlyStoppingConfig:
    """Early stopping configuration."""

    patience: int = 20
    min_delta: float = 0.0
    mode: str = "min"


# ============================================================================
# Experiment
# ============================================================================

@dataclass(slots=True)
class ExperimentConfig:
    """General experiment information."""

    name: str = "brats_segmentation"
    seed: int = 42


# ============================================================================
# Root configuration
# ============================================================================

@dataclass(slots=True)
class ProjectConfig:
    """Complete project configuration."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )