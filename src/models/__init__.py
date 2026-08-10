"""Public API for model and training components."""

from .trainer import Trainer, TrainingHistory
from .unet import UNet3D
from .losses import DiceCrossEntropyLoss

self.scaler = torch.amp.GradScaler(
    "cuda",
    enabled=self.device.type == "cuda",
)

__all__ = [
    "UNet3D",
    "Trainer",
    "TrainingHistory",
    "DiceCrossEntropyLoss",
]