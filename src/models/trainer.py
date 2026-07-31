"""Training utilities for the 3D U-Net segmentation model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    """Stores training and validation metrics for each epoch."""

    train_loss: list[float]
    val_loss: list[float]

    @property
    def epochs(self) -> int:
        return len(self.train_loss)


class Trainer:
    """Minimal, explicit training loop for a PyTorch segmentation model.

    The trainer deliberately contains no dataset-specific preprocessing
    logic. The Dataset and preprocessing pipeline are responsible for
    producing tensors in the correct format.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch and return the mean loss."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for images, masks in dataloader:
            if masks is None:
                raise ValueError("Training requires segmentation masks.")

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Cannot train on an empty dataloader.")

        return total_loss / num_batches

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Run one validation epoch and return the mean loss."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for images, masks in dataloader:
            if masks is None:
                raise ValueError("Validation requires segmentation masks.")

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Cannot validate on an empty dataloader.")

        return total_loss / num_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 1,
    ) -> TrainingHistory:
        """Train the model for ``epochs`` and optionally validate."""
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        history = TrainingHistory(train_loss=[], val_loss=[])

        for _ in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history.train_loss.append(train_loss)

            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                history.val_loss.append(val_loss)

        return history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        history: Optional[TrainingHistory] = None,
    ) -> Path:
        """Save model and optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, object] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if history is not None:
            checkpoint["history"] = {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,
            }

        torch.save(checkpoint, path)
        return path