"""Training utilities for the 3D U-Net segmentation model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

import json

import matplotlib.pyplot as plt

from src.utils.early_stopping import EarlyStopping
from src.utils.metrics import mean_dice


@dataclass
class TrainingHistory:
    """Stores training and validation metrics for each epoch."""

    train_loss: list[float]
    val_loss: list[float]
    train_dice: list[float]
    val_dice: list[float]

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

    def train_epoch(self, dataloader: DataLoader, progress_bar=None, epoch=0,epochs=1) -> tuple[float, float]:
        """Run one training epoch and return the mean loss."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        total_dice = 0.0

        for images, masks in dataloader:
            if masks is None:
                raise ValueError("Training requires segmentation masks.")

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            predictions = torch.argmax(logits, dim=1)

            dice = mean_dice(
                prediction=predictions,
                target=masks,
                num_classes=logits.shape[1],
                include_background=False,
            )

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            total_dice += float(dice.item())
            num_batches += 1

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=f"{epoch+1}/{epochs}",
                    loss=f"{loss.item():.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        if num_batches == 0:
            raise ValueError("Cannot train on an empty dataloader.")

        return (
            total_loss / num_batches,
            total_dice / num_batches,
        )

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> tuple[float, float]:
        """Run one validation epoch and return the mean loss."""
        self.model.eval()

        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0

        for images, masks in dataloader:
            if masks is None:
                raise ValueError("Validation requires segmentation masks.")

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            predictions = torch.argmax(logits, dim=1)
            
            dice = mean_dice(
                prediction=predictions,
                target=masks,
                num_classes=logits.shape[1],
                include_background=False,
            )

            total_loss += float(loss.item())
            total_dice += float(dice.item())
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Cannot validate on an empty dataloader.")

        return (
            total_loss / num_batches,
            total_dice / num_batches,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        epochs: int = 1,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Optional[str | Path] = None,
        scheduler: Optional[LRScheduler] = None,
    ) -> TrainingHistory:
        """Train the model for ``epochs`` and optionally validate."""
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        history = TrainingHistory(
            train_loss=[],
            val_loss=[],
            train_dice=[],
            val_dice=[],
        )

        best_val_loss = float("inf")

        checkpoint_dir_path = None
        if checkpoint_dir is not None:
            checkpoint_dir_path = Path(checkpoint_dir)
            checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

        from tqdm.auto import tqdm

        total_steps = epochs * len(train_loader)

        progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            unit="batch",
            dynamic_ncols=True,
        )

        for epoch in range(epochs):
            train_loss, train_dice = self.train_epoch(
                train_loader,
                progress_bar=progress_bar,
                epoch=epoch,
                epochs=epochs,
            )

            history.train_loss.append(train_loss)
            history.train_dice.append(train_dice)

            val_loss = None

            if val_loader is not None:
                val_loss, val_dice = self.validate_epoch(val_loader)

                history.val_loss.append(val_loss)
                history.val_dice.append(val_dice)

            if scheduler is not None:
                scheduler.step()

            if checkpoint_dir_path is not None:
                self.save_checkpoint(
                    checkpoint_dir_path / "last.pt",
                    epoch + 1,
                    history,
                )

                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        checkpoint_dir_path / "best.pt",
                        epoch + 1,
                        history,
                    )

            if (
                early_stopping is not None
                and val_loss is not None
                and early_stopping.step(val_loss)
            ):
                break
        progress_bar.close()

        if checkpoint_dir_path is not None:
            self.save_history(
                history,
                checkpoint_dir_path,
            )

            self.save_training_plots(
                history,
                checkpoint_dir_path,
            )

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
                "train_dice": history.train_dice,
                "val_dice": history.val_dice,
            }

        torch.save(checkpoint, path)
    def save_history(
        self,
        history: TrainingHistory,
        output_dir: str | Path,
    ) -> Path:
        """Save the training history as a JSON file."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history_path = output_dir / "history.json"

        with history_path.open("w") as f:
            json.dump(
                {
                    "train_loss": history.train_loss,
                    "val_loss": history.val_loss,
                    "train_dice": history.train_dice,
                    "val_dice": history.val_dice,
                },
                f,
                indent=4,
            )

        return history_path


    def save_training_plots(
        self,
        history: TrainingHistory,
        output_dir: str | Path,
    ) -> None:
        """Save loss and Dice curves."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        #
        # LOSS
        #

        plt.figure(figsize=(7,5))

        plt.plot(
            history.train_loss,
            label="Train",
        )

        if history.val_loss:
            plt.plot(
                history.val_loss,
                label="Validation",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            output_dir / "loss.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        #
        # DICE
        #

        plt.figure(figsize=(7,5))

        plt.plot(
            history.train_dice,
            label="Train",
        )

        if history.val_dice:
            plt.plot(
                history.val_dice,
                label="Validation",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.title("Training Dice")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            output_dir / "dice.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()
        return path