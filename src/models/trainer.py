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
from src.utils.metrics import dice_per_class
from src.inference.sliding_window import SlidingWindowInference


@dataclass
class TrainingHistory:
    """Stores training and validation metrics for each epoch."""

    train_loss: list[float]
    val_loss: list[float]

    train_dice: list[float]
    val_dice: list[float]

    train_ncr_dice: list[float]
    val_ncr_dice: list[float]

    train_ed_dice: list[float]
    val_ed_dice: list[float]

    train_et_dice: list[float]
    val_et_dice: list[float]

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

        # Resolve device FIRST.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # Move model to selected device.
        self.model.to(self.device)

        # Mixed precision scaler.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda",
        )

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader: DataLoader,
        sliding_window_overlap: float = 0.25,
    ) -> tuple[float, float, float, float, float]:
        """
        Run one validation epoch.

        Args:
            dataloader:
                Validation DataLoader.
            sliding_window_overlap:
                Overlap fraction for sliding window inference.
                Lower values are faster but slightly less accurate.
                Recommended: 0.25 for speed, 0.5 for accuracy.

        Returns:
            (
                mean_loss,
                mean_dice,
                ncr_dice,
                ed_dice,
                et_dice,
            )

        Classes absent from both prediction and target are excluded
        from the Dice averages.
        """

        self.model.eval()

        total_loss = 0.0

        # --------------------------------------------------
        # DICE ACCUMULATORS
        #
        # We keep separate sums and counts because a class may
        # be absent from both prediction and target in a batch.
        # Such a class returns NaN and must not contribute to
        # the mean.
        # --------------------------------------------------

        total_dice = 0.0
        total_ncr_dice = 0.0
        total_ed_dice = 0.0
        total_et_dice = 0.0

        valid_dice_count = 0
        valid_ncr_count = 0
        valid_ed_count = 0
        valid_et_count = 0

        num_batches = 0

        sliding_window = SlidingWindowInference(
            patch_size=(96, 96, 96),
            overlap=sliding_window_overlap,
            device=self.device,
        )

        for images, masks in dataloader:

            if masks is None:
                raise ValueError(
                    "Validation requires segmentation masks."
                )

            images = images.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )

            masks = masks.to(
                self.device,
                dtype=torch.long,
                non_blocking=True,
            )

            # --------------------------------------------------
            # FORWARD + LOSS
            # --------------------------------------------------

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                batch_logits = []
                for i in range(images.shape[0]):
                    single_image = images[i:i+1]
                    probs = sliding_window.predict(self.model, single_image)
                    batch_logits.append(torch.log(probs.clamp_min(1e-8)))
                
                logits = torch.stack(batch_logits, dim=0)

                loss = self.criterion(
                    logits,
                    masks,
                )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            # --------------------------------------------------
            # PER-CLASS DICE
            #
            # class_dice:
            #   0 -> Background
            #   1 -> NCR
            #   2 -> ED
            #   3 -> ET
            #
            # A class absent from both prediction and target
            # returns NaN.
            # --------------------------------------------------

            class_dice = dice_per_class(
                prediction=predictions,
                target=masks,
                num_classes=logits.shape[1],
            )

            # --------------------------------------------------
            # GLOBAL MEAN DICE
            #
            # Exclude background and ignore NaN classes.
            # --------------------------------------------------

            foreground_dice = class_dice[1:]

            valid_foreground_dice = foreground_dice[
                ~torch.isnan(foreground_dice)
            ]

            if valid_foreground_dice.numel() > 0:

                dice = valid_foreground_dice.mean()

                total_dice += float(
                    dice.item()
                )

                valid_dice_count += 1

            # --------------------------------------------------
            # INDIVIDUAL REGION DICE
            #
            # class 1 = NCR
            # class 2 = ED
            # class 3 = ET
            # --------------------------------------------------

            ncr_dice = class_dice[1]
            ed_dice = class_dice[2]
            et_dice = class_dice[3]

            # --------------------------------------------------
            # NCR
            # --------------------------------------------------

            if not torch.isnan(ncr_dice):

                total_ncr_dice += float(
                    ncr_dice.item()
                )

                valid_ncr_count += 1

            # --------------------------------------------------
            # ED
            # --------------------------------------------------

            if not torch.isnan(ed_dice):

                total_ed_dice += float(
                    ed_dice.item()
                )

                valid_ed_count += 1

            # --------------------------------------------------
            # ET
            # --------------------------------------------------

            if not torch.isnan(et_dice):

                total_et_dice += float(
                    et_dice.item()
                )

                valid_et_count += 1

            # --------------------------------------------------
            # LOSS
            # --------------------------------------------------

            total_loss += float(
                loss.item()
            )

            num_batches += 1

        # --------------------------------------------------
        # EMPTY DATALOADER
        # --------------------------------------------------

        if num_batches == 0:
            raise ValueError(
                "Cannot validate on an empty dataloader."
            )

        # --------------------------------------------------
        # FINAL METRICS
        # --------------------------------------------------

        mean_loss = (
            total_loss / num_batches
        )

        mean_dice = (
            total_dice / valid_dice_count
            if valid_dice_count > 0
            else 0.0
        )

        ncr_dice = (
            total_ncr_dice / valid_ncr_count
            if valid_ncr_count > 0
            else 0.0
        )

        ed_dice = (
            total_ed_dice / valid_ed_count
            if valid_ed_count > 0
            else 0.0
        )

        et_dice = (
            total_et_dice / valid_et_count
            if valid_et_count > 0
            else 0.0
        )

        return (
            mean_loss,
            mean_dice,
            ncr_dice,
            ed_dice,
            et_dice,
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        *,
        progress_bar=None,
        epoch: int = 0,
        epochs: int = 1,
    ) -> tuple[float, float, float, float, float]:
        """Run one training epoch and return mean loss and Dice."""

        self.model.train()

        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        total_ncr_dice = 0.0
        total_ed_dice = 0.0
        total_et_dice = 0.0

        valid_dice_count = 0
        valid_ncr_count = 0
        valid_ed_count = 0
        valid_et_count = 0

        for images, masks in dataloader:
            if masks is None:
                raise ValueError("Training requires segmentation masks.")

            images = images.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            masks = masks.to(
                self.device,
                dtype=torch.long,
                non_blocking=True,
            )

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            predictions = torch.argmax(logits, dim=1)

            class_dice = dice_per_class(
                prediction=predictions,
                target=masks,
                num_classes=logits.shape[1],
            )

            foreground_dice = class_dice[1:]
            valid_foreground_dice = foreground_dice[~torch.isnan(foreground_dice)]
            if valid_foreground_dice.numel() > 0:
                total_dice += float(valid_foreground_dice.mean().item())
                valid_dice_count += 1

            ncr_d = class_dice[1]
            ed_d = class_dice[2]
            et_d = class_dice[3]

            if not torch.isnan(ncr_d):
                total_ncr_dice += float(ncr_d.item())
                valid_ncr_count += 1

            if not torch.isnan(ed_d):
                total_ed_dice += float(ed_d.item())
                valid_ed_count += 1

            if not torch.isnan(et_d):
                total_et_dice += float(et_d.item())
                valid_et_count += 1

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.detach().item())
            num_batches += 1

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    epoch=f"{epoch + 1}/{epochs}",
                    loss=f"{loss.item():.4f}",
                    ncr=f"{ncr_d.item() if not torch.isnan(ncr_d) else 0.0:.4f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

        if num_batches == 0:
            raise ValueError("Cannot train on an empty dataloader.")

        return (
            total_loss / num_batches,
            total_dice / valid_dice_count if valid_dice_count > 0 else 0.0,
            total_ncr_dice / valid_ncr_count if valid_ncr_count > 0 else 0.0,
            total_ed_dice / valid_ed_count if valid_ed_count > 0 else 0.0,
            total_et_dice / valid_et_count if valid_et_count > 0 else 0.0,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        epochs: int = 1,
        start_epoch: int = 0,
        history: Optional[TrainingHistory] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Optional[str | Path] = None,
        scheduler: Optional[LRScheduler] = None,
        val_every_n_epochs: int = 1,
        sliding_window_overlap: float = 0.25,
    ) -> TrainingHistory:
        """
        Train the model for ``epochs`` and optionally validate.

        The best checkpoint is selected according to validation
        NCR Dice, because improving necrotic-core segmentation
        is the primary optimization objective.

        Args:
            train_loader:
                DataLoader for training data.
            val_loader:
                Optional DataLoader for validation data.
            epochs:
                Total number of epochs to train.
            start_epoch:
                Starting epoch number (for resuming).
            history:
                Optional training history to continue.
            early_stopping:
                Optional early stopping callback.
            checkpoint_dir:
                Directory to save checkpoints.
            scheduler:
                Optional learning rate scheduler.
            val_every_n_epochs:
                Validate every N epochs. Default 1 (every epoch).
                Set to 5 to validate every 5 epochs and speed up training.
            sliding_window_overlap:
                Overlap fraction for sliding window validation inference.
                Lower values are faster. Recommended: 0.25 for speed, 0.5 for accuracy.
        """

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        # --------------------------------------------------
        # HISTORY
        # --------------------------------------------------

        if history is None:
            history = TrainingHistory(
                train_loss=[],
                val_loss=[],

                train_dice=[],
                val_dice=[],

                train_ncr_dice=[],
                val_ncr_dice=[],

                train_ed_dice=[],
                val_ed_dice=[],

                train_et_dice=[],
                val_et_dice=[],
            )

        # --------------------------------------------------
        # BEST NCR DICE
        # --------------------------------------------------

        best_ncr_dice = float("-inf")

        # --------------------------------------------------
        # CHECKPOINT DIRECTORY
        # --------------------------------------------------

        checkpoint_dir_path = None

        if checkpoint_dir is not None:
            checkpoint_dir_path = Path(checkpoint_dir)

            checkpoint_dir_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        # --------------------------------------------------
        # PROGRESS BAR
        # --------------------------------------------------

        from tqdm.auto import tqdm

        total_steps = epochs * len(train_loader)

        progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            unit="batch",
            dynamic_ncols=True,
        )

        # --------------------------------------------------
        # TRAINING LOOP
        # --------------------------------------------------

        for epoch in range(start_epoch, epochs):

            # ==================================================
            # TRAIN
            # ==================================================

            (
                train_loss,
                train_dice,
                train_ncr_dice,
                train_ed_dice,
                train_et_dice,
            ) = self.train_epoch(
                train_loader,
                progress_bar=progress_bar,
                epoch=epoch,
                epochs=epochs,
            )

            # --------------------------------------------------
            # SAVE TRAIN METRICS
            # --------------------------------------------------

            history.train_loss.append(
                train_loss
            )

            history.train_dice.append(
                train_dice
            )

            history.train_ncr_dice.append(
                train_ncr_dice
            )

            history.train_ed_dice.append(
                train_ed_dice
            )

            history.train_et_dice.append(
                train_et_dice
            )

            # ==================================================
            # VALIDATION
            # ==================================================

            val_loss = None
            val_dice = None
            val_ncr_dice = None
            val_ed_dice = None
            val_et_dice = None

            should_validate = (
                val_loader is not None
                and (epoch + 1) % val_every_n_epochs == 0
            )

            if should_validate:

                print(
                    f"\n[Epoch {epoch + 1}/{epochs}] "
                    f"Running validation..."
                )

                (
                    val_loss,
                    val_dice,
                    val_ncr_dice,
                    val_ed_dice,
                    val_et_dice,
                ) = self.validate_epoch(
                    val_loader,
                    sliding_window_overlap=sliding_window_overlap,
                )

                # --------------------------------------------------
                # SAVE VALIDATION METRICS
                # --------------------------------------------------

                history.val_loss.append(
                    val_loss
                )

                history.val_dice.append(
                    val_dice
                )

                history.val_ncr_dice.append(
                    val_ncr_dice
                )

                history.val_ed_dice.append(
                    val_ed_dice
                )

                history.val_et_dice.append(
                    val_et_dice
                )
            else:
                # --------------------------------------------------
                # SKIP VALIDATION BUT MAINTAIN HISTORY LENGTH
                # --------------------------------------------------
                if val_loader is not None:
                    history.val_loss.append(None)
                    history.val_dice.append(None)
                    history.val_ncr_dice.append(None)
                    history.val_ed_dice.append(None)
                    history.val_et_dice.append(None)

            # ==================================================
            # SCHEDULER
            # ==================================================

            if scheduler is not None:
                scheduler.step()

            # ==================================================
            # CHECKPOINTS
            # ==================================================

            if checkpoint_dir_path is not None:

                # --------------------------------------------------
                # ALWAYS SAVE LAST
                # --------------------------------------------------

                self.save_checkpoint(
                    checkpoint_dir_path / "last.pt",
                    epoch + 1,
                    history,
                )

                # --------------------------------------------------
                # SAVE BEST ACCORDING TO NCR DICE
                # --------------------------------------------------

                if (
                    val_ncr_dice is not None
                    and val_ncr_dice > best_ncr_dice
                ):

                    best_ncr_dice = val_ncr_dice

                    self.save_checkpoint(
                        checkpoint_dir_path / "best.pt",
                        epoch + 1,
                        history,
                    )

                    print(
                        f"\n"
                        f"New best model! "
                        f"NCR Dice = "
                        f"{val_ncr_dice:.4f}"
                    )

            # ==================================================
            # EARLY STOPPING
            # ==================================================

            if (
                early_stopping is not None
                and val_ncr_dice is not None
                and early_stopping.step(
                    val_ncr_dice
                )
            ):
                print(
                    "\nEarly stopping triggered."
                )
                break

        # --------------------------------------------------
        # CLOSE PROGRESS BAR
        # --------------------------------------------------

        progress_bar.close()

        # ==================================================
        # SAVE HISTORY / PLOTS
        # ==================================================

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
        """Save model, optimizer state and training history."""

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint: Dict[str, object] = {
            "epoch": epoch,

            "model_state_dict": (
                self.model.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
        }

        if history is not None:

            checkpoint["history"] = {
                "train_loss": history.train_loss,
                "val_loss": history.val_loss,

                "train_dice": history.train_dice,
                "val_dice": history.val_dice,

                "train_ncr_dice": history.train_ncr_dice,
                "val_ncr_dice": history.val_ncr_dice,

                "train_ed_dice": history.train_ed_dice,
                "val_ed_dice": history.val_ed_dice,

                "train_et_dice": history.train_et_dice,
                "val_et_dice": history.val_et_dice,
            }

        torch.save(
            checkpoint,
            path,
        )

        return path
    
    def save_history(
        self,
        history: TrainingHistory,
        output_dir: str | Path,
    ) -> Path:
        """Save the complete training history as JSON."""

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        history_path = (
            output_dir / "history.json"
        )

        with history_path.open(
            "w"
        ) as f:

            json.dump(
                {
                    "train_loss": history.train_loss,
                    "val_loss": history.val_loss,

                    "train_dice": history.train_dice,
                    "val_dice": history.val_dice,

                    "train_ncr_dice": (
                        history.train_ncr_dice
                    ),

                    "val_ncr_dice": (
                        history.val_ncr_dice
                    ),

                    "train_ed_dice": (
                        history.train_ed_dice
                    ),

                    "val_ed_dice": (
                        history.val_ed_dice
                    ),

                    "train_et_dice": (
                        history.train_et_dice
                    ),

                    "val_et_dice": (
                        history.val_et_dice
                    ),
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