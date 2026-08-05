"""Model inference (Predictor) for trained 3D segmentation models.

This module encapsulates everything needed to run a trained model at
inference time: moving it to the right device, keeping it in eval mode,
disabling gradient tracking, and turning raw model logits into class
probabilities and a final segmentation mask.

It is intentionally independent of the Trainer and the Dataset: a
Predictor only needs an already-built, already-initialized
``torch.nn.Module`` - it never loads checkpoints from disk and has no
opinion about how the model was trained or what architecture it is.
Sliding-window inference, test-time augmentation, model ensembling,
postprocessing, checkpoint loading, NIfTI export, visualization, and
metric computation are all out of scope here; each is a natural
extension point for a later module, layered on top of this one without
requiring any change to it.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class Predictor:
    """Runs inference with a trained 3D segmentation model.

    Wraps a ``torch.nn.Module`` that returns multiclass logits of shape
    ``(N, C, D, H, W)`` and provides three increasingly processed views of
    its output: raw logits, softmax probabilities, and a final integer
    segmentation mask.

    Example:
        >>> predictor = Predictor(model, device="cpu")
        >>> mask = predictor(volume)  # __call__ is an alias for predict_mask
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Args:
            model: An already-built, already-initialized model. Any
                ``torch.nn.Module`` that, given a ``(N, C, D, H, W)``
                input, returns multiclass logits of shape
                ``(N, C', D, H, W)`` works - the Predictor makes no
                architecture-specific assumptions.
            device: Where to run inference: ``"cpu"``, ``"cuda"``,
                ``"cuda:0"``, a ``torch.device``, or ``None`` to
                auto-select CUDA if available, otherwise CPU.

        Raises:
            TypeError: If ``model`` is not a ``torch.nn.Module``, or
                ``device`` is neither ``None``, a ``str``, nor a
                ``torch.device``.
            ValueError: If ``device`` is an invalid device string, or
                requests CUDA on a machine where CUDA isn't available.
        """
        self._validate_model(model)
        self.device = self._resolve_device(device)
        self.model = model.to(self.device)

        logger.info("Predictor initialized on device '%s'.", self.device)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model and return raw logits, with no activation applied.

        Args:
            x: Input volume, either ``(C, D, H, W)`` (a batch dimension is
                added automatically) or ``(N, C, D, H, W)``.

        Returns:
            The model's raw output, always batched: ``(N, C, D, H, W)``
            (``N=1`` if ``x`` had no batch dimension - that added
            dimension is *not* removed here, only :meth:`predict_mask`
            does that). Never modifies ``x``.

        Raises:
            TypeError: If ``x`` is not a ``torch.Tensor``.
            ValueError: If ``x`` doesn't have 4 or 5 dimensions.
        """
        batched_input, _ = self._prepare_input(x)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(batched_input)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model and return per-class probabilities.

        Args:
            x: Same accepted shapes as :meth:`predict_logits`.

        Returns:
            ``torch.softmax(logits, dim=1)``, same shape as
            :meth:`predict_logits`'s output. Every voxel's probabilities
            across the class dimension sum to 1.

        Raises:
            TypeError: If ``x`` is not a ``torch.Tensor``.
            ValueError: If ``x`` doesn't have 4 or 5 dimensions.
        """
        logits = self.predict_logits(x)
        return torch.softmax(logits, dim=1)

    def predict_with_probabilities(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both the predicted mask and the class probabilities."""

        probabilities = self.predict_probabilities(x)
        prediction = torch.argmax(
            probabilities,
            dim=1,
        ).to(torch.long)

        if x.dim() == 4:
            prediction = prediction.squeeze(0)
            probabilities = probabilities.squeeze(0)

        return prediction, probabilities

    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model and return the final integer segmentation mask.

        Args:
            x: Same accepted shapes as :meth:`predict_logits`.

        Returns:
            ``argmax(probabilities, dim=1)`` as a ``torch.long`` tensor.
            Shape ``(N, D, H, W)`` if ``x`` already had a batch dimension;
            ``(D, H, W)`` if it didn't - the batch dimension that was
            added internally is removed again, since the caller never
            asked for one.

        Raises:
            TypeError: If ``x`` is not a ``torch.Tensor``.
            ValueError: If ``x`` doesn't have 4 or 5 dimensions.
        """
        self._validate_input_tensor(x)
        had_batch = x.dim() == 5

        probabilities = self.predict_probabilities(x)
        mask = torch.argmax(probabilities, dim=1).to(torch.long)

        if not had_batch:
            mask = mask.squeeze(0)
        return mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`predict_mask`."""
        return self.predict_mask(x)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Validate ``x``, add a batch dimension if missing, and move it
        to this Predictor's device.

        Never modifies ``x`` itself: ``unsqueeze`` and ``to`` both return
        new tensors (``Tensor.to`` returns ``x`` unchanged only when it
        already matches the target device, in which case there is nothing
        to modify anyway).

        Returns:
            A tuple of (tensor ready for the model, whether ``x`` already
            had a batch dimension).
        """
        self._validate_input_tensor(x)
        had_batch = x.dim() == 5
        batched = x if had_batch else x.unsqueeze(0)
        return batched.to(self.device), had_batch

    @staticmethod
    def _validate_input_tensor(x: torch.Tensor) -> None:
        """Check that ``x`` is a tensor with 4 or 5 dimensions."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(x).__name__}.")
        if x.dim() not in (4, 5):
            raise ValueError(
                "Expected a tensor of shape (C, D, H, W) or (N, C, D, H, W), "
                f"got {x.dim()} dimension(s) with shape {tuple(x.shape)}."
            )

    @staticmethod
    def _validate_model(model: torch.nn.Module) -> None:
        """Check that ``model`` is a real ``torch.nn.Module``."""
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}.")

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        """Resolve ``device`` into a concrete, available ``torch.device``."""
        if device is None:
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            resolved = device
        elif isinstance(device, str):
            try:
                resolved = torch.device(device)
            except RuntimeError as exc:
                raise ValueError(f"Invalid device string: {device!r}.") from exc
        else:
            raise TypeError(
                f"device must be a str, torch.device, or None, got {type(device).__name__}."
            )

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{resolved}' but CUDA is not available on this machine."
            )
        return resolved
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "Predictor":
        """Create a Predictor from a saved checkpoint.

        Args:
            checkpoint_path:
                Path to a checkpoint generated by Trainer.save_checkpoint().
            model:
                Model instance with the same architecture used during
                training.
            device:
                Device where inference will run.

        Returns:
            Predictor ready for inference.

        Raises:
            FileNotFoundError:
                If the checkpoint does not exist.
            KeyError:
                If the checkpoint does not contain
                ``model_state_dict``.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: '{checkpoint_path}'."
            )

        resolved_device = cls._resolve_device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=resolved_device,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain 'model_state_dict'."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        logger.info(
            "Loaded checkpoint '%s'.",
            checkpoint_path,
        )

        return cls(
            model=model,
            device=resolved_device,
        )
