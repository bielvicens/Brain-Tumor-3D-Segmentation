from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


class SlidingWindowInference:
    """Run 3D segmentation inference over an entire volume.

    The volume is divided into overlapping patches. Each patch is passed
    through the model and the resulting class probabilities are averaged
    in overlapping regions.

    Args:
        patch_size:
            Spatial size of each inference patch.
        overlap:
            Fraction of overlap between consecutive patches.
            For example, 0.5 means 50% overlap.
        device:
            Device used for inference.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:

        if len(patch_size) != 3:
            raise ValueError(
                "patch_size must contain exactly three dimensions."
            )

        if any(size <= 0 for size in patch_size):
            raise ValueError(
                "patch_size values must be positive."
            )

        if not 0.0 <= overlap < 1.0:
            raise ValueError(
                "overlap must satisfy 0 <= overlap < 1."
            )

        self.patch_size = tuple(patch_size)
        self.overlap = overlap
        self.device = torch.device(device)

    def predict(
        self,
        model: torch.nn.Module,
        volume: torch.Tensor,
    ) -> torch.Tensor:
        """Predict class probabilities for the complete volume.

        Args:
            model:
                Segmentation model returning logits with shape
                ``(N, C, D, H, W)``.

            volume:
                Input tensor with shape ``(C, D, H, W)`` or
                ``(1, C, D, H, W)``.

        Returns:
            Tensor with shape ``(C_out, D, H, W)`` containing averaged
            class probabilities.
        """

        if not isinstance(volume, torch.Tensor):
            raise TypeError(
                "volume must be a torch.Tensor."
            )

        if volume.ndim == 4:
            volume = volume.unsqueeze(0)
        elif volume.ndim != 5:
            raise ValueError(
                "volume must have shape (C,D,H,W) or (N,C,D,H,W)."
            )

        if volume.shape[0] != 1:
            raise ValueError(
                "SlidingWindowInference currently supports batch size 1."
            )

        volume = volume.to(self.device)

        model = model.to(self.device)
        model.eval()

        spatial_shape = tuple(volume.shape[2:])

        padded_volume, padding = self._pad_if_needed(volume)

        padded_shape = tuple(padded_volume.shape[2:])

        starts = [
            self._compute_starts(
                dimension,
                patch,
            )
            for dimension, patch in zip(
                padded_shape,
                self.patch_size,
            )
        ]

        probability_sum = None
        count_map = torch.zeros(
            (1, 1, *padded_shape),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            for d_start in starts[0]:
                for h_start in starts[1]:
                    for w_start in starts[2]:

                        d_end = d_start + self.patch_size[0]
                        h_end = h_start + self.patch_size[1]
                        w_end = w_start + self.patch_size[2]

                        patch = padded_volume[
                            :,
                            :,
                            d_start:d_end,
                            h_start:h_end,
                            w_start:w_end,
                        ]

                        logits = model(patch)

                        probabilities = F.softmax(
                            logits,
                            dim=1,
                        )

                        if probability_sum is None:
                            num_classes = probabilities.shape[1]

                            probability_sum = torch.zeros(
                                (
                                    1,
                                    num_classes,
                                    *padded_shape,
                                ),
                                dtype=probabilities.dtype,
                                device=self.device,
                            )

                        probability_sum[
                            :,
                            :,
                            d_start:d_end,
                            h_start:h_end,
                            w_start:w_end,
                        ] += probabilities

                        count_map[
                            :,
                            :,
                            d_start:d_end,
                            h_start:h_end,
                            w_start:w_end,
                        ] += 1.0

        if probability_sum is None:
            raise RuntimeError(
                "Sliding-window inference produced no patches."
            )

        probabilities = probability_sum / count_map.clamp_min(1.0)

        probabilities = self._remove_padding(
            probabilities,
            padding,
            spatial_shape,
        )

        return probabilities.squeeze(0)

    def predict_mask(
        self,
        model: torch.nn.Module,
        volume: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final segmentation mask."""

        probabilities = self.predict(
            model=model,
            volume=volume,
        )

        return torch.argmax(
            probabilities,
            dim=0,
        ).long()

    def _pad_if_needed(
        self,
        volume: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:

        padding = []

        for dimension, patch in zip(
            reversed(volume.shape[2:]),
            reversed(self.patch_size),
        ):
            difference = max(0, patch - dimension)

            before = difference // 2
            after = difference - before

            padding.extend(
                [before, after]
            )

        if any(padding):
            volume = F.pad(
                volume,
                padding,
                mode="constant",
                value=0.0,
            )

        return volume, tuple(padding)

    def _remove_padding(
        self,
        volume: torch.Tensor,
        padding: tuple[int, ...],
        original_shape: tuple[int, ...],
    ) -> torch.Tensor:

        d_before, d_after = padding[4], padding[5]
        h_before, h_after = padding[2], padding[3]
        w_before, w_after = padding[0], padding[1]

        d_start = d_before
        h_start = h_before
        w_start = w_before

        d_end = d_start + original_shape[0]
        h_end = h_start + original_shape[1]
        w_end = w_start + original_shape[2]

        return volume[
            :,
            :,
            d_start:d_end,
            h_start:h_end,
            w_start:w_end,
        ]

    def _compute_starts(
        self,
        dimension: int,
        patch: int,
    ) -> list[int]:

        if dimension <= patch:
            return [0]

        stride = max(
            1,
            int(patch * (1.0 - self.overlap)),
        )

        starts = list(
            range(
                0,
                dimension - patch + 1,
                stride,
            )
        )

        final_start = dimension - patch

        if starts[-1] != final_start:
            starts.append(final_start)

        return starts