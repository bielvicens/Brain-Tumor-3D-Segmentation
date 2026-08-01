"""Tests for src.inference.predictor.Predictor.

Uses minimal dummy torch.nn.Module models defined here - never the real
U-Net - since Predictor is meant to work with *any* model returning
multiclass logits.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import patch

import pytest
import torch

from src.inference.predictor import Predictor


# ----------------------------------------------------------------------
# Dummy models
# ----------------------------------------------------------------------
class _DummySegmentationModel(torch.nn.Module):
    """A minimal stand-in for a real segmentation network: a 1x1x1 conv
    mapping input channels to `out_channels` logits, preserving the
    spatial shape exactly. Used only to exercise Predictor's plumbing."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ModeTrackingModel(torch.nn.Module):
    """Records the training-mode flag and grad-enabled state seen during
    its own forward pass, so tests can assert on them directly."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(2, 2, kernel_size=1)
        self.was_training_during_forward: Optional[bool] = None
        self.grad_was_enabled_during_forward: Optional[bool] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.was_training_during_forward = self.training
        self.grad_was_enabled_during_forward = torch.is_grad_enabled()
        return self.conv(x)


class _FixedLogitsModel(torch.nn.Module):
    """Ignores the *content* of its input but respects its batch size,
    always returning the same per-voxel logits - lets tests assert exact
    expected softmax/argmax results."""

    def __init__(self, per_class_logits: torch.Tensor) -> None:
        super().__init__()
        # per_class_logits: shape (C, D, H, W)
        self.register_buffer("per_class_logits", per_class_logits)
        # A real (unused) parameter, so `.to(device)` has something to move.
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.per_class_logits.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)


class _NotAModule:
    """Deliberately not a torch.nn.Module, to test model validation."""

    def __call__(self, x):
        return x


# ----------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------
def test_predictor_rejects_non_module_model() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        Predictor(_NotAModule())  # type: ignore[arg-type]


def test_predictor_rejects_invalid_device_type() -> None:
    with pytest.raises(TypeError, match="device"):
        Predictor(_DummySegmentationModel(), device=123)  # type: ignore[arg-type]


def test_predictor_rejects_invalid_device_string() -> None:
    with pytest.raises(ValueError):
        Predictor(_DummySegmentationModel(), device="not-a-real-device")


def test_predictor_rejects_cuda_when_unavailable() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(ValueError, match="CUDA is not available"):
            Predictor(_DummySegmentationModel(), device="cuda")


def test_predictor_auto_selects_cpu_when_cuda_unavailable() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        predictor = Predictor(_DummySegmentationModel(), device=None)
    assert predictor.device.type == "cpu"


def test_predictor_accepts_torch_device_instance() -> None:
    predictor = Predictor(_DummySegmentationModel(), device=torch.device("cpu"))
    assert predictor.device == torch.device("cpu")


def test_predictor_moves_model_to_requested_device() -> None:
    predictor = Predictor(_DummySegmentationModel(), device="cpu")
    assert next(predictor.model.parameters()).device.type == "cpu"


# ----------------------------------------------------------------------
# Inference on CPU / basic shapes
# ----------------------------------------------------------------------
def test_inference_runs_on_cpu() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3), device="cpu")
    x = torch.randn(4, 8, 8, 8)
    mask = predictor.predict_mask(x)
    assert mask.device.type == "cpu"


def test_predict_logits_supports_unbatched_input() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(4, 8, 8, 8)  # (C, D, H, W)
    logits = predictor.predict_logits(x)
    assert logits.shape == (1, 3, 8, 8, 8)  # batch dim added, N=1


def test_predict_logits_supports_batched_input() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(2, 4, 8, 8, 8)  # (N, C, D, H, W)
    logits = predictor.predict_logits(x)
    assert logits.shape == (2, 3, 8, 8, 8)  # batch preserved


def test_predict_probabilities_shape_matches_logits() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(2, 4, 8, 8, 8)
    logits = predictor.predict_logits(x)
    probabilities = predictor.predict_probabilities(x)
    assert probabilities.shape == logits.shape


def test_predict_mask_shape_with_batch() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(2, 4, 8, 8, 8)  # (N, C, D, H, W)
    mask = predictor.predict_mask(x)
    assert mask.shape == (2, 8, 8, 8)  # (N, D, H, W)


def test_predict_mask_shape_without_batch() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(4, 8, 8, 8)  # (C, D, H, W)
    mask = predictor.predict_mask(x)
    assert mask.shape == (8, 8, 8)  # batch dim removed again


def test_predict_mask_dtype_is_long() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(4, 8, 8, 8)
    mask = predictor.predict_mask(x)
    assert mask.dtype == torch.long


# ----------------------------------------------------------------------
# Numerical correctness
# ----------------------------------------------------------------------
def test_predict_probabilities_sum_to_one_per_voxel() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(2, 4, 6, 6, 6)
    probabilities = predictor.predict_probabilities(x)
    voxel_sums = probabilities.sum(dim=1)
    assert torch.allclose(voxel_sums, torch.ones_like(voxel_sums), atol=1e-5)


def test_predict_probabilities_values_in_zero_one_range() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(4, 6, 6, 6)
    probabilities = predictor.predict_probabilities(x)
    assert torch.all(probabilities >= 0.0)
    assert torch.all(probabilities <= 1.0)


def test_predict_logits_applies_no_activation() -> None:
    # A logit clearly outside [0, 1] proves no softmax/sigmoid was applied.
    fixed_logits = torch.tensor([[[[10.0]]], [[[-5.0]]], [[[2.0]]]])
    predictor = Predictor(_FixedLogitsModel(fixed_logits))
    x = torch.zeros(4, 1, 1, 1)

    logits = predictor.predict_logits(x)

    assert torch.equal(logits[0], fixed_logits.to(logits.device))


def test_argmax_selects_the_class_with_highest_probability() -> None:
    # Single voxel, 3 classes, logits [1.0, 5.0, 2.0] -> class 1 wins.
    fixed_logits = torch.tensor([[[[1.0]]], [[[5.0]]], [[[2.0]]]])  # (3, 1, 1, 1)
    predictor = Predictor(_FixedLogitsModel(fixed_logits))
    x = torch.zeros(4, 1, 1, 1)  # content irrelevant, model ignores it

    mask = predictor.predict_mask(x)
    assert mask.shape == (1, 1, 1)
    assert mask.item() == 1


def test_argmax_is_consistent_with_manual_softmax_computation() -> None:
    fixed_logits = torch.tensor([[[[0.1]]], [[[0.9]]], [[[-0.3]]]])
    predictor = Predictor(_FixedLogitsModel(fixed_logits))
    x = torch.zeros(4, 1, 1, 1)

    expected_class = int(torch.argmax(torch.softmax(fixed_logits, dim=0), dim=0).item())
    assert predictor.predict_mask(x).item() == expected_class


# ----------------------------------------------------------------------
# eval() mode / no gradient tracking
# ----------------------------------------------------------------------
def test_model_is_switched_to_eval_mode_during_forward() -> None:
    model = _ModeTrackingModel()
    model.train()  # deliberately leave it in train mode beforehand
    predictor = Predictor(model, device="cpu")

    predictor.predict_mask(torch.randn(2, 4, 4, 4))

    assert model.was_training_during_forward is False


def test_gradients_are_never_computed_during_inference() -> None:
    model = _ModeTrackingModel()
    predictor = Predictor(model, device="cpu")

    x = torch.randn(2, 4, 4, 4)
    logits = predictor.predict_logits(x)

    assert model.grad_was_enabled_during_forward is False
    assert logits.requires_grad is False


def test_model_parameters_still_require_grad_but_output_does_not() -> None:
    # The model's own parameters are trainable (as in a real training
    # setup) - it's specifically the *forward pass under the Predictor*
    # that must not build a graph, not the parameters themselves.
    model = _DummySegmentationModel(in_channels=4, out_channels=3)
    assert any(p.requires_grad for p in model.parameters())

    predictor = Predictor(model, device="cpu")
    logits = predictor.predict_logits(torch.randn(4, 4, 4, 4))
    assert logits.requires_grad is False
    assert logits.grad_fn is None


# ----------------------------------------------------------------------
# Input is never modified
# ----------------------------------------------------------------------
def test_input_tensor_is_not_modified_by_predict_logits() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(4, 4, 4, 4)
    original = x.clone()

    predictor.predict_logits(x)

    assert torch.equal(x, original)
    assert x.shape == (4, 4, 4, 4)  # unsqueeze happened on a copy, not on x


def test_input_tensor_is_not_modified_by_predict_mask() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(2, 4, 4, 4, 4)
    original = x.clone()

    predictor.predict_mask(x)

    assert torch.equal(x, original)


# ----------------------------------------------------------------------
# __call__ is an alias for predict_mask
# ----------------------------------------------------------------------
def test_call_is_equivalent_to_predict_mask() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    x = torch.randn(4, 6, 6, 6)

    # Same model/weights, so both calls should produce identical output.
    result_via_call = predictor(x)
    result_via_method = predictor.predict_mask(x)

    assert torch.equal(result_via_call, result_via_method)
    assert result_via_call.shape == result_via_method.shape == (6, 6, 6)


# ----------------------------------------------------------------------
# Errors: wrong number of dimensions
# ----------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(4, 4, 4), (4,), (1, 2, 4, 4, 4, 4)])
def test_predict_logits_rejects_wrong_ndim(shape) -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    with pytest.raises(ValueError, match="dimension"):
        predictor.predict_logits(torch.randn(*shape))


def test_predict_probabilities_rejects_wrong_ndim() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    with pytest.raises(ValueError, match="dimension"):
        predictor.predict_probabilities(torch.randn(4, 4, 4))


def test_predict_mask_rejects_wrong_ndim() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    with pytest.raises(ValueError, match="dimension"):
        predictor.predict_mask(torch.randn(4, 4, 4))


# ----------------------------------------------------------------------
# Errors: invalid input type
# ----------------------------------------------------------------------
def test_predict_logits_rejects_non_tensor_input() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    with pytest.raises(TypeError, match="torch.Tensor"):
        predictor.predict_logits([[1, 2], [3, 4]])  # type: ignore[arg-type]


def test_predict_probabilities_rejects_non_tensor_input() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    with pytest.raises(TypeError, match="torch.Tensor"):
        predictor.predict_probabilities("not a tensor")  # type: ignore[arg-type]


def test_predict_mask_rejects_non_tensor_input() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    with pytest.raises(TypeError, match="torch.Tensor"):
        predictor.predict_mask(None)  # type: ignore[arg-type]


def test_call_rejects_non_tensor_input() -> None:
    predictor = Predictor(_DummySegmentationModel(in_channels=4, out_channels=3))
    with pytest.raises(TypeError, match="torch.Tensor"):
        predictor(42)  # type: ignore[arg-type]
