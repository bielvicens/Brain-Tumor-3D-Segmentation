"""Tests for the 3D U-Net model."""

import pytest
import torch

from src.models.unet import DoubleConv3D, UNet3D


def test_default_model_accepts_four_modalities() -> None:
    model = UNet3D()
    x = torch.randn(1, 4, 16, 16, 16)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 4, 16, 16, 16)


def test_model_preserves_spatial_shape() -> None:
    model = UNet3D(in_channels=4, out_channels=4, base_channels=4)
    x = torch.randn(1, 4, 18, 20, 22)
    with torch.no_grad():
        y = model(x)

    assert y.shape[2:] == x.shape[2:]


def test_custom_number_of_input_and_output_channels() -> None:
    model = UNet3D(in_channels=2, out_channels=3, base_channels=4)
    x = torch.randn(1, 2, 16, 16, 16)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 3, 16, 16, 16)


def test_model_supports_odd_spatial_dimensions() -> None:
    model = UNet3D(base_channels=4)
    x = torch.randn(1, 4, 17, 19, 21)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 4, 17, 19, 21)


def test_model_is_fully_convolutional() -> None:
    model = UNet3D(base_channels=4)
    x1 = torch.randn(1, 4, 16, 16, 16)
    x2 = torch.randn(1, 4, 20, 18, 16)

    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)

    assert y1.shape[2:] == x1.shape[2:]
    assert y2.shape[2:] == x2.shape[2:]


def test_invalid_input_rank_raises() -> None:
    model = UNet3D(base_channels=4)

    with pytest.raises(ValueError, match="5"):
        model(torch.randn(4, 16, 16, 16))


def test_invalid_channel_count_raises() -> None:
    model = UNet3D(in_channels=4, base_channels=4)

    with pytest.raises(ValueError, match="input channels"):
        model(torch.randn(1, 3, 16, 16, 16))


def test_invalid_constructor_arguments_raise() -> None:
    with pytest.raises(ValueError):
        UNet3D(in_channels=0)

    with pytest.raises(ValueError):
        UNet3D(out_channels=0)

    with pytest.raises(ValueError):
        UNet3D(base_channels=0)


def test_double_conv_rejects_invalid_channels() -> None:
    with pytest.raises(ValueError):
        DoubleConv3D(0, 4)


def test_model_has_trainable_parameters() -> None:
    model = UNet3D(base_channels=4)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_model_outputs_logits_not_probabilities() -> None:
    model = UNet3D(base_channels=4)
    x = torch.randn(1, 4, 16, 16, 16)

    with torch.no_grad():
        y = model(x)

    # The model head deliberately returns raw logits. There is no softmax
    # inside the model because the training loss will handle normalization.
    assert y.dtype == x.dtype
    assert not torch.allclose(y.sum(dim=1), torch.ones_like(y[:, 0]), atol=1e-3)


def test_backward_pass_works() -> None:
    model = UNet3D(base_channels=4)
    x = torch.randn(1, 4, 16, 16, 16)
    target = torch.randint(0, 4, (1, 16, 16, 16))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
