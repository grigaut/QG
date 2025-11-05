"""System specs."""

import torch

from mqgeometry.specs._utils import Device, TensorSpecs

DEVICE = Device.set_automatically()


def use_cpu() -> None:
    """Use CPU."""
    DEVICE.use_cpu()


def use_cuda() -> None:
    """Use cuda."""
    DEVICE.use_cuda()


def from_tensor(tensor: torch.Tensor) -> TensorSpecs:
    """Replicate specs of a given tensor.

    Args:
        tensor (torch.Tensor): Original tensor.

    Returns:
        TensorSpecs: Tensor specs.
    """
    return TensorSpecs(dtype=tensor.dtype, device=tensor.device)
