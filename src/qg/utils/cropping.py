"""Reshaping utils."""

import torch


def crop(t: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Crop the last two dimensions of a tensor.

    Args:
        t (torch.Tensor): Tensor to crop.
        offset (int, optional): Crop width. Defaults to 0.

    Returns:
        torch.Tensor: Cropped tensor.
    """
    if offset < 0:
        msg = "Cropping offset must be greater or equal to 0."
        raise ValueError(msg)
    if offset == 0:
        return t
    return t[..., offset:-offset, offset:-offset]
