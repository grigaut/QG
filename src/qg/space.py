"""Compute space transformations."""

import torch

from qg.fd import interp_TP


def compute_xy_u(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the x and y coordinates of the u grid.

    Args:
        x (torch.Tensor): x coordinates of the psi grid.
        y (torch.Tensor): y coordinates of the psi grid.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x and y coordinates of the u grid.
    """
    xu = (x[:, 1:] + x[:, :-1]) / 2
    yu = (y[:, 1:] + y[:, :-1]) / 2
    return xu, yu


def compute_xy_v(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the x and y coordinates of the v grid.

    Args:
        x (torch.Tensor): x coordinates of the psi grid.
        y (torch.Tensor): y coordinates of the psi grid.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x and y coordinates of the v grid.
    """
    xv = (x[1:, :] + x[:-1, :]) / 2
    yv = (y[1:, :] + y[:-1, :]) / 2
    return xv, yv


def compute_xy_q(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the x and y coordinates of the q grid.

    Args:
        x (torch.Tensor): x coordinates of the psi grid.
        y (torch.Tensor): y coordinates of the psi grid.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: x and y coordinates of the q grid.
    """
    xq = interp_TP(x)
    yq = interp_TP(y)
    return xq, yq
