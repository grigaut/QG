"""Parameters generator."""

import itertools
from math import log, sqrt
from typing import Any

import torch


def gaussian_exp_field(
    order: int,  # noqa: ARG001
    spacing: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
    Lt_max: float,
    l_t: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Subdivide space into regular-spaced chunks.

    Compute standard deviation for gaussian enveloppes
    that cross at 0.5.

    Args:
        order (int): Taylor series order.
        spacing (int): Width of the chunks.
        xx_ref (torch.Tensor): Reference x locations.
        yy_ref (torch.Tensor): Reference Y locations.
        Lt_max (float): Max duration.
        l_t (float): Time spacing.

    Returns:
        dict[int,dict[str, Any]]: 0 -> parameters.
    """
    Nx, Ny = xx_ref.shape
    nx = int((Nx) / spacing)
    ny = int((Ny) / spacing)

    offset_x = int((Nx - spacing * nx) / 2)
    offset_y = int((Ny - spacing * ny) / 2)

    xc = xx_ref[offset_x + int(spacing / 2) :: spacing, 0].cpu().flatten()
    yc = yy_ref[0, offset_y + int(spacing / 2) :: spacing].cpu().flatten()

    centers = [(x.item(), y.item()) for x, y in itertools.product(xc, yc)]

    l_x = (xx_ref[1, 0] - xx_ref[0, 0]).cpu().item() * spacing
    l_y = (yy_ref[0, 1] - yy_ref[0, 0]).cpu().item() * spacing

    sx = l_x / sqrt(2 * log(2))
    sy = l_y / sqrt(2 * log(2))

    space = {
        0: {
            "centers": centers,
            "sigma_x": sx,
            "sigma_y": sy,
            "numel": len(centers),
        }
    }
    time = {}
    sigma_t = l_t / sqrt(2 * log(2))
    # Compute required number of time patches
    nt = int(Lt_max // l_t) + 1
    # Compute required offset to center patches on the domain
    offset_t = Lt_max - (nt - 1) * l_t
    # Compute patch centers
    tc = [offset_t * 0.5 + i * l_t for i in range(nt)]

    time[0] = {
        "centers": tc,
        "sigma_t": sigma_t,
        "numel": len(tc),
    }

    return space, time
