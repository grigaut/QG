"""Wavemet parameters generation."""

from __future__ import annotations

import itertools
from math import log, sqrt
from typing import Any

import torch

from qg.logging.core import getLogger
from qg.logging.utils import meters2text, sec2text, step

logger = getLogger(__name__)


def dyadic_decomposition(
    order: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
    *,
    Lxy_max: float,
    Lt_max: float,
    sigma_t_sigma_xy_ratio: float = 1.17,
    sigma_xy_l_p_ratio: float = 1.75 / torch.pi * 2 * sqrt(2),
    l_xy_sigma_xy_ratio: float = 2 * sqrt(log(2)),
    l_t_sigma_t_ratio: float = 2 * sqrt(log(2)),
    use_time_frontier_as_centers: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Compute wavelet parameters using dyadic subdivisions.

    Args:
        order (int): Decomposition order (number of subdivision to perform).
        xx_ref (torch.Tensor): Reference x locations to use to place centers.
        yy_ref (torch.Tensor): Reference y locations to use to place centers.
        Lxy_max (float): Distance to start subdividing from.
        Lt_max (float): Total time duration to localize time patches.
        sigma_t_sigma_xy_ratio (float, optional): Ratio between time gaussian
            enveloppe and space gaussian enveloppe. Default value is set so
            to roughly match 900 km to 20 days and 80 km to 2 days.
            Defaults to 2.
        sigma_xy_l_p_ratio (float, optional): Ratio between space gaussian
            enveloppe std and subdivide length. Defaults set so that σ' allows
            exp((-x²-y²)/σ') ~ cos²(πx/(2σ))cos²(πy/(2σ)) close to 0.
            Defaults to 1.75/π*2*√2.
        l_xy_sigma_xy_ratio (float, optional): Ratio between space patches
            width and gaussian enveloppe std. Default set so that enveloppes
            overlap at 0.5. Defaults to 2*√(log(2)).
        l_t_sigma_t_ratio (float, optional): Ratio between time patches
            width and gaussian enveloppe std. Default set so that enveloppes
            overlap at 0.5. Defaults to 2*√(log(2)).
        use_time_frontier_as_centers (bool, optional): Center gaussian
            enveloppes on time boundaries. If set to True, no guarante that
            time enveloppes will overlap at 0.5. Defaults to False.

    Returns:
        tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
            Space params, time params.
    """
    Lx = (xx_ref[-1, 0] - xx_ref[0, 0]).item()
    x0 = xx_ref[0, 0].item()
    Ly = (yy_ref[0, -1] - yy_ref[0, 0]).item()
    y0 = yy_ref[0, 0].item()

    space = {}
    time = {}

    with logger.section(f"Performing {order} dyadic subdivisions..."):
        for p in range(order):
            # Subdivide Lxy_max
            l_p = Lxy_max / 2**p
            # Compute k
            k = 2 * torch.pi / l_p
            # Compute associated std for spatial gaussian enveloppe
            sigma_xy = l_p * sigma_xy_l_p_ratio
            # Compute space patches width
            l_xy = l_xy_sigma_xy_ratio * sigma_xy
            # Compute number of patches requires in both direction
            nx = int(Lx / l_xy) + 1
            ny = int(Ly / l_xy) + 1
            # Compute required offset to center patches on the domain
            offset_x = Lx - (l_xy * (nx - 1))
            offset_y = Ly - (l_xy * (ny - 1))
            # Compute patch centers
            xc = [x0 + offset_x * 0.5 + i * l_xy for i in range(nx)]
            yc = [y0 + offset_y * 0.5 + i * l_xy for i in range(ny)]

            xyc = [(x, y) for x, y in itertools.product(xc, yc)]

            space[p] = {
                "centers": xyc,
                "kx": k,
                "ky": k,
                "sigma_x": sigma_xy,
                "sigma_y": sigma_xy,
                "numel": len(xyc),
            }
            # Compute std for time gaussian enveloppe
            sigma_t = sigma_t_sigma_xy_ratio * sigma_xy
            # Compute associated time patch size
            l_t = l_t_sigma_t_ratio * sigma_t
            # Compute required number of time patches
            nt = int(Lt_max // l_t) + 1

            if use_time_frontier_as_centers:
                # Compute adjusted patch lengths
                l_t = Lt_max / nt
                # Compute patch centers
                tc = [i * l_t for i in range(nt + 1)]
            else:
                # Compute required offset to center patches on the domain
                offset_t = Lt_max - (nt - 1) * l_t
                # Compute patch centers
                tc = [offset_t * 0.5 + i * l_t for i in range(nt)]

            time[p] = {
                "centers": tc,
                "sigma_t": sigma_t,
                "numel": len(tc),
            }
            msg = (
                f"[Subdivision {step(p + 1, order)}] {meters2text(l_xy)} x "
                f"{meters2text(l_xy)} space patches associated to"
                f" {sec2text(l_t)} long time patches."
            )
            logger.detail(msg)

    return space, time


def linear_decomposition(
    order: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
    *,
    Lxy_max: float,
    Lt_max: float,
    sigma_t_sigma_xy_ratio: float = 1.17,
    sigma_xy_l_p_ratio: float = 1.75 / torch.pi * 2 * sqrt(2),
    l_xy_sigma_xy_ratio: float = 2 * sqrt(log(2)),
    l_t_sigma_t_ratio: float = 2 * sqrt(log(2)),
    use_time_frontier_as_centers: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Compute wavelet parameters using linearily-spaced subdivision factors.

    Args:
        order (int): Decomposition order (number of subdivision to perform).
        xx_ref (torch.Tensor): Reference x locations to use to place centers.
        yy_ref (torch.Tensor): Reference y locations to use to place centers.
        Lxy_max (float): Distance to start subdividing from.
        Lt_max (float): Total time duration to localize time patches.
        sigma_t_sigma_xy_ratio (float, optional): Ratio between time gaussian
            enveloppe and space gaussian enveloppe. Default value is set so
            to roughly match 900 km to 20 days and 80 km to 2 days.
            Defaults to 1.1.
        sigma_xy_l_p_ratio (float, optional): Ratio between space gaussian
            enveloppe std and subdivide length. Defaults set so that σ' allows
            exp((-x²-y²)/σ') ~ cos²(πx/(2σ))cos²(πy/(2σ)) close to 0.
            Defaults to 1.75/π*2*√2.
        l_xy_sigma_xy_ratio (float, optional): Ratio between space patches
            width and gaussian enveloppe std. Default set so that enveloppes
            overlap at 0.5. Defaults to 2*√(log(2)).
        l_t_sigma_t_ratio (float, optional): Ratio between time patches
            width and gaussian enveloppe std. Default set so that enveloppes
            overlap at 0.5. Defaults to 2*√(log(2)).
        use_time_frontier_as_centers (bool, optional): Center gaussian
            enveloppes on time boundaries. If set to True, no guarante that
            time enveloppes will overlap at 0.5. Defaults to False.

    Returns:
        tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
            Space params, time params.
    """
    Lx = (xx_ref[-1, 0] - xx_ref[0, 0]).item()
    x0 = xx_ref[0, 0].item()
    Ly = (yy_ref[0, -1] - yy_ref[0, 0]).item()
    y0 = yy_ref[0, 0].item()

    space = {}
    time = {}

    with logger.section(f"Performing {order} linear subdivisions..."):
        for p in range(order):
            # Subdivide Lxy_max
            l_p = Lxy_max / (p + 1)
            # Compute k
            k = 2 * torch.pi / l_p
            # Compute associated std for spatial gaussian enveloppe
            sigma_xy = l_p * sigma_xy_l_p_ratio
            # Compute space patches width
            l_xy = l_xy_sigma_xy_ratio * sigma_xy
            # Compute number of patches requires in both direction
            nx = int(Lx / l_xy) + 1
            ny = int(Ly / l_xy) + 1
            # Compute required offset to center patches on the domain
            offset_x = Lx - (l_xy * (nx - 1))
            offset_y = Ly - (l_xy * (ny - 1))
            # Compute patch centers
            xc = [x0 + offset_x * 0.5 + i * l_xy for i in range(nx)]
            yc = [y0 + offset_y * 0.5 + i * l_xy for i in range(ny)]

            xyc = [(x, y) for x, y in itertools.product(xc, yc)]

            space[p] = {
                "centers": xyc,
                "kx": k,
                "ky": k,
                "sigma_x": sigma_xy,
                "sigma_y": sigma_xy,
                "numel": len(xyc),
            }
            # Compute std for time gaussian enveloppe
            sigma_t = sigma_t_sigma_xy_ratio * sigma_xy
            # Compute associated time patch size
            l_t = l_t_sigma_t_ratio * sigma_t
            # Compute required number of time patches
            nt = int(Lt_max // l_t) + 1

            if use_time_frontier_as_centers:
                # Compute adjusted patch lengths
                l_t = Lt_max / nt
                # Compute patch centers
                tc = [i * l_t for i in range(nt + 1)]
            else:
                # Compute required offset to center patches on the domain
                offset_t = Lt_max - (nt - 1) * l_t
                # Compute patch centers
                tc = [offset_t * 0.5 + i * l_t for i in range(nt)]

            time[p] = {
                "centers": tc,
                "sigma_t": sigma_t,
                "numel": len(tc),
            }

            msg = (
                f"[Subdivision {step(p + 1, order)}] {meters2text(l_xy)} x "
                f"{meters2text(l_xy)} space patches associated to"
                f" {sec2text(l_t)} long time patches."
            )
            logger.detail(msg)
    return space, time


def regular_subdivisions(
    order: int,
    xx_ref: torch.Tensor,
    yy_ref: torch.Tensor,
    *,
    Lxy_max: float,
    Lt_max: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Generate space parameters for the Wavelet basis.

    Args:
        order (int): Order of decomposition.
        xx_ref (torch.Tensor): X locations.
        yy_ref (torch.Tensor): Y locations.
        Lxy_max (float): Distance to start subdividing from.
        Lt_max (float): Total time duration to localize time patches.

    Returns:
        tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
            Space params, time params.
    """
    space = {}
    time = {}
    lx = (xx_ref[-1, 0] - xx_ref[0, 0]).cpu().item()
    ly = (yy_ref[0, -1] - yy_ref[0, 0]).cpu().item()
    Lx = lx if Lxy_max is None else Lxy_max
    Ly = ly if Lxy_max is None else Lxy_max
    ratio = 1 / sqrt(log(2))
    for p in range(order):
        Lx_p = Lx / 2**p
        Ly_p = Ly / 2**p
        kx_p = 2 * torch.pi / Lx_p
        ky_p = 2 * torch.pi / Ly_p

        lx_p = lx / 2**p
        xs = [xx_ref[0, 0] + (2 * k + 1) / 2 * lx_p for k in range(2**p)]
        ly_p = ly / 2**p
        ys = [yy_ref[0, 0] + (2 * k + 1) / 2 * ly_p for k in range(2**p)]

        centers = [
            (x.cpu().item(), y.cpu().item()) for x, y in itertools.product(xs, ys)
        ]

        sigma_x = lx_p / 2 * ratio  # For the gaussian enveloppe
        sigma_y = ly_p / 2 * ratio  # For the gaussian enveloppe

        space[p] = {
            "centers": centers,
            "kx": kx_p,
            "ky": ky_p,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "numel": len(centers),
        }

        lt_p = Lt_max / 2**p
        tc = [(2 * k + 1) / 2 * lt_p for k in range(2**p)]

        sigma_t = lt_p / 2 * ratio
        time[p] = {
            "centers": tc,
            "sigma_t": sigma_t,
            "numel": len(tc),
        }

    return space, time
