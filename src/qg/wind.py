"""Generate wind forcing."""

import torch


def compute_double_gyre_wind_curl(
    tau0: float, xv: torch.Tensor, yv: torch.Tensor, n_ens: int = 1
) -> torch.Tensor:
    """Compute wind curl.

    Args:
        xv (torch.Tensor): X vertices.
        yv (torch.Tensor): Y vertices.
        tau0 (float): Wind magnitude.
        n_ens (int): Ensemble_number.

    Returns:
        torch.Tensor: Wind curl.
    """
    nx = xv.shape[0] - 1
    Ly = yv[-1] - yv[0]
    yc = 0.5 * (yv[1:] + yv[:-1])  # cell centers
    curl_tau = (
        -tau0 * 2 * torch.pi / Ly * torch.sin(2 * torch.pi * yc / Ly).tile((nx, 1))
    )
    return curl_tau.unsqueeze(0).repeat(n_ens, 1, 1, 1)
