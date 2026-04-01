"""Potential vorticity computation."""

import torch

from qg.fd import interp_TP, laplacian


def compute_q1_interior(
    psi1: torch.Tensor,
    psi2: torch.Tensor,
    A11: torch.Tensor,
    A12: torch.Tensor,
    dx: float,
    dy: float,
    f0: float,
    beta_effect: torch.Tensor,
) -> torch.Tensor:
    """Compute potential vorticity in the top layer interior.

    WARNING: This function only compute potential vorticity in
        the **interior**.

    Args:
        psi1 (torch.Tensor): Top layer stream function.
            └── (n_ens, 1, nx+1, ny+1)-shaped
        psi2 (torch.Tensor): Second layer stream function.
            └── (n_ens, 1, nx+1, ny+1)-shaped
        A11 (torch.Tensor): 1st row, 1st column component of the
            stretching matrix
        A12 (torch.Tensor): 1st row, 2nd column component of the
            stretching matrix
        dx (float): Horizontal distance step in the X direction.
        dy (float): Horizontal distance step in the Y direction.
        f0 (float): Coriolis parameter.
        beta_effect (torch.Tensor): Beta effect.
            └── (1, ny+1)-shaped

    Returns:
        torch.Tensor: Δѱ₁ - f₀² / H₁ (1/g₁ + 1/g₂) ѱ₁ + (f₀² / H₁ /  g₂) ѱ₂
            └── (n_ens, 1, nx-2, ny-2)-shaped
    """
    return (
        interp_TP(
            laplacian(psi1, dx, dy)
            - f0**2 * (A11 * psi1[..., 1:-1, 1:-1] + A12 * psi2[..., 1:-1, 1:-1])
        )
        + beta_effect
    )
