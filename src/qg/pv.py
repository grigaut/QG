"""Potential vorticity computation."""

import torch

from qg.fd import interp_TP, laplacian


def compute_q1_interior(
    psi1: torch.Tensor,
    psi2: torch.Tensor,
    H1: torch.Tensor,  # noqa: N803
    g1: torch.Tensor,
    g2: torch.Tensor,
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
        H1 (torch.Tensor): Top layer's reference thickness.
        g1 (torch.Tensor): Top layer's reduced gravity.
        g2 (torch.Tensor): Second layer's reduced gravity.
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
            - f0**2 * (1 / H1 / g1 + 1 / H1 / g2) * psi1[..., 1:-1, 1:-1]
            + f0**2 * (1 / H1 / g2) * psi2[..., 1:-1, 1:-1]
        )
        + beta_effect
    )


def compute_q2_2l_interior(
    psi1: torch.Tensor,
    psi2: torch.Tensor,
    H2: torch.Tensor,  # noqa: N803
    g2: torch.Tensor,
    dx: float,
    dy: float,
    f0: float,
    beta_effect: torch.Tensor,
) -> torch.Tensor:
    """Compute potential vorticity in the second layer interior.

    WARNING: This function considers ѱ₃ = 0.
    WARNING: This function only compute potential vorticity in
        the **interior**.

    Args:
        psi1 (torch.Tensor): Top layer stream function.
            └── (n_ens, 1, nx+1, ny+1)-shaped
        psi2 (torch.Tensor): Second layer stream function.
            └── (n_ens, 1, nx+1, ny+1)-shaped
        H2 (torch.Tensor): Top layer's reference thickness.
        g2 (torch.Tensor): Second layer's reduced gravity.
        dx (float): Horizontal distance step in the X direction.
        dy (float): Horizontal distance step in the Y direction.
        f0 (float): Coriolis parameter.
        beta_effect (torch.Tensor): Beta effect.
            └── (1, ny+1)-shaped

    Returns:
        torch.Tensor: Δѱ₂ - (f₀² / H₂ /g₂) (ѱ₂ - ѱ₁)
            └── (n_ens, 1, nx-2, ny-2)-shaped
    """
    return (
        interp_TP(
            laplacian(psi2, dx, dy)
            - f0**2 * (1 / H2 / g2) * (psi2[..., 1:-1, 1:-1] - psi1[..., 1:-1, 1:-1])
        )
        + beta_effect
    )
