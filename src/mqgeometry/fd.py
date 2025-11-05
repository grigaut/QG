import torch
import torch.nn.functional as F


def grad_perp(f, dx, dy):
    """Orthogonal gradient"""
    return (f[..., :-1] - f[..., 1:]) / dy, (f[..., 1:, :] - f[..., :-1, :]) / dx


def interp_TP(f):
    return 0.25 * (
        f[..., 1:, 1:] + f[..., 1:, :-1] + f[..., :-1, 1:] + f[..., :-1, :-1]
    )


def laplacian1D(f: torch.Tensor, dx: float) -> torch.Tensor:
    """3-points discrete 1D laplacian.

    Δu[i] ≈ (u[i+1]-2*u[i]+u[i-1])/dx**2

    Args:
        f (torch.Tensor): Function to differentiate.
            └── (..., nx)-shaped

        dx (float): Horizontal step.

    Returns:
        torch.Tensor: Laplacian of f.
            └── (..., nx-2)-shaped
    """
    return (f[..., 2:] - 2 * f[..., 1:-1] + f[..., :-2]) / dx**2


def laplacian(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Non-padded laplacian."""
    return (f[..., 2:, 1:-1] + f[..., :-2, 1:-1] - 2 * f[..., 1:-1, 1:-1]) / dx**2 + (
        f[..., 1:-1, 2:] + f[..., 1:-1, :-2] - 2 * f[..., 1:-1, 1:-1]
    ) / dy**2


def laplacian_h(f, dx, dy):
    return F.pad(
        (f[..., 2:, 1:-1] + f[..., :-2, 1:-1] - 2 * f[..., 1:-1, 1:-1]) / dx**2
        + (f[..., 1:-1, 2:] + f[..., 1:-1, :-2] - 2 * f[..., 1:-1, 1:-1]) / dy**2,
        (1, 1, 1, 1),
        mode="constant",
        value=0.0,
    )
