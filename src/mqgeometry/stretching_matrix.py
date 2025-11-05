"""Stretching matrix-related functions."""

from __future__ import annotations

import torch

from mqgeometry.specs import defaults


def compute_A(  # noqa: N802
    H: torch.Tensor,  # noqa: N803
    g_prime: torch.Tensor,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the stretching operator matrix A.

    Args:
        H (torch.Tensor): Layers reference height.
            └── (nl, )-shaped
        g_prime (torch.Tensor): Reduced gravity values.
            └── (nl, )-shaped
        dtype (torch.dtype): Data type
        device (torch.device | None): Device. Defaults to None.

    Returns:
        torch.Tensor: Streching operator matrix
            └── (nl, nl)-shaped
    """
    nl = H.shape[0]
    if nl == 1:
        return torch.tensor(
            [[1.0 / (H * g_prime)]], **defaults.get(dtype=dtype, device=device)
        )
    A = torch.zeros(  # noqa: N806
        (nl, nl), **defaults.get(dtype=dtype, device=device)
    )
    A[0, 0] = 1.0 / (H[0] * g_prime[0]) + 1.0 / (H[0] * g_prime[1])
    A[0, 1] = -1.0 / (H[0] * g_prime[1])
    for i in range(1, nl - 1):
        A[i, i - 1] = -1.0 / (H[i] * g_prime[i])
        A[i, i] = 1.0 / H[i] * (1 / g_prime[i + 1] + 1 / g_prime[i])
        A[i, i + 1] = -1.0 / (H[i] * g_prime[i + 1])
    A[-1, -1] = 1.0 / (H[nl - 1] * g_prime[nl - 1])
    A[-1, -2] = -1.0 / (H[nl - 1] * g_prime[nl - 1])
    return A


def compute_layers_to_mode_decomposition(
    A: torch.Tensor,  # noqa: N803
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Layers to mode decomposition.

    A = Cm2l @ Λ @ Cl2m

    Eigen values are sorted in descending order. Hence, if the model has 3
    layers, first eigen value will correspond to second baroclinic mode,
    second eigen value to first baroclinic mode and last eigen value to the
    barotropic mode.

    Args:
        A (torch.Tensor): Stretching Operator.
            └── (nl, nl)-shaped

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Cm2l, Λ, Cl2m.
            ├── Cm2l: (nl, nl)-shaped
            ├── Λ: (nl,)-shaped
            └── Cl2m: (nl, nl)-shaped
    """
    # layer-to-mode and mode-to-layer matrices
    lambd_r, R = torch.linalg.eig(A)  # noqa: N806
    _, L = torch.linalg.eig(A.T)  # noqa: N806
    lambd: torch.Tensor = lambd_r.real
    R, L = R.real, L.real  # noqa: N806
    # Diagonalization of A: A = Cm2l @ Λ @ Cl2m
    # layer to mode
    Cl2m = torch.diag(1.0 / torch.diag(L.T @ R)) @ L.T  # noqa: N806
    # mode to layer
    Cm2l = R  # noqa: N806
    return Cm2l, lambd, Cl2m
