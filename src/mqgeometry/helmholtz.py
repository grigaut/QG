"""
Helmholtz equation solver with type-I discrete sine transform
and capacitance matrix method.
Louis Thiry, 2023.
"""

import torch
import torch.nn.functional as F


def dstI1D(x, norm="ortho"):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j * F.pad(x, (1, 1)), dim=-1, norm=norm)[
        ..., 1 : x.shape[-1] + 1
    ]


def dstI2D(x, norm="ortho"):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1, -2), norm=norm).transpose(-1, -2)


def compute_laplace_dst(nx, ny, dx, dy, arr_kwargs) -> torch.Tensor:
    """Discrete sine transform of the 2D centered discrete laplacian
    operator."""
    x, y = torch.meshgrid(
        torch.arange(1, nx, **arr_kwargs),
        torch.arange(1, ny, **arr_kwargs),
        indexing="ij",
    )
    return (
        2 * (torch.cos(torch.pi / nx * x) - 1) / dx**2
        + 2 * (torch.cos(torch.pi / ny * y) - 1) / dy**2
    )


def solve_helmholtz_dst(rhs: torch.Tensor, helmholtz_dst: torch.Tensor) -> torch.Tensor:
    return F.pad(
        dstI2D(dstI2D(rhs.type(helmholtz_dst.dtype)) / helmholtz_dst),
        (1, 1, 1, 1),
    ).type(rhs.dtype)


def compute_capacitance_matrices(helmholtz_dst, bound_xids, bound_yids):
    nl = helmholtz_dst.shape[-3]
    M = bound_xids.shape[0]

    # compute G matrices
    G_matrices = torch.zeros((nl, M, M), dtype=torch.float64, device="cpu")
    rhs = torch.zeros(
        helmholtz_dst.shape[-3:], dtype=torch.float64, device=helmholtz_dst.device
    )
    for m in range(M):
        rhs.fill_(0)
        rhs[..., bound_xids[m], bound_yids[m]] = 1
        sol = dstI2D(dstI2D(rhs) / helmholtz_dst.type(torch.float64))
        G_matrices[:, m] = sol[..., bound_xids, bound_yids].cpu()

    # invert G matrices to get capacitance matrices
    capacitance_matrices = torch.zeros_like(G_matrices)
    for l in range(nl):
        capacitance_matrices[l] = torch.linalg.inv(G_matrices[l])

    return capacitance_matrices.to(helmholtz_dst.device)


def solve_helmholtz_dst_cmm(
    rhs, helmholtz_dst, cap_matrices, bound_xids, bound_yids, mask
):
    sol_rect = dstI2D(dstI2D(rhs.type(helmholtz_dst.dtype)) / helmholtz_dst)
    alphas = torch.einsum(
        "...ij,...j->...i",
        cap_matrices,
        -sol_rect[..., bound_xids, bound_yids].type(torch.float64),
    )
    rhs_2 = rhs.clone()
    rhs_2[..., bound_xids, bound_yids] = alphas
    sol = dstI2D(dstI2D(rhs_2.type(helmholtz_dst.dtype)) / helmholtz_dst).type(
        torch.float64
    )
    return F.pad(sol, (1, 1, 1, 1)) * mask
