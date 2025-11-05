"""Test inhomogeneous pv solver."""

import pytest
import torch

from mqgeometry.stretching_matrix import compute_A
from mqgeometry.solver.boundary_conditions.base import Boundaries
from mqgeometry.fd import interp_TP, laplacian
from mqgeometry.solver.pv_inversion import (
    InhomogeneousPVInversion,
    InhomogeneousPVInversionCollinear,
)
from mqgeometry.specs import defaults


@pytest.fixture
def shape() -> tuple[int, int, int]:
    """Shape."""
    return (3, 25, 40)


@pytest.fixture
def pv(shape: tuple[int, int, int]) -> torch.Tensor:
    """Random potential vorticity."""
    return torch.rand(shape, **defaults.get())


@pytest.fixture
def boundaries(shape: tuple[int, int, int, int]) -> Boundaries:
    """Boundary conditions."""
    nl, nx, ny = shape
    x = torch.linspace(0, 2 * torch.pi, nx + 1, **defaults.get())
    y = torch.linspace(0, 2 * torch.pi, ny + 1, **defaults.get())

    top = torch.stack([torch.sin((i + 1) * x).unsqueeze(-1) for i in range(nl)])
    bottom = torch.stack([2 * torch.sin((i + 1) * x).unsqueeze(-1) for i in range(nl)])
    left = torch.stack([torch.sin((i + 1) * y).unsqueeze(-2) for i in range(nl)])
    right = torch.stack([2 * torch.sin((i + 1) * y).unsqueeze(-2) for i in range(nl)])
    return Boundaries(top, bottom, left, right)


def test_solver_boundaries(pv: torch.Tensor, boundaries: Boundaries) -> None:
    """Test solver behavior by checking bioundaries."""
    A = torch.diag(  # noqa: N806
        torch.tensor([i + 1 for i in range(pv.shape[0])], **defaults.get())
    )
    solver = InhomogeneousPVInversion(A, 1, 1, 1)
    solver.set_boundaries(boundaries)
    pv_i = interp_TP(pv)
    sf = solver.compute_stream_function(pv_i)
    torch.testing.assert_close(sf[0, ..., :, -1], boundaries.top[..., :, 0])
    torch.testing.assert_close(sf[0, ..., :, 0], boundaries.bottom[..., :, 0])
    torch.testing.assert_close(sf[0, ..., 0, :], boundaries.left[..., 0, :])
    torch.testing.assert_close(sf[0, ..., -1, :], boundaries.right[..., 0, :])


def test_solver() -> None:
    """Test the inhomoegensous solver."""
    H = torch.tensor([400, 1100], **defaults.get())  # noqa: N806
    g_prime = torch.tensor([9.81, 0.025], **defaults.get())
    f0 = 1e-4
    dx = 1e4
    dy = 1e4

    A = compute_A(H, g_prime, **defaults.get())  # noqa: N806

    psi = torch.rand((2, 2, 50, 75), **defaults.get())

    psi_bc = Boundaries.extract(psi, 0, -1, 0, -1, width=1)

    q = laplacian(psi, dx, dy) - f0**2 * torch.einsum(
        "lm,...mxy->...lxy", A, psi[..., 1:-1, 1:-1]
    )
    solver = InhomogeneousPVInversion(A, f0, dx, dy)
    solver.set_boundaries(psi_bc)
    psi_ = solver.compute_stream_function(q)

    torch.testing.assert_close(psi, psi_, rtol=1e-12, atol=1e-12)


def test_collinear_solver() -> None:
    """Test the inhomogeneous solver."""
    H = torch.tensor([400, 1100], **defaults.get())  # noqa: N806
    g_prime = torch.tensor([9.81, 0.025], **defaults.get())
    f0 = 1e-4
    dx = 1e4
    dy = 1e4

    A = compute_A(H, g_prime, **defaults.get())  # noqa: N806

    A_11 = A[0, 0]  # noqa: N806
    A_12 = A[0, 1]  # noqa: N806

    psi = torch.rand((2, 1, 50, 75), **defaults.get())
    alpha = torch.ones_like(psi) * 0.5

    psi_bc = Boundaries.extract(psi, 0, -1, 0, -1, width=1)

    q = (
        laplacian(psi, dx, dy)
        - f0**2 * (A_11 + alpha[..., 1:-1, 1:-1] * A_12) * psi[..., 1:-1, 1:-1]
    )
    solver = InhomogeneousPVInversionCollinear(A, alpha, f0, dx, dy)
    solver.set_boundaries(psi_bc)
    psi_ = solver.compute_stream_function(q)

    torch.testing.assert_close(psi, psi_, rtol=1e-12, atol=1e-12)
