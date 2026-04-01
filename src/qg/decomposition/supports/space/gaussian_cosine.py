"""Gaussian and Cosine space functions."""

from functools import cached_property

import torch

from qg.decomposition.supports.space.cosine import (
    CosineBasisFunctions,
)
from qg.decomposition.supports.space.gaussian import (
    NormalizedGaussianSupport,
)


class ExpCosSupport:
    """Space support using cosine and exponential functions."""

    @cached_property
    def field(self) -> torch.Tensor:
        """Field."""
        return self._compute()

    @cached_property
    def dx(self) -> torch.Tensor:
        """X-derivative."""
        return self._compute_dx()

    @cached_property
    def dy(self) -> torch.Tensor:
        """Y-derivative."""
        return self._compute_dy()

    @cached_property
    def dx2(self) -> torch.Tensor:
        """Second X-derivative."""
        return self._compute_dx2()

    @cached_property
    def dydx(self) -> torch.Tensor:
        """X-Y-derivative."""
        return self._compute_dydx()

    @cached_property
    def dy2(self) -> torch.Tensor:
        """Second Y-derivative."""
        return self._compute_dy2()

    @cached_property
    def dxdy(self) -> torch.Tensor:
        """Y-X-derivative."""
        return self._compute_dxdy()

    @cached_property
    def dx3(self) -> torch.Tensor:
        """Third X-derivative."""
        return self._compute_dx3()

    @cached_property
    def dy3(self) -> torch.Tensor:
        """Third Y-derivative."""
        return self._compute_dy3()

    @cached_property
    def dydx2(self) -> torch.Tensor:
        """X-X-Y-derivative."""
        return self._compute_dydx2()

    @cached_property
    def dxdy2(self) -> torch.Tensor:
        """Y-Y-X-derivative."""
        return self._compute_dxdy2()

    def __init__(
        self, exp_field: NormalizedGaussianSupport, gamma: CosineBasisFunctions
    ) -> None:
        """Instantite the space support basis.

        Args:
            exp_field (NormalizedGaussianSupport): Exponential field.
            gamma (CosineBasisFunctions): Cosine functions.
        """
        self._gamma = gamma
        self._exp = exp_field

    def _compute(self) -> torch.Tensor:
        return torch.einsum("cxy,cxyop->cxyop", self._exp.field, self._gamma.field)

    def _compute_dx(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dx_e_g = torch.einsum("cxy,cxyop->cxyop", e.dx, g.field)
        e_dx_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dx)
        return dx_e_g + e_dx_g

    def _compute_dy(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dy_e_g = torch.einsum("cxy,cxyop->cxyop", e.dy, g.field)
        e_dy_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dy)
        return dy_e_g + e_dy_g

    def _compute_dx2(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dx2_e_g = torch.einsum("cxy,cxyop->cxyop", e.dx2, g.field)
        dx_e_dx_g = 2 * torch.einsum("cxy,cxyop->cxyop", e.dx, g.dx)
        e_dx2_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dx2)
        return dx2_e_g + dx_e_dx_g + e_dx2_g

    def _compute_dy2(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dy2_e_g = torch.einsum("cxy,cxyop->cxyop", e.dy2, g.field)
        dy_e_dy_g = 2 * torch.einsum("cxy,cxyop->cxyop", e.dy, g.dy)
        e_dy2_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dy2)
        return dy2_e_g + dy_e_dy_g + e_dy2_g

    def _compute_dydx(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dxdy_e_g = torch.einsum("cxy,cxyop->cxyop", e.dxdy, g.field)
        dx_e_dy_g = torch.einsum("cxy,cxyop->cxyop", e.dx, g.dy)
        dy_e_dx_g = torch.einsum("cxy,cxyop->cxyop", e.dy, g.dx)
        e_dxdy_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dxdy)

        return dxdy_e_g + dx_e_dy_g + dy_e_dx_g + e_dxdy_g

    def _compute_dxdy(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dydx_e_g = torch.einsum("cxy,cxyop->cxyop", e.dydx, g.field)
        dy_e_dx_g = torch.einsum("cxy,cxyop->cxyop", e.dy, g.dx)
        dx_e_dy_g = torch.einsum("cxy,cxyop->cxyop", e.dx, g.dy)
        e_dydx_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dydx)

        return dydx_e_g + dx_e_dy_g + dy_e_dx_g + e_dydx_g

    def _compute_dx3(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dx3_e_g = torch.einsum("cxy,cxyop->cxyop", e.dx3, g.field)
        dx2_e_dx_g = torch.einsum("cxy,cxyop->cxyop", e.dx2, g.dx)
        dx_e_dx2_g = torch.einsum("cxy,cxyop->cxyop", e.dx, g.dx2)
        e_dx3_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dx3)

        return dx3_e_g + 3 * (dx2_e_dx_g + dx_e_dx2_g) + e_dx3_g

    def _compute_dydx2(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dx2dy_e_g = torch.einsum("cxy,cxyop->cxyop", e.dydx2, g.field)
        dx2_e_dy_g = torch.einsum("cxy,cxyop->cxyop", e.dx2, g.dy)
        dxdy_e_dx_g = torch.einsum("cxy,cxyop->cxyop", e.dydx, g.dx)
        dx_e_dxdy_g = torch.einsum("cxy,cxyop->cxyop", e.dx, g.dydx)
        dy_e_dx2_g = torch.einsum("cxy,cxyop->cxyop", e.dy, g.dx2)
        e_dx2dy_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dydx2)
        return (
            dx2dy_e_g
            + dx2_e_dy_g
            + 2 * (dxdy_e_dx_g + dx_e_dxdy_g)
            + dy_e_dx2_g
            + e_dx2dy_g
        )

    def _compute_dy3(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dy3_e_g = torch.einsum("cxy,cxyop->cxyop", e.dy3, g.field)
        dy2_e_dy_g = torch.einsum("cxy,cxyop->cxyop", e.dy2, g.dy)
        dy_e_dy2_g = torch.einsum("cxy,cxyop->cxyop", e.dy, g.dy2)
        e_dy3_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dy3)

        return dy3_e_g + 3 * (dy2_e_dy_g + dy_e_dy2_g) + e_dy3_g

    def _compute_dxdy2(self) -> torch.Tensor:
        e = self._exp
        g = self._gamma

        dy2dx_e_g = torch.einsum("cxy,cxyop->cxyop", e.dxdy2, g.field)
        dy2_e_dx_g = torch.einsum("cxy,cxyop->cxyop", e.dy2, g.dx)
        dydx_e_dy_g = torch.einsum("cxy,cxyop->cxyop", e.dxdy, g.dy)
        dy_e_dydx_g = torch.einsum("cxy,cxyop->cxyop", e.dy, g.dxdy)
        dx_e_dy2_g = torch.einsum("cxy,cxyop->cxyop", e.dx, g.dy2)
        e_dy2dx_g = torch.einsum("cxy,cxyop->cxyop", e.field, g.dxdy2)
        return (
            dy2dx_e_g
            + dy2_e_dx_g
            + 2 * (dydx_e_dy_g + dy_e_dydx_g)
            + dx_e_dy2_g
            + e_dy2dx_g
        )
