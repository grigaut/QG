"""Wavelet basis functions."""

from functools import cached_property

import torch


class CosineBasisFunctions:
    """Cosine basis functions."""

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
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kx: float,
        ky: float,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
        phase: torch.Tensor,
    ) -> None:
        """Instantiate the class.

        Args:
            x (torch.Tensor): X locations.
            y (torch.Tensor): Y locations.
            kx (float): X wave number.
            ky (float): Y wave number.
            cos_t (torch.Tensor): Cosine of orientations.
            sin_t (torch.Tensor): Sine of orientations.
            phase (torch.Tensor): Phases.
        """
        kx_cos = kx * torch.einsum("cxy,o->cxyo", x, cos_t)
        ky_sin = ky * torch.einsum("cxy,o->cxyo", y, sin_t)
        self.cos_t = cos_t
        self.sin_t = sin_t
        self.cos = torch.cos((kx_cos + ky_sin)[..., None] + phase)
        self.sin = torch.sin((kx_cos + ky_sin)[..., None] + phase)
        self.kx = kx
        self.ky = ky

    def _compute(self) -> torch.Tensor:
        return self.cos

    def _compute_dx(self) -> torch.Tensor:
        return -self.kx * torch.einsum("o,cxyop->cxyop", self.cos_t, self.sin)

    def _compute_dy(self) -> torch.Tensor:
        return -self.ky * torch.einsum("o,cxyop->cxyop", self.sin_t, self.sin)

    def _compute_dx2(self) -> torch.Tensor:
        return -(self.kx**2) * torch.einsum(
            "o,cxyop->cxyop", self.cos_t.pow(2), self.cos
        )

    def _compute_dy2(self) -> torch.Tensor:
        return -(self.ky**2) * torch.einsum(
            "o,cxyop->cxyop", self.sin_t.pow(2), self.cos
        )

    def _compute_dydx(self) -> torch.Tensor:
        return -(self.ky * self.kx) * torch.einsum(
            "o,cxyop->cxyop", self.sin_t * self.cos_t, self.cos
        )

    def _compute_dxdy(self) -> torch.Tensor:
        return -(self.kx * self.ky) * torch.einsum(
            "o,cxyop->cxyop", self.cos_t * self.sin_t, self.cos
        )

    def _compute_dx3(self) -> torch.Tensor:
        return (self.kx**3) * torch.einsum(
            "o,cxyop->cxyop", self.cos_t.pow(3), self.sin
        )

    def _compute_dydx2(self) -> torch.Tensor:
        return (self.kx**2 * self.ky) * torch.einsum(
            "o,cxyop->cxyop", self.sin_t * self.cos_t.pow(2), self.sin
        )

    def _compute_dy3(self) -> torch.Tensor:
        return (self.ky**3) * torch.einsum(
            "o,cxyop->cxyop", self.sin_t.pow(3), self.sin
        )

    def _compute_dxdy2(self) -> torch.Tensor:
        return (self.ky**2 * self.kx) * torch.einsum(
            "o,cxyop->cxyop", self.cos_t * self.sin_t.pow(2), self.sin
        )


class CosineBasisFunctions1D:
    """Cosine basis functions."""

    @cached_property
    def field(self) -> torch.Tensor:
        """Field."""
        return self._compute()

    @cached_property
    def dx(self) -> torch.Tensor:
        """X-derivative."""
        return self._compute_dx()

    @cached_property
    def dx2(self) -> torch.Tensor:
        """Second X-derivative."""
        return self._compute_dx2()

    @cached_property
    def dx3(self) -> torch.Tensor:
        """Third X-derivative."""
        return self._compute_dx3()

    def __init__(
        self,
        x: torch.Tensor,
        kx: float,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
        phase: torch.Tensor,
    ) -> None:
        """Instantiate the class.

        Args:
            x (torch.Tensor): X locations.
            kx (float): X wave number.
            cos_t (torch.Tensor): Cosine of orientations.
            sin_t (torch.Tensor): Sine of orientations.
            phase (torch.Tensor): Phases.
        """
        kx_cos = kx * torch.einsum("cx,o->cxo", x, cos_t)
        self.cos_t = cos_t
        self.sin_t = sin_t
        self.cos = torch.cos((kx_cos)[..., None] + phase)
        self.sin = torch.sin((kx_cos)[..., None] + phase)
        self.kx = kx

    def _compute(self) -> torch.Tensor:
        return self.cos

    def _compute_dx(self) -> torch.Tensor:
        return -self.kx * torch.einsum("o,cxop->cxop", self.cos_t, self.sin)

    def _compute_dx2(self) -> torch.Tensor:
        return -(self.kx**2) * torch.einsum(
            "o,cxop->cxop", self.cos_t.pow(2), self.cos
        )

    def _compute_dx3(self) -> torch.Tensor:
        return (self.kx**3) * torch.einsum(
            "o,cxop->cxop", self.cos_t.pow(3), self.sin
        )
