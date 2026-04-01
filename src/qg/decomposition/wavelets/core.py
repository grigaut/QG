"""Wavelets implementation."""

from __future__ import annotations

from typing import Any

from qg.decomposition.base import SpaceTimeDecomposition
from qg.decomposition.coefficients import DecompositionCoefs
from qg.decomposition.supports.space.cosine import (
    CosineBasisFunctions,
)
from qg.decomposition.supports.space.gaussian import (
    GaussianSupport,
    NormalizedGaussianSupport,
)
from qg.decomposition.supports.space.gaussian_cosine import ExpCosSupport
from qg.decomposition.supports.time.gaussian import (
    GaussianTimeSupport,
)
from qg.decomposition.wavelets.param_generators import (
    dyadic_decomposition,
    linear_decomposition,
)

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch


class WaveletBasis(SpaceTimeDecomposition[ExpCosSupport, GaussianTimeSupport]):
    """Wavelet decomposition.

    ΣΣ[E(t)/ΣE(t)]Σe(x,y)ΣΣcγ(x,y)

    E(t) = exp(-(t-tc)²/σ_t²)
    e(x,y) = E(x,y) / ΣE(x,y)
    E(x,y) = exp(-(x-xc)²/σ_x²)exp(-(y-yc)²/σ_y²)
    γ(x,y) = cos(kx x cos(θ) + ky y sin(θ) + φ)
    """

    type = "wavelets"
    _n_theta = 10

    @property
    def n_theta(self) -> int:
        """Number of orientations to consider."""
        return self._n_theta

    @n_theta.setter
    def n_theta(self, n_theta: int) -> None:
        self._n_theta = n_theta
        theta = torch.linspace(0, torch.pi, self.n_theta + 1, **self._specs)[:-1]
        self._cos_t = torch.cos(theta)
        self._sin_t = torch.sin(theta)

    def __init__(
        self,
        space_params: dict[int, dict[str, Any]],
        time_params: dict[int, dict[str, Any]],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate the Wavelet basis.

        Args:
            space_params (dict[int, dict[str, Any]]): Space parameters.
            time_params (dict[int, dict[str, Any]]): Time parameters.
            dtype (torch.dtype | None, optional): Data type.
                Defaults to None.
            device (torch.device | None, optional): Ddevice.
                Defaults to None.
        """
        super().__init__(space_params, time_params, dtype=dtype, device=device)
        self.n_theta = self._n_theta
        self.phase = torch.tensor([0, torch.pi / 2], **self._specs)

    def numel(self) -> int:
        """Total number of elements."""
        n = sum(s["numel"] * self._time[k]["numel"] for k, s in self._space.items())

        return n * self.phase.numel() * self.n_theta

    def generate_random_coefs(self) -> DecompositionCoefs:
        """Generate random coefficient.

        Useful to properly instantiate coefs.

        Returns:
            DecompositionCoefs: Level -> coefficients.
                ├── 0: (1, 1)-shaped
                ├── 1: (2, 4)-shaped
                ├── 2: (4, 16)-shaped
                ├── ...
                ├── p: (2**p, (2**p)**2)-shaped
                ├── ...
                └── order: (2**order, (2**order)**2)-shaped
        """
        coefs = {}
        for k in self._space:
            coefs[k] = torch.randn(
                (
                    self._time[k]["numel"],
                    self._space[k]["numel"],
                    self.n_theta,
                    2,
                ),
                **self._specs,
            )

        return DecompositionCoefs.from_dict(coefs)

    def _compute_space_params(
        self, params: dict[str, Any], xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        centers = params["centers"]
        kx = params["kx"]
        ky = params["ky"]
        xc = torch.tensor([c[0] for c in centers], **self._specs)
        yc = torch.tensor([c[1] for c in centers], **self._specs)

        x = xx[None, :, :] - xc[:, None, None]
        y = yy[None, :, :] - yc[:, None, None]
        return (x, y, kx, ky)

    def _build_space(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build space-related fields.

        Σe(x,y)ΣΣcγ(x,y)

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop",
                c,
                space_support.field,
            ).mean(dim=[-1, -2])
        return fields

    def _build_space_dx(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build x-derivatives of space fields.

        ΣΣΣc[e'(x,y)γ(x,y)+e'(x,y)γ'(x,y)]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)

            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, space_support.dx
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dx2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build second order x-derivatives of space fields.

        ΣΣΣc[e''(x,y)γ(x,y) + 2e(x,y)'γ'(x,y) + e(x,y)γ''(x,y)]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)

            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, space_support.dx2
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dx3(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build third order x-derivatives of space fields.

        ΣΣΣc[e'''γ + 3e''γ'+ 3e'γ'' +eγ''']

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )
            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)

            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop",
                coefs,
                space_support.dx3,
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dydx2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build the x-x-y derivative of space fields.

        ΣΣΣc[e_xxy γ + e_xx γ_y + 2e_xy γ + 2e_x γ_y + e_y γ_xx + e γ_xxy]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)

            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop",
                coefs,
                space_support.dydx2,
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dy(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build t-derivatives of space fields.

        ΣΣΣc[e'(x,y)γ(x,y)+e'(x,y)γ'(x,y)]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )
            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)

            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, space_support.dy
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dy2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build second order y-derivatives of space fields.

        ΣΣΣc[e''(x,y)γ(x,y) + 2e(x,y)'γ'(x,y) + e(x,y)γ''(x,y)]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )
            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)

            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, space_support.dy2
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dy3(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build third order y-derivatives of space fields.

        ΣΣΣc[e'''γ + 3e''γ'+ 3e'γ'' +eγ''']

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, space_support.dy3
            ).mean(dim=[-1, -2])

        return fields

    def _build_space_dxdy2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build the y-y-x derivative of space fields.

        ΣΣΣc[e_yyx γ + e_yy γ_x + 2e_yx γ + 2e_y γ_x + e_x γ_yy + e γ_yyx]

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            dict[int, torch.Tensor]: Level -> field
        """
        fields = {}

        for lvl, coefs in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y, kx, ky = self._compute_space_params(params, xx, yy)

            phase = self.phase[None, None, None, None, :]

            gamma = CosineBasisFunctions(
                x,
                y,
                kx,
                ky,
                self._cos_t,
                self._sin_t,
                phase,
            )

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)

            space_support = ExpCosSupport(e, gamma)
            fields[lvl] = torch.einsum(
                "tcop,cxyop->txyop", coefs, space_support.dxdy2
            ).mean(dim=[-1, -2])

        return fields

    def freeze_time_normalization(self, t: torch.Tensor) -> None:
        """Freeze time normalization.

        Args:
            t (torch.Tensor): Time to freeze normalization at.
        """
        self.generate_time_support = (
            lambda time_params, space_fields: self._generate_frozen_time_support(
                t, time_params, space_fields
            )
        )

    def unfreeze_time_normalization(self) -> None:
        """Unfreeze time normalization."""
        self.generate_time_support = self._generate_time_support

    def _generate_frozen_time_support(
        self,
        t: torch.Tensor,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, torch.Tensor],
    ) -> GaussianTimeSupport:
        """Generate frozen time support.

        Args:
            t (torch.Tensor): Time to freeze normalization at.
            time_params (dict[int, dict[str, Any]]): Time parameters.
            space_fields (dict[int, torch.Tensor]): Space fields.

        Returns:
            GaussianTimeSupport: Frozen gaussian time support.
        """
        gts = GaussianTimeSupport(time_params, space_fields)
        gts.freeze_normalization(t)
        return gts

    def _generate_time_support(
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, torch.Tensor],
    ) -> GaussianTimeSupport:
        """Generate time support.

        Args:
            time_params (dict[int, dict[str, Any]]): Time parameters.
            space_fields (dict[int, torch.Tensor]): Space fields.

        Returns:
            GaussianTimeSupport: Gaussian time support.
        """
        return GaussianTimeSupport(time_params, space_fields)

    generate_time_support = _generate_time_support

    @classmethod
    def from_dyadic_decomposition(
        cls,
        order: int,
        xx_ref: torch.Tensor,
        yy_ref: torch.Tensor,
        Lxy_max: float,
        Lt_max: float,
    ) -> Self:
        """Generate space and time basis parameters to instantiate the object.

        Args:
            order (int): "Depth" of decomposition.
            xx_ref (torch.Tensor): X locations to use as reference.
            yy_ref (torch.Tensor): Y locations to use as reference.
            Lxy_max (float): Max horizontal scale.
            Lt_max (float): Max time scale.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: WaveletBasis.
        """
        space_params, time_params = dyadic_decomposition(
            order=order,
            xx_ref=xx_ref,
            yy_ref=yy_ref,
            Lxy_max=Lxy_max,
            Lt_max=Lt_max,
        )
        return cls(space_params, time_params)

    @classmethod
    def from_linear_decomposition(
        cls,
        order: int,
        xx_ref: torch.Tensor,
        yy_ref: torch.Tensor,
        Lxy_max: float,
        Lt_max: float,
    ) -> Self:
        """Generate space and time basis parameters to instantiate the object.

        Args:
            order (int): "Depth" of decomposition.
            xx_ref (torch.Tensor): X locations to use as reference.
            yy_ref (torch.Tensor): Y locations to use as reference.
            Lxy_max (float): Max horizontal scale.
            Lt_max (float): Max time scale.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: WaveletBasis.
        """
        space_params, time_params = linear_decomposition(
            order=order,
            xx_ref=xx_ref,
            yy_ref=yy_ref,
            Lxy_max=Lxy_max,
            Lt_max=Lt_max,
        )
        return cls(space_params, time_params)

    def get_params(self) -> dict[str, Any]:
        """Return decomposition params as dict.

        Returns:
            dict[str, Any]: Decomposition params.
        """
        params = super().get_params()
        params["n_theta"] = self.n_theta
        return params

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        """Build class from params dict.

        Args:
            params (dict[str, Any]): Decomposition params.

        Returns:
            Self: Instance of class.
        """
        obj = cls(space_params=params["space"], time_params=params["time"])
        obj.n_theta = params["n_theta"]
        return obj
