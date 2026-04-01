"""Taylor series time support with exponential space supports."""

from typing import Any

import torch

from qg.decomposition.base import SpaceTimeDecomposition
from qg.decomposition.coefficients import DecompositionCoefs
from qg.decomposition.supports.space.gaussian import (
    GaussianSupport,
    NormalizedGaussianSupport,
)
from qg.decomposition.supports.time.gaussian import GaussianTimeSupport


class GaussianExpBasis(
    SpaceTimeDecomposition[NormalizedGaussianSupport, GaussianTimeSupport]
):
    """Gaussian Time support and Gaussian space support."""

    type = "gaussian-exps"

    def _compute_space_params(
        self, params: dict[str, Any], xx: torch.Tensor, yy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        centers = params["centers"]
        xc = torch.tensor([c[0] for c in centers], **self._specs)
        yc = torch.tensor([c[1] for c in centers], **self._specs)

        x = xx[None, :, :] - xc[:, None, None]
        y = yy[None, :, :] - yc[:, None, None]
        return (x, y)

    def numel(self) -> int:  # noqa: D102
        return sum(s["numel"] * self._time[k]["numel"] for k, s in self._space.items())

    numel.__doc__ = SpaceTimeDecomposition[
        NormalizedGaussianSupport, GaussianTimeSupport
    ].numel.__doc__

    def generate_random_coefs(self) -> DecompositionCoefs:  # noqa: D102
        coefs = {}
        for k in self._space:
            coefs[k] = torch.randn(
                (
                    self._time[k]["numel"],
                    self._space[k]["numel"],
                ),
                **self._specs,
            )

        return DecompositionCoefs.from_dict(coefs)

    generate_random_coefs.__doc__ = SpaceTimeDecomposition[
        NormalizedGaussianSupport, GaussianTimeSupport
    ].generate_random_coefs.__doc__

    def _build_space(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.field)
        return fields

    def _build_space_dx(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dx)
        return fields

    def _build_space_dy(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dy)
        return fields

    def _build_space_dx2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dx2)
        return fields

    def _build_space_dy2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dy2)
        return fields

    def _build_space_dx3(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dx3)
        return fields

    def _build_space_dy3(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dy3)
        return fields

    def _build_space_dydx2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dydx2)
        return fields

    def _build_space_dxdy2(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        fields = {}

        for lvl, c in self._coefs.items():
            params = self._space[lvl]
            sx: float = params["sigma_x"]
            sy: float = params["sigma_y"]

            x, y = self._compute_space_params(params, xx, yy)

            E = GaussianSupport(x, y, sx, sy)
            e = NormalizedGaussianSupport(E)
            fields[lvl] = torch.einsum("tc,cxy->txy", c, e.dxdy2)
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
