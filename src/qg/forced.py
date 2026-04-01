"""Affine + Collinear QG model."""

import torch
from qg.fd import laplacian
from qg.logging.core import getLogger
from qg.qgm import QGFV

logger = getLogger(__name__)


class Forced(QGFV):
    _forcing: torch.Tensor = None

    @property
    def forcing(self) -> torch.Tensor:
        """Forcing term.

        └── (n_ens, nl, nx, ny)-shaped
        """
        if self._forcing is None:
            return torch.zeros_like(self.q)
        return self._forcing

    @forcing.setter
    def forcing(self, forcing: torch.Tensor) -> None:
        self._forcing = forcing

    @property
    def wind_scaling(self) -> torch.Tensor:
        """Wind forcing scaling."""
        try:
            return self._wind_scaling
        except AttributeError:
            return self.H[0, 0, 0].item()

    @wind_scaling.setter
    def wind_scaling(self, wind_scaling: torch.Tensor) -> None:
        self._wind_scaling = wind_scaling

    def set_wind_forcing(self, curl_tau):
        self.wind_forcing = curl_tau / self.wind_scaling

    def advection_rhs_no_bc(self) -> torch.Tensor:
        u, v = self.grad_perp(self.psi, self.dx, self.dy)
        div_flux = self.div_flux(self.q, u[..., 1:-1, :], v[..., 1:-1])

        # wind forcing + bottom drag
        omega = self.interp_TP(
            self.laplacian_h(self.psi, self.dx, self.dy) * self.masks.psi
        )
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.nl == 1:
            fcg_drag = self.wind_forcing + bottom_drag
        elif self.nl == 2:
            fcg_drag = torch.cat([self.wind_forcing, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [
                    self.wind_forcing,
                    self.zeros_inside,
                    bottom_drag,
                ],
                dim=-3,
            )

        return (-div_flux + fcg_drag + self.forcing) * self.masks.q

    def advection_rhs_with_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = self.grad_perp(self.psi, self.dx, self.dy)
        q_with_bc = self.pv_bc.expand(self.q)

        div_flux = self.div_flux(q_with_bc, u, v)

        # wind forcing + bottom drag
        sf_boundary = self._sf_bc_interp(self.time.item())
        sf_wide = sf_boundary.expand(self.psi[..., 1:-1, 1:-1])
        omega = self.interp_TP(laplacian(sf_wide, self.dx, self.dy))
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]

        if self.nl == 1:
            fcg_drag = self.wind_forcing + bottom_drag
        elif self.nl == 2:
            fcg_drag = torch.cat([self.wind_forcing, bottom_drag], dim=-3)
        else:
            fcg_drag = torch.cat(
                [
                    self.wind_forcing,
                    self.zeros_inside,
                    bottom_drag,
                ],
                dim=-3,
            )

        return (-div_flux + fcg_drag + self.forcing) * self.masks.q
