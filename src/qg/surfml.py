"""Affine + Collinear QG model."""

from typing import Any
import torch
from qg.decomposition.base import SpaceTimeDecomposition
from qg.decomposition.supports.space.base import SpaceSupportFunction
from qg.decomposition.supports.time.base import TimeSupportFunction
from qg.fd import interp_TP, laplacian
from qg.logging.core import getLogger
from qg.masks import Masks
from qg.qgm import QGFV
import torch.nn.functional as F
from qg.solver.pv_inversion import (
    HomogeneousPVInversion,
    InhomogeneousPVInversion,
)
from qg.space import compute_xy_q, compute_xy_u, compute_xy_v
from qg.specs import defaults
from qg.stretching_matrix import compute_A_tilde

logger = getLogger(__name__)


class SurfML(QGFV):
    _alpha: torch.Tensor = None

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        if self._alpha is None:
            self._alpha = torch.zeros((), **self.arr_kwargs)
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha
        self.compute_auxillary_matrices()
        self._set_solver()

    @property
    def basis(
        self,
    ) -> SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]:
        """Decomposition basis."""
        return self._basis

    @basis.setter
    def basis(
        self,
        basis: SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction],
    ) -> None:
        self._basis = basis

        X_psi, Y_psi = torch.meshgrid(self.xv, self.yv, indexing="ij")

        X_q, Y_q = compute_xy_q(X_psi, Y_psi)
        X_u, Y_u = compute_xy_u(X_psi, Y_psi)
        X_v, Y_v = compute_xy_v(X_psi, Y_psi)

        self.fpsi2 = basis.localize(X_psi, Y_psi)

        self._fpsi2 = basis.localize(X_q, Y_q)
        self._fpsi2_dx = basis.localize_dx(X_u, Y_u)
        self._fpsi2_dy = basis.localize_dy(X_v, Y_v)

        self.compute_q_from_psi()

    def __init__(self, param: dict[str, Any]) -> None:
        self.reset_time()
        self.xv: torch.Tensor = param["xv"]
        self.yv: torch.Tensor = param["yv"]
        self.Lx = self.xv[-1] - self.xv[0]
        self.Ly = self.yv[-1] - self.yv[0]
        self.H: torch.Tensor = param["H"]
        self.g_prime: torch.Tensor = param["g_prime"]
        self.f0: float = param["f0"]
        self.beta: float = param["beta"]
        self.bottom_drag_coef: float = param["bottom_drag_coef"]
        self.dt: float = param["dt"]

        # ensemble/device/dtype
        self.device = param["device"]
        self.arr_kwargs = {"dtype": torch.float64, "device": self.device}

        # grid params
        self.n_ens: int = param.get("n_ens", 1)
        self.nl: int = self.H.shape[0]
        self.nx: int = self.xv.shape[0] - 1
        self.ny: int = self.yv.shape[0] - 1
        n_ens, nl, nx, ny = self.n_ens, self.nl, self.nx, self.ny
        self.psi_shape = (self.n_ens, self.nl - 1, self.nx + 1, self.ny + 1)
        self.q_shape = (self.n_ens, self.nl - 1, self.nx, self.ny)
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny
        self.flux_stencil: int = param.get("flux_stencil", 5)
        self.y = (self.yv[:-1] + self.yv[1:])[None, :] / 2
        self.y0 = 0.5 * (self.yv[0] + self.yv[-1])

        mask = param["mask"] if "mask" in param.keys() else torch.ones(nx, ny)
        self.masks = Masks(mask.type(torch.float64).to(self.device))

        # auxillary matrices for elliptic equation
        self.compute_auxillary_matrices()

        # initialize state variables
        self.psi = torch.zeros((n_ens, nl - 1, nx + 1, ny + 1), **self.arr_kwargs)

        self.zeros_inside = (
            torch.zeros((n_ens, nl - 2, nx, ny), **self.arr_kwargs) if nl > 2 else None
        )

        # wind forcing
        self.wind_forcing = torch.zeros((1, 1, nx, ny), **self.arr_kwargs)

        # precompile torch functions if torch >= 2.0
        self._set_flux()
        self._set_solver()

    def _set_solver(self) -> None:
        """Set Helmholtz equation solver."""
        # PV equation solver
        self._solver_homogeneous = HomogeneousPVInversion(
            self.A[:1, :1],
            self.f0,
            self.dx,
            self.dy,
            self.masks,
        )
        self._solver_inhomogeneous = InhomogeneousPVInversion(
            self.A[:1, :1],
            self.f0,
            self.dx,
            self.dy,
            self.masks,
        )
        if self.with_bc:
            sf_bc = self._sf_bc_interp(self.time.item())
            self._solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

    def _with_boundaries(self) -> None:
        if self.with_bc:
            return
        self.with_bc = True
        self._set_flux()

    def compute_auxillary_matrices(self):
        # A operator
        H = self.H[:, 0, 0]
        g_prime = self.g_prime[:, 0, 0]

        self.A = compute_A_tilde(H, g_prime, self.alpha, **defaults.get())
        self._A11 = self.A[:1, :1]
        self._A12 = self.A[:1, 1:2]

    def compute_forcing(
        self,
        time: torch.Tensor,
        psi1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forcing.

        Args:
            time (torch.Tensor): Time to evaluate at.
            psi1 (torch.Tensor): Top layer stream function.

        Returns:
            torch.Tensor: -f₀²J(ѱ₁, ѱ₂)/H₂g₂
        """
        u, v = self.grad_perp(psi1, self.dx, self.dy)

        dt_psi2 = self._fpsi2.dt(time)
        dx_psi2 = self._fpsi2_dx(time)
        dy_psi2 = self._fpsi2_dy(time)

        u_dxpsi2 = u * dx_psi2
        v_dypsi2 = v * dy_psi2

        adv = (u_dxpsi2[..., 1:, :] + u_dxpsi2[..., :-1, :]) / 2 + (
            v_dypsi2[..., 1:] + v_dypsi2[..., :-1]
        ) / 2
        return (self.f0**2) * self._A12 * (dt_psi2 + adv)

    def compute_q_from_psi(self) -> None:
        try:
            psi2 = self.fpsi2(self.time)
        except AttributeError:
            msg = "No basis specifified, using 0s for psi2 when computing q."
            logger.warning(msg)
            psi2 = torch.zeros(
                (self.n_ens, self.nx + 1, self.ny + 1), **self.arr_kwargs
            )

        if self.with_bc:
            boundary = self._sf_bc_interp(self.time.item())
            psi_with_bc = boundary.get_band(1).expand(self.psi[:, :1])
            self.p = psi_with_bc
            lap_psi = laplacian(psi_with_bc, self.dx, self.dy) * self.masks.psi
        else:
            lap_psi = (
                F.pad(
                    laplacian(self.psi, self.dx, self.dy),
                    (1, 1, 1, 1),
                    mode="constant",
                    value=0,
                )
                * self.masks.psi
            )

        stretching = (
            -(self.f0**2) * (self._A11 * self.psi + self._A12 * psi2) * self.masks.psi
        )

        self.q = self.masks.q * (
            interp_TP(lap_psi + stretching) + self.beta * (self.y - self.y0)
        )

    def advection_rhs_no_bc(self) -> torch.Tensor:
        u, v = self.grad_perp(self.psi, self.dx, self.dy)
        div_flux = self.div_flux(self.q, u[..., 1:-1, :], v[..., 1:-1])

        # wind forcing + bottom drag
        omega = self.interp_TP(
            self.laplacian_h(self.psi, self.dx, self.dy) * self.masks.psi
        )
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]
        if self.nl - 1 == 1:
            fcg_drag = self.wind_forcing + bottom_drag
        elif self.nl - 1 == 2:
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

        forcing = self.compute_forcing(self._substep_time, self.psi[:, :1])
        return (-div_flux + fcg_drag + forcing) * self.masks.q

    def compute_time_derivatives_no_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        dq = self.advection_rhs_no_bc()

        # Solve Helmholtz equation
        dq_i = self.interp_TP(dq)
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=True,
        )

        return dpsi, dq

    def advection_rhs_with_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = self.grad_perp(self.psi, self.dx, self.dy)
        q_with_bc = self.pv_bc.expand(self.q)

        div_flux = self.div_flux(q_with_bc, u, v)

        # wind forcing + bottom drag
        sf_boundary = self._sf_bc_interp(self.time.item())
        sf_wide = sf_boundary.expand(self.psi[..., 1:-1, 1:-1])
        omega = interp_TP(laplacian(sf_wide, self.dx, self.dy))
        bottom_drag = -self.bottom_drag_coef * omega[..., [-1], :, :]

        if self.nl - 1 == 1:
            fcg_drag = self.wind_forcing + bottom_drag
        elif self.nl - 1 == 2:
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

        forcing = self.compute_forcing(self._substep_time, self.psi[:, :1])
        return (-div_flux + fcg_drag + forcing) * self.masks.q

    def compute_time_derivatives_with_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        dq = self.advection_rhs_with_bc()

        dq_i = self.interp_TP(dq)

        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i,
            ensure_mass_conservation=False,
        )

        return dpsi, dq
