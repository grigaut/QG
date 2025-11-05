"""Affine + Collinear QG model."""

import torch
from mqgeometry.fd import interp_TP, laplacian
from mqgeometry.masks import Masks
from mqgeometry.qgm import QGFV
import torch.nn.functional as F

from mqgeometry.solver.pv_inversion import (
    HomogeneousPVInversionCollinear,
    InhomogeneousPVInversionCollinear,
)


class QGMixed(QGFV):
    _alpha: torch.Tensor = None
    _psi2_init: torch.Tensor = None
    _dpsi2: torch.Tensor = None

    @property
    def alpha(self) -> torch.Tensor:
        """Collinearity coefficient."""
        if self._alpha is None:
            self._alpha = torch.zeros(self.psi_shape, **self.arr_kwargs)
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: torch.Tensor) -> None:
        self._alpha = alpha
        self._set_solver()

    @property
    def psi2_init(self) -> torch.Tensor:
        """Initial value of ѱ₂."""
        if self._psi2_init is None:
            self._psi2_init = torch.zeros(self.psi_shape, **self.arr_kwargs)
        return self._psi2_init

    @psi2_init.setter
    def psi2_init(self, psi2_init: torch.Tensor) -> None:
        self._psi2_init = psi2_init

    @property
    def dpsi2(self) -> torch.Tensor:
        """Time derivative of ѱ₂."""
        if self._dpsi2 is None:
            self._dpsi2 = torch.zeros(self.psi_shape, **self.arr_kwargs)
        return self._dpsi2

    @dpsi2.setter
    def dpsi2(self, dpsi2: torch.Tensor) -> None:
        self._dpsi2 = dpsi2

    def __init__(self, param):
        self.reset_time()
        self.Lx = param["Lx"]
        self.Ly = param["Ly"]
        self.nl = param["nl"]
        self.H = param["H"]
        self.g_prime = param["g_prime"]
        self.f0 = param["f0"]
        self.beta = param["beta"]
        self.bottom_drag_coef = param["bottom_drag_coef"]
        self.dt = param["dt"]

        # ensemble/device/dtype
        self.n_ens = param["n_ens"]
        self.device = param["device"]
        self.arr_kwargs = {"dtype": torch.float64, "device": self.device}

        # grid params
        self.ny = param["ny"]
        self.nx = param["nx"]
        n_ens, nl, nx, ny = self.n_ens, self.nl, self.nx, self.ny
        self.psi_shape = (self.n_ens, self.nl - 1, self.nx + 1, self.ny + 1)
        self.q_shape = (self.n_ens, self.nl - 1, self.nx, self.ny)
        self.dx = torch.tensor(self.Lx / nx, **self.arr_kwargs)
        self.dy = torch.tensor(self.Ly / ny, **self.arr_kwargs)
        self.flux_stencil = param["flux_stencil"]
        self.y = torch.linspace(
            0.5 * self.dy, self.Ly - 0.5 * self.dy, ny, **self.arr_kwargs
        ).unsqueeze(0)
        self.y0 = 0.5 * self.Ly

        mask = param["mask"] if "mask" in param.keys() else torch.ones(nx, ny)
        self.masks = Masks(mask.type(torch.float64).to(self.device))

        # auxillary matrices for elliptic equation
        self.compute_auxillary_matrices()

        # initialize state variables
        self.psi = torch.zeros((n_ens, nl - 1, nx + 1, ny + 1), **self.arr_kwargs)
        self.compute_q_from_psi()
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
        self._solver_homogeneous = HomogeneousPVInversionCollinear(
            self.A,
            self.alpha,
            self.f0,
            self.dx,
            self.dy,
            self.masks,
        )
        self._solver_inhomogeneous = InhomogeneousPVInversionCollinear(
            self.A,
            self.alpha,
            self.f0,
            self.dx,
            self.dy,
            self.masks,
        )
        if self.with_bc:
            sf_bc = self._sf_bc_interp(self.time.item())
            self._solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

    def compute_auxillary_matrices(self):
        # A operator
        super().compute_auxillary_matrices()
        self._A11 = self.A[0, 0]
        self._A12 = self.A[0, 1]

    def compute_q_from_psi(self) -> None:
        self.q = self.masks.q * (
            interp_TP(
                self.masks.psi
                * (
                    F.pad(
                        laplacian(self.psi, self.dx, self.dy),
                        (1, 1, 1, 1),
                        mode="constant",
                        value=0,
                    )
                    - self.f0**2 * self._A11 * self.psi
                    - self.f0**2
                    * self._A12
                    * (self.alpha * self.psi + self.psi2_init + self.time * self.dpsi2)
                )
            )
            + self.beta * (self.y - self.y0)
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

        return (-div_flux + fcg_drag) * self.masks.q

    def compute_time_derivatives_no_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        dq = self.advection_rhs_no_bc()

        # Solve Helmholtz equation
        dq_i = self.interp_TP(dq)
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i + self.f0**2 * self._A12 * self.dpsi2[..., 1:-1, 1:-1],
            ensure_mass_conservation=True,
        )

        return dpsi, dq

    def advection_rhs_with_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        u, v = self.grad_perp(self.psi, self.dx, self.dy)
        q_with_bc = self.pv_bc.expand(self.q)

        div_flux = self.div_flux(q_with_bc, u, v)

        # wind forcing + bottom drag
        if self.with_bc and self.bottom_drag_coef != 0:
            print("WARNING: non-zero bottom drag coef with BC.")
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

        return (-div_flux + fcg_drag) * self.masks.q

    def compute_time_derivatives_with_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        dq = self.advection_rhs_with_bc()

        dq_i = self.interp_TP(dq)

        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i + self.f0**2 * self._A12 * self.dpsi2[..., 1:-1, 1:-1],
            ensure_mass_conservation=False,
        )

        return dpsi, dq
