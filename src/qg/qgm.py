from typing import Any
import torch

from qg import logging
from qg.fd import grad_perp, interp_TP, laplacian, laplacian_h
from qg.flux import (
    div_flux_3pts,
    div_flux_3pts_mask,
    div_flux_5pts,
    div_flux_5pts_mask,
    div_flux_5pts_only,
)
from qg.interpolation import _Interpolation
from qg.masks import Masks
from qg.solver.boundary_conditions.base import Boundaries
from qg.solver.pv_inversion import (
    HomogeneousPVInversion,
    InhomogeneousPVInversion,
)
from qg.stretching_matrix import compute_A

logger = logging.getLogger(__name__)


class QGFV:
    """Finite volume multi-layer QG solver."""

    with_bc = False

    @property
    def time(self) -> torch.Tensor:
        """Simulated time."""
        return torch.tensor(self.n_steps * self.dt, **self.arr_kwargs)

    def __init__(self, param: dict[str, Any]):
        # physical params
        self.reset_time()
        xv: torch.Tensor = param["xv"]
        yv: torch.Tensor = param["yv"]
        self.Lx = xv[-1] - xv[0]
        self.Ly = yv[-1] - yv[0]
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
        self.nx: int = xv.shape[0] - 1
        self.ny: int = yv.shape[0] - 1
        n_ens, nl, nx, ny = self.n_ens, self.nl, self.nx, self.ny
        self.psi_shape = (self.n_ens, self.nl, self.nx + 1, self.ny + 1)
        self.q_shape = (self.n_ens, self.nl, self.nx, self.ny)
        self.dx = self.Lx / nx
        self.dy = self.Ly / ny
        self.flux_stencil: int = param.get("flux_stencil", 5)
        self.y = (yv[:-1] + yv[1:])[None, :] / 2
        self.y0 = 0.5 * (yv[0] + yv[-1])

        mask = param["mask"] if "mask" in param.keys() else torch.ones(nx, ny)
        self.masks = Masks(mask.type(torch.float64).to(self.device))

        # auxillary matrices for elliptic equation
        self.compute_auxillary_matrices()

        # initialize state variables
        self.psi = torch.zeros((n_ens, nl, nx + 1, ny + 1), **self.arr_kwargs)
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
        self._solver_homogeneous = HomogeneousPVInversion(
            self.A,
            self.f0,
            self.dx,
            self.dy,
            self.masks,
        )

    def _set_flux(self) -> None:
        """Set the fluxes utils."""
        if self.with_bc:
            return self._set_flux_inhomogeneous()
        return self._set_flux_homogeneous()

    def _set_flux_homogeneous(self) -> None:
        """Set the flux.

        Raises:
            ValueError: If invalid stencil.
        """
        # flux computations
        if self.flux_stencil == 5:
            if len(self.masks.psi_irrbound_xids) > 0:

                def div_flux(
                    q: torch.Tensor,
                    u: torch.Tensor,
                    v: torch.Tensor,
                ) -> torch.Tensor:
                    return div_flux_5pts_mask(
                        q,
                        u,
                        v,
                        self.dx,
                        self.dy,
                        self.masks.u_distbound1[..., 1:-1, :],
                        self.masks.u_distbound2[..., 1:-1, :],
                        self.masks.u_distbound3plus[..., 1:-1, :],
                        self.masks.v_distbound1[..., 1:-1],
                        self.masks.v_distbound2[..., 1:-1],
                        self.masks.v_distbound3plus[..., 1:-1],
                    )
            else:

                def div_flux(
                    q: torch.Tensor,
                    u: torch.Tensor,
                    v: torch.Tensor,
                ) -> torch.Tensor:
                    return div_flux_5pts(
                        q,
                        u,
                        v,
                        self.dx,
                        self.dy,
                    )
        elif self.flux_stencil == 3:
            if len(self.masks.psi_irrbound_xids) > 0:

                def div_flux(
                    q: torch.Tensor,
                    u: torch.Tensor,
                    v: torch.Tensor,
                ) -> torch.Tensor:
                    return div_flux_3pts_mask(
                        q,
                        u,
                        v,
                        self.dx,
                        self.dy,
                        self.masks.u_distbound1[..., 1:-1, :],
                        self.masks.u_distbound2plus[..., 1:-1, :],
                        self.masks.v_distbound1[..., 1:-1],
                        self.masks.v_distbound2plus[..., 1:-1],
                    )
            else:

                def div_flux(
                    q: torch.Tensor,
                    u: torch.Tensor,
                    v: torch.Tensor,
                ) -> torch.Tensor:
                    return div_flux_3pts(
                        q,
                        u,
                        v,
                        self.dx,
                        self.dy,
                    )

        comp = torch.__version__[0] == "2"
        self.grad_perp = (
            torch.compile(grad_perp)
            if comp
            else torch.jit.trace(grad_perp, (self.psi, self.dx, self.dy))
        )
        self.interp_TP = torch.compile(interp_TP) if comp else interp_TP
        self.laplacian_h = (
            torch.compile(laplacian_h)
            if comp
            else torch.jit.trace(laplacian_h, (self.psi, self.dx, self.dy))
        )
        self.div_flux = (
            torch.compile(div_flux)
            if comp
            else torch.jit.trace(
                div_flux,
                (
                    self.q,
                    self.grad_perp(self.psi, self.dx, self.dy)[0][..., 1:-1, :],
                    self.grad_perp(self.psi, self.dx, self.dy)[1][..., 1:-1],
                ),
            )
        )
        if not comp:
            logger.info(
                "Need torch >= 2.0 to use torch.compile, current version "
                f"{torch.__version__}, using torch.jit.trace."
            )

    def _set_flux_inhomogeneous(self) -> None:
        """Set the flux.

        Raises:
            ValueError: If invalid stencil.
        """
        if self.flux_stencil == 5:
            if len(self.masks.psi_irrbound_xids) > 0:
                msg = (
                    "Inhomogeneous pv reconstruction not "
                    "implemented for non-regular geometry."
                )
                raise NotImplementedError(msg)

            def div_flux(
                q: torch.Tensor, u: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                return div_flux_5pts_only(
                    q,
                    u,
                    v,
                    self.dx,
                    self.dy,
                )
        elif self.flux_stencil == 3:
            msg = "Inhomogeneous pv reconstruction not implemented for 3 pts stencil."
            raise NotImplementedError(msg)
        else:
            msg = f"Invalid stencil value: {self.flux_stencil}"
            raise ValueError(msg)
        comp = torch.__version__[0] == "2"
        self.grad_perp = (
            torch.compile(grad_perp)
            if comp
            else torch.jit.trace(grad_perp, (self.psi, self.dx, self.dy))
        )
        self.interp_TP = torch.compile(interp_TP) if comp else interp_TP
        self.lap = (
            torch.compile(laplacian)
            if comp
            else torch.jit.trace(laplacian, (self.psi, self.dx, self.dy))
        )
        self.div_flux = (
            torch.compile(div_flux)
            if comp
            else torch.jit.trace(
                div_flux,
                (
                    self.pv_bc.expand(self.q),
                    *self.grad_perp(self.psi, self.dx, self.dy),
                ),
            )
        )
        if not comp:
            logger.info(
                "Need torch >= 2.0 to use torch.compile, current version "
                f"{torch.__version__}, using torch.jit.trace."
            )

    def _with_boundaries(self) -> None:
        """Switch to an inhomogeneous solver."""
        if self.with_bc:
            return
        self.with_bc = True
        self._set_flux()
        self._solver_inhomogeneous = InhomogeneousPVInversion(
            self.A,
            self.f0,
            self.dx,
            self.dy,
            self.masks,
        )

    def _set_boundaries(self, time: float) -> None:
        """Set the boundaries to match given time.

        Args:
            time (float): Time.
        """
        sf_bc = self._sf_bc_interp(time)

        self._solver_inhomogeneous.set_boundaries(sf_bc.get_band(0))

        pv_bc = self._pv_bc_interp(time)
        if pv_bc.width != 3:  # noqa: PLR2004
            msg = "For wide boundary, pv_bc must be 3 points wide."
            raise ValueError(msg)
        self.pv_bc = pv_bc

    def set_boundary_maps(
        self,
        sf_bc_interp: _Interpolation[Boundaries],
        pv_bc_interp: _Interpolation[Boundaries],
    ) -> None:
        """Set the boundary maps.

        Args:
            sf_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for stream function at locations
                (imin,imax+1,jmin,jmax+1).
            pv_bc_interp (LinearInterpolation[Boundaries]): Boundary map
                for potential vorticity at locations
                (imin,imax,jmin,jmax).
        """
        self._with_boundaries()
        self._sf_bc_interp = sf_bc_interp
        self._pv_bc_interp = pv_bc_interp
        self._set_boundaries(self.time.item())

    def compute_auxillary_matrices(self):
        # A operator
        H = self.H.squeeze().reshape((-1,))
        g_prime = self.g_prime.squeeze().reshape((-1,))
        self.A = compute_A(H, g_prime, **self.arr_kwargs)

    def set_psiq(self, psi: torch.Tensor, q: torch.Tensor) -> None:
        """Set the values of ѱ and q."""
        if psi.shape != self.psi_shape:
            msg = f"ѱ should be {self.psi_shape}-shaped."
            raise ValueError(msg)
        if q.shape != self.q_shape:
            msg = f"q should be {self.q_shape}-shaped."
            raise ValueError(msg)
        self.psi = psi
        self.q = q

    def compute_q_from_psi(self):
        self.q = self.masks.q * (
            interp_TP(
                self.masks.psi
                * (
                    laplacian_h(self.psi, self.dx, self.dy)
                    - self.f0**2 * torch.einsum("lm,...mxy->...lxy", self.A, self.psi)
                )
            )
            + self.beta * (self.y - self.y0)
        )

    def set_wind_forcing(self, curl_tau):
        self.wind_forcing = curl_tau / self.H[0]

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

        return (-div_flux + fcg_drag) * self.masks.q

    def compute_time_derivatives_no_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        dq = self.advection_rhs_no_bc()

        # Solve Helmholtz equation
        dq_i = self.interp_TP(dq)
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i, ensure_mass_conservation=True
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

        return (-div_flux + fcg_drag) * self.masks.q

    def compute_time_derivatives_with_bc(self) -> tuple[torch.Tensor, torch.Tensor]:
        dq = self.advection_rhs_with_bc()

        dq_i = self.interp_TP(dq)

        # Solve Helmholtz equation
        dpsi = self._solver_homogeneous.compute_stream_function(
            dq_i, ensure_mass_conservation=False
        )

        return dpsi, dq

    def step_no_bc(self) -> None:
        """Time itegration with SSP-RK3 scheme."""

        dpsi_0, dq_0 = self.compute_time_derivatives_no_bc()
        self.q = self.q + self.dt * dq_0
        self.psi = self.psi + self.dt * dpsi_0

        dpsi_1, dq_1 = self.compute_time_derivatives_no_bc()
        self.q = self.q + (self.dt / 4) * (dq_1 - 3 * dq_0)
        self.psi = self.psi + (self.dt / 4) * (dpsi_1 - 3 * dpsi_0)

        dpsi_2, dq_2 = self.compute_time_derivatives_no_bc()
        self.q = self.q + (self.dt / 12) * (8 * dq_2 - dq_1 - dq_0)
        self.psi = self.psi + (self.dt / 12) * (8 * dpsi_2 - dpsi_1 - dpsi_0)
        self.n_steps += 1

    def step_with_bc(self) -> None:
        psi_bc = self._solver_inhomogeneous.psiq_bc[0]

        dpsi_0, dq_0 = self.compute_time_derivatives_with_bc()
        self.q = self.q + self.dt * dq_0

        self.psi = self.psi - psi_bc
        coef = 1
        self._set_boundaries(self.time.item() + coef * self.dt)
        psi_bc = self._solver_inhomogeneous.psiq_bc[0]
        self.psi = self.psi + self.dt * dpsi_0 + psi_bc

        dpsi_1, dq_1 = self.compute_time_derivatives_with_bc()
        self.q = self.q + (self.dt / 4) * (dq_1 - 3 * dq_0)

        self.psi = self.psi - psi_bc
        coef = 1 / 2
        self._set_boundaries(self.time.item() + coef * self.dt)
        psi_bc = self._solver_inhomogeneous.psiq_bc[0]
        self.psi = self.psi + (self.dt / 4) * (dpsi_1 - 3 * dpsi_0) + psi_bc

        dpsi_2, dq_2 = self.compute_time_derivatives_with_bc()
        self.q = self.q + (self.dt / 12) * (8 * dq_2 - dq_1 - dq_0)

        self.psi = self.psi - psi_bc
        coef = 1
        self._set_boundaries(self.time.item() + coef * self.dt)
        psi_bc = self._solver_inhomogeneous.psiq_bc[0]
        self.psi = self.psi + (self.dt / 12) * (8 * dpsi_2 - dpsi_1 - dpsi_0) + psi_bc

        self.n_steps += 1

    def step(self) -> None:
        if self.with_bc:
            return self.step_with_bc()
        return self.step_no_bc()

    def reset_time(self) -> None:
        self.n_steps = 0
