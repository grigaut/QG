"""Solvers for Helmholtz equations with Dirichlet boundary conditions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from mqgeometry.fd import interp_TP
from mqgeometry.solver.boundary_conditions.base import Boundaries
from mqgeometry.solver.boundary_conditions.interpolation import (
    BilinearExtendedBoundary,
)
from mqgeometry.helmholtz import (
    compute_capacitance_matrices,
    compute_laplace_dst,
    solve_helmholtz_dst,
    solve_helmholtz_dst_cmm,
)
from mqgeometry.specs import defaults

from mqgeometry.stretching_matrix import compute_layers_to_mode_decomposition

if TYPE_CHECKING:
    from mqgeometry.masks import Masks


class BasePVInversion(ABC):
    """Base class for Helmholtz solvers with Dirichlet boundary conditions.

    Convention:
        - potential vorticity: (..., nl, nx, ny)-shaped
        - stream function: (..., nl, nx+1, ny+1)-shaped
    """

    _nl = None
    _nx = None
    _ny = None

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        f0: float,
        dx: float,
        dy: float,
        masks: Masks | None = None,
    ) -> None:
        """Base class for potential vorticity inversion.

        Args:
            A (torch.Tensor): Stretching matrix.
            f0 (float): Coriolis parameter.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
            masks (Masks | None, optional): Masks. Defaults to None.
        """
        self._A = A
        self._nl = A.shape[0]
        self._f0 = f0
        self._dx = dx
        self._dy = dy

        Cm2l, lambd, Cl2m = compute_layers_to_mode_decomposition(A)  # noqa: N806
        self._Cm2l = Cm2l
        self._lambd = lambd.reshape((1, lambd.shape[0], 1, 1))
        self._Cl2m = Cl2m
        self._masks = masks

    @abstractmethod
    def compute_stream_function(self, pv: torch.Tensor) -> torch.Tensor:
        """Compute the stream function from the potential vorticity.

        Args:
            pv (torch.Tensor): Potential vorticity, shape (..., nl, nx, ny).

        Returns:
            torch.Tensor: Stream function, shape (..., nl, nx+1, ny+1).
        """


class HomogeneousPVInversion(BasePVInversion):
    """Homogeneous potential vorticity inversion."""

    def _compute_homsol(
        self,
        cst: torch.Tensor,
        helmholtz_dstI: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        """Compute homogeneous solution to use for mass conservation.

        Actual value is set in __init__ depending on the geometry.
        """

    def _solve_modespace(
        self,
        rhs: torch.Tensor,
        helmholtz_dstI: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        """Solve the equation in modal space.

        Args:
            rhs (torch.Tensor): Right hand side of the modal
                helmholtz equation, shape (..., nl, nx, ny).
            helmholtz_dstI (torch.Tensor): Helmholtz operator in
                spectral space, shape (..., nl, nx+1, ny+1).


        Returns:
            torch.Tensor: Stream function in modal space,
                shape (..., nl, nx+1, ny+1).
        """

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        f0: float,
        dx: float,
        dy: float,
        masks: Masks | None = None,
    ) -> None:
        """Potential vorticity inversion with homogeneous boundary.

        Args:
            A (torch.Tensor): Stretching matrix.
            f0 (float): Coriolis parameter.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
            masks (Masks | None, optional): Masks. Defaults to None.
        """
        super().__init__(A, f0, dx, dy, masks)
        if masks is not None and len(masks.psi_irrbound_xids) > 0:
            self._compute_homsol = self._homsol_irregular_geometry
            self._solve_modespace = self._solve_irregular_geometry
        else:
            self._compute_homsol = self._homsol_regular_geometry
            self._solve_modespace = self._solve_regular_geometry

    def _set_shape(
        self,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set the shape.

        Args:
            nl (int): Number of layer.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype | None, optional): Data type.
                Defaults to None.
            device (torch.device | None, optional): Device.
                Defaults to None.
        """
        dtype = defaults.get_dtype(dtype=dtype)
        device = defaults.get_device(device=device)
        if (nl, nx, ny) != (self._nl, self._nx, self._ny):
            self._helmholtz_dstI = self._compute_helmholtz_dstI(
                nx, ny, **defaults.get(device=device, dtype=dtype)
            )
            self._homsol, self._homsol_mean = self._compute_homogeneous_solution(
                nl, nx, ny, **defaults.get(device=device, dtype=dtype)
            )
        elif (
            dtype != self._helmholtz_dstI.dtype or device != self._helmholtz_dstI.device
        ):
            self._helmholtz_dstI = self._helmholtz_dstI.to(dtype=dtype, device=device)
            self._homsol, self._homsol_mean = (
                self._homsol.to(dtype=dtype, device=device),
                self._homsol_mean.to(dtype=dtype, device=device),
            )
        self._nl = nl
        self._nx = nx
        self._ny = ny

    def _compute_helmholtz_dstI(  # noqa: N802
        self,
        nx: int,
        ny: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute the Helmholtz operator in spectral space.

        Args:
            nx (int): Number of grid points in the x-direction.
            ny (int): Number of grid points in the y-direction.
            dtype (torch.dtype | None, optional): Data type of the tensor.
                Defaults to None.
            device (torch.device | None, optional): Device to create
            the tensor on. Defaults to None.

        Returns:
            torch.Tensor: Helmholtz operator in spectral space.
        """
        laplacian = compute_laplace_dst(
            nx, ny, self._dx, self._dy, defaults.get(device=device, dtype=dtype)
        )
        laplacian = laplacian.unsqueeze(0).unsqueeze(0)
        return laplacian - self._f0**2 * self._lambd

    def _compute_homogeneous_solution(
        self,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Compute homogeneous solution to use for mass conservation.

        Args:
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype | None, optional): Data type.
                Defaults to None.
            device (torch.device | None, optional): Device.
                Defaults to None.
        """
        cst = torch.ones(
            (1, nl, nx + 1, ny + 1),
            device=device,
            dtype=dtype,
        )
        sol = self._compute_homsol(cst, self._helmholtz_dstI)
        self._homsol = cst + sol * self._f0**2 * self._lambd
        self._homsol_mean = self._homsol.mean((-1, -2), keepdim=True)
        return self._homsol, self._homsol_mean

    def _homsol_irregular_geometry(
        self,
        cst: torch.Tensor,
        helmholtz_dstI: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        """Compute homogeneous solution on irregular geometry.

        Args:
            cst (torch.Tensor): Constant modal potential vorticity.
                └── (..., nl, nx+1, ny+1)-shaped
            helmholtz_dstI (torch.Tensor): Helmholtz operator in
                spectral space.
                └── (..., nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Solution ψ to ∇²ψ - f²_0 Λ ψ = 1
        """
        self._cap_matrices = compute_capacitance_matrices(
            helmholtz_dstI,
            self._masks.psi_irrbound_xids,
            self._masks.psi_irrbound_yids,
        )
        return solve_helmholtz_dst_cmm(
            (cst * self._masks.psi)[..., 1:-1, 1:-1],
            helmholtz_dstI,
            self._cap_matrices,
            self._masks.psi_irrbound_xids,
            self._masks.psi_irrbound_yids,
            self._masks.psi,
        )

    def _homsol_regular_geometry(
        self,
        cst: torch.Tensor,
        helmholtz_dstI: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        """Compute homogeneous solution on regular geometry.

        Args:
            cst (torch.Tensor): Constant modal potential vorticity.
                └── (..., nl, nx+1, ny+1)-shaped
            helmholtz_dstI (torch.Tensor): Helmholtz operator in
                spectral space.
                └── (..., nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Solution ψ to ∇²ψ - f²_0 Λ ψ = 1
        """
        return solve_helmholtz_dst(
            cst[..., 1:-1, 1:-1],
            helmholtz_dstI,
        )

    def _solve_irregular_geometry(
        self,
        rhs: torch.Tensor,
        helmholtz_dstI: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        """Solve the equation in modal space.

        Args:
            rhs (torch.Tensor): Right hand side of the modal
                helmholtz equation, shape (..., nl, nx, ny).
            helmholtz_dstI (torch.Tensor): Helmholtz operator in
                spectral space, shape (..., nl, nx+1, ny+1).


        Returns:
            torch.Tensor: Stream function in modal space,
                shape (..., nl, nx+1, ny+1).
        """
        return solve_helmholtz_dst_cmm(
            rhs * self._masks.psi[..., 1:-1, 1:-1],
            helmholtz_dstI,
            self._cap_matrices,
            self._masks.psi_irrbound_xids,
            self._masks.psi_irrbound_yids,
            self._masks.psi,
        )

    def _solve_regular_geometry(
        self,
        rhs: torch.Tensor,
        helmholtz_dstI: torch.Tensor,  # noqa: N803
    ) -> torch.Tensor:
        """Solve the equation in modal space.

        Args:
            rhs (torch.Tensor): Right hand side of the modal
                helmholtz equation, shape (..., nl, nx, ny).
            helmholtz_dstI (torch.Tensor): Helmholtz operator in
                spectral space, shape (..., nl, nx+1, ny+1).


        Returns:
            torch.Tensor: Stream function in modal space,
                shape (..., nl, nx+1, ny+1).
        """
        return solve_helmholtz_dst(rhs, helmholtz_dstI)

    def _correct_sf_for_mass_conservation(
        self,
        sf_modes: torch.Tensor,
    ) -> torch.Tensor:
        """Correct the stream function to ensure mass conservation.

        Args:
            sf_modes (torch.Tensor): Stream function modes
                └── (..., nl, nx+1, ny+1)-shaped

        Returns:
            torch.Tensor: Corrected stream function modes
                └── (..., nl, nx+1, ny+1)-shaped
        """
        sf_modes_mean = sf_modes.mean((-1, -2), keepdim=True)
        alpha = -sf_modes_mean / self._homsol_mean
        return sf_modes + alpha * self._homsol

    def compute_stream_function(
        self,
        pv: torch.Tensor,
        *,
        ensure_mass_conservation: bool = True,
    ) -> torch.Tensor:
        """Compute the stream function from the potential vorticity.

        Homogeneous boundary conditions are assumed.

        Args:
            pv (torch.Tensor): Potential vorticity.
                └── (..., nl, nx-1, ny-1)-shaped
            ensure_mass_conservation (bool, optional): Whether to ensure mass
                conservation.

        Returns:
            torch.Tensor: Stream function
                └── (..., nl, nx+1, ny+1)-shaped
        """
        nl, nx, ny = pv.shape[-3:]
        self._set_shape(nl, nx + 1, ny + 1)
        rhs = torch.einsum("lm,...mxy->...lxy", self._Cl2m, pv)
        sf_modes = self._solve_modespace(rhs, self._helmholtz_dstI)
        if ensure_mass_conservation:
            sf_modes = self._correct_sf_for_mass_conservation(sf_modes)
        return torch.einsum("ml,...lxy->...mxy", self._Cm2l, sf_modes)


class InhomogeneousPVInversion(BasePVInversion):
    """Inhomogeneous potential vorticity inversion.

    Boundary interpolation is conducted using BilinearExtendedBoundary.
    """

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        f0: float,
        dx: float,
        dy: float,
        masks: Masks | None = None,
    ) -> None:
        """Potential vorticity inversion with inhomogeneous boundary.

        Args:
            A (torch.Tensor): Stretching matrix.
            f0 (float): Coriolis parameter.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
            masks (Masks | None, optional): Masks. Defaults to None.
        """
        super().__init__(A, f0, dx, dy, masks)
        self._homogeneous_solver = HomogeneousPVInversion(A, f0, dx, dy, masks)
        if self._masks is not None and len(self._masks.psi_irrbound_xids) > 0:
            msg = "Irregular geometries not supported for inhomogeneous boundaries."
            raise NotImplementedError(msg)

    def set_boundaries(self, boundaries: Boundaries) -> None:
        """Set the boundary field.

        Args:
            boundaries (Boundaries): Boundary conditions.
        """
        if boundaries.width != 1:
            msg = (
                "Only boundaries with width=1 are supported. "
                f"Got width={boundaries.width}."
            )
            raise ValueError(msg)
        self._boundary = BilinearExtendedBoundary(boundaries)
        sf_b = self._boundary.compute()
        pv_b_int = self._compute_interior_pv_boundary()
        sf_h = self._homogeneous_solver.compute_stream_function(
            -pv_b_int, ensure_mass_conservation=False
        )
        sf_bc = sf_b + sf_h
        pv_bc = torch.zeros_like(interp_TP(sf_b))
        self._psi_b = sf_b
        self._psi_h = sf_h
        self.psiq_bc = (sf_bc, pv_bc)

    def set_boundaries_from_tensors(
        self,
        top: torch.Tensor,
        bottom: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> None:
        """Set the boundary values.

        Args:
            top (torch.Tensor): Boundary condition at the top (y=y_max)
                └── (..., nl, ny)-shaped
            bottom (torch.Tensor): Boundary condition at the bottom (y=y_min)
                └── (..., nl, ny)-shaped
            left (torch.Tensor): Boundary condition on the left (x=x_min)
                └── (..., nl, nx)-shaped
            right (torch.Tensor): Boundary condition on the right (x=x_max)
                └── (..., nl, nx)-shaped
        """
        self.set_boundaries(Boundaries(top=top, bottom=bottom, left=left, right=right))

    def compute_stream_function(self, pv: torch.Tensor) -> torch.Tensor:
        """Compute the stream function from the potential vorticity.

        Boundary conditions are set to match self._boundary.

        Args:
            pv (torch.Tensor): Potential vorticity
                └── (..., nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function
                └── (..., nl, nx+1, ny+1)-shaped
        """
        sf_i, sf_h, sf_b = self.compute_stream_function_components(pv)
        return sf_i + sf_h + sf_b

    def compute_stream_function_components(
        self, pv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the stream function components from the potential vorticity.

        Boundary conditions are set to match self._boundary.

        Args:
            pv (torch.Tensor): Potential vorticity
                └── (..., nl, nx, ny)-shaped

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Homogeneous stream function,
                boundary stream function.
                ├── ѱ_i (..., nl, nx+1, ny+1)-shaped
                ├── ѱ_h (..., nl, nx+1, ny+1)-shaped
                └── ѱ_b (..., nl, nx+1, ny+1)-shaped
        """
        sf_i = self._homogeneous_solver.compute_stream_function(
            pv, ensure_mass_conservation=False
        )
        return sf_i, self._psi_h, self._psi_b

    def _compute_interior_pv_boundary(self) -> torch.Tensor:
        laplacian_boundary = self._boundary.compute_laplacian(self._dx, self._dy)
        sf_boundary_interior = self._boundary.compute()[..., 1:-1, 1:-1]
        return laplacian_boundary - self._f0**2 * torch.einsum(
            "lm,...mxy->...lxy", self._A, sf_boundary_interior
        )


class HomogeneousPVInversionCollinear(HomogeneousPVInversion):
    """Homogeneous PV inversion solving collinear equation.

    q = Δѱ - f_0^2 * (A_11 + ɑ * A_12) * ѱ.
    """

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        alpha: torch.Tensor,
        f0: float,
        dx: float,
        dy: float,
        masks: Masks | None = None,
    ) -> None:
        """Potential vorticity inversion with homogeneous boundary.

        Args:
            A (torch.Tensor): Stretching matrix.
                └── (..., 1, 2, 2)-shaped
            alpha (torch.Tensor): Collinearity coefficient.
                └── (..., 1, nx+1, ny+1)-shaped
            f0 (float): Coriolis parameter.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
            masks (Masks | None, optional): Masks. Defaults to None.
        """
        BasePVInversion.__init__(self, A[:1, :1], f0, dx, dy, masks)
        self._A12 = A[0, 1]
        self._alpha = alpha
        if masks is not None and len(masks.psi_irrbound_xids) > 0:
            raise NotImplementedError

        self._compute_homsol = self._homsol_regular_geometry
        self._solve_modespace = self._solve_regular_geometry

    def _compute_helmholtz_dstI(  # noqa: N802
        self,
        nx: int,
        ny: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute the Helmholtz operator in spectral space.

        Args:
            nx (int): Number of grid points in the x-direction.
            ny (int): Number of grid points in the y-direction.
            dtype (torch.dtype | None, optional): Data type of the tensor.
                Defaults to None.
            device (torch.device | None, optional): Device to create
            the tensor on. Defaults to None.

        Returns:
            torch.Tensor: Helmholtz operator in spectral space.
        """
        laplacian = compute_laplace_dst(
            nx, ny, self._dx, self._dy, defaults.get(dtype=dtype, device=device)
        )
        laplacian = laplacian.unsqueeze(0).unsqueeze(0)
        return laplacian - self._f0**2 * (
            self._lambd + self._alpha[..., 1:-1, 1:-1] * self._A12
        )

    def _compute_homogeneous_solution(
        self,
        nl: int,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Compute homogeneous solution to use for mass conservation.

        Args:
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype | None, optional): Data type.
                Defaults to None.
            device (torch.device | None, optional): Device.
                Defaults to None.
        """
        cst = torch.ones(
            (1, nl, nx + 1, ny + 1),
            device=device,
            dtype=dtype,
        )
        sol = self._compute_homsol(cst, self._helmholtz_dstI)
        self._homsol = cst + sol * self._f0**2 * (self._lambd + self._alpha * self._A12)
        self._homsol_mean = self._homsol.mean((-1, -2), keepdim=True)
        return self._homsol, self._homsol_mean


class InhomogeneousPVInversionCollinear(InhomogeneousPVInversion):
    """Inhomogeneous PV inversion solving collinear equation.

    q = Δѱ - f_0^2 * (A_11 + ɑ * A_12) * ѱ.
    """

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        alpha: torch.Tensor,
        f0: float,
        dx: float,
        dy: float,
        masks: Masks | None = None,
    ) -> None:
        """Potential vorticity inversion with inhomogeneous boundary.

        Args:
            A (torch.Tensor): Stretching matrix.
                └── (..., 1, 2, 2)-shaped
            alpha (torch.Tensor): Collinearity coefficient.
                └── (..., 1, nx+1, ny+1)-shaped
            f0 (float): Coriolis parameter.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
            masks (Masks | None, optional): Masks. Defaults to None.
        """
        BasePVInversion.__init__(self, A, f0, dx, dy, masks)
        self._homogeneous_solver = HomogeneousPVInversionCollinear(
            A, alpha, f0, dx, dy, masks
        )
        self._alpha = alpha
        self._A11 = A[0, 0]
        self._A12 = A[0, 1]
        if self._masks is not None and len(self._masks.psi_irrbound_xids) > 0:
            msg = "Irregular geometries not supported for inhomogeneous boundaries."
            raise NotImplementedError(msg)

    def _compute_interior_pv_boundary(self) -> torch.Tensor:
        laplacian_boundary = self._boundary.compute_laplacian(self._dx, self._dy)
        sf_boundary_interior = self._boundary.compute()[..., 1:-1, 1:-1]
        return (
            laplacian_boundary
            - self._f0**2
            * (self._A11 + self._alpha[..., 1:-1, 1:-1] * self._A12)
            * sf_boundary_interior
        )
