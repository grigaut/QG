"""Non-homogenous boundary conditions."""

from __future__ import annotations

from mqgeometry.solver.boundary_conditions.base import Boundaries
from mqgeometry.specs import defaults

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch

from mqgeometry import specs
from mqgeometry.fd import laplacian1D


class BilinearExtendedBoundary:
    """Boundary extrapolation using bilinear functions."""

    def __init__(self, boundaries: Boundaries) -> None:
        """Instantiate the boundary condition.

        Args:
            boundaries (Boundaries): Boundary conditions.
        """
        if boundaries.width != 1:
            msg = "Bilinear extension only works for 1-point wide boundaries."
            raise ValueError(msg)
        self._boundaries = boundaries
        self._compute_grids(
            boundaries.nx,
            boundaries.ny,
            **specs.from_tensor(boundaries.top),
        )

    def _compute_grids(
        self,
        nx: int,
        ny: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Compute the grids.

        Args:
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype | None, optional): Data type for the grid.
                Defaults to None.
            device (torch.device | None, optional): Device for the grid.
                Defaults to None.
        """
        tensor_specs = defaults.get(dtype=dtype, device=device)
        self._x = torch.linspace(0, 1, nx, **tensor_specs)
        self._y = torch.linspace(0, 1, ny, **tensor_specs)

    def compute(
        self,
    ) -> torch.Tensor:
        """Extrapolate boundary values on rectangular domain to the interior.

        Approach: find a bilinear function that matches values at corners
            and interpolate residual boundary data using linear function
            of the complementary coordinate.

        Returns:
            torch.Tensor: Bilinear-extended boundary field
                └── (..., nl, nx, ny)-shaped
        """
        ## Compute grid parameters
        xx, yy = torch.meshgrid(self._x, self._y, indexing="ij")
        # Reshape
        xx = xx[None, :, :]  # (1, nx, ny)-shaped
        yy = yy[None, :, :]  # (1, nx, ny)-shaped
        x = self._x[None, :, None]  # (1, nx, 1)-shaped
        y = self._y[None, None, :]  # (1, 1, ny)-shaped
        ## Variables
        ut = self._boundaries.top[..., :, 0]
        ub = self._boundaries.bottom[..., :, 0]
        ul = self._boundaries.left[..., 0, :]
        ur = self._boundaries.right[..., 0, :]
        ## Compute corner values
        u_bl = (ub[..., :, 0] + ul[..., :, 0]) / 2  # (..., nl)-shaped
        u_br = (ub[..., :, -1] + ur[..., :, 0]) / 2  # (..., nl)-shaped
        u_tl = (ut[..., :, 0] + ul[..., :, -1]) / 2  # (..., nl)-shaped
        u_tr = (ut[..., :, -1] + ur[..., :, -1]) / 2  # (..., nl)-shaped
        # Reshape
        u_bl = u_bl[..., :, None, None]  # (..., nl, 1, 1)-shaped
        u_br = u_br[..., :, None, None]  # (..., nl, 1, 1)-shaped
        u_tl = u_tl[..., :, None, None]  # (..., nl, 1, 1)-shaped
        u_tr = u_tr[..., :, None, None]  # (..., nl, 1, 1)-shaped
        ## Boundary values that are zero at corners
        # ula and ura: (..., nl, 1, ny)-shaped
        ula = ul[..., :, None, :] - u_bl * (1 - y) - u_tl * y
        ura = ur[..., :, None, :] - u_br * (1 - y) - u_tr * y
        # uta and uba: (..., nl, nx, 1)-shaped
        uba = ub[..., :, :, None] - u_bl * (1 - x) - u_br * x
        uta = ut[..., :, :, None] - u_tl * (1 - x) - u_tr * x
        # Sum linear extension of zero-ed boundary values + corner contribution
        # Empty corner contribution
        u_empty_corner = ula * (1 - xx) + ura * xx + uba * (1 - yy) + uta * yy
        # Corner contribution
        u_corners = (
            u_bl * (1 - xx) * (1 - yy)
            + u_br * xx * (1 - yy)
            + u_tl * (1 - xx) * yy
            + u_tr * xx * yy
        )
        return u_empty_corner + u_corners

    def compute_laplacian(
        self,
        dx: float,
        dy: float,
    ) -> torch.Tensor:
        """Compute the laplacian of the bilinear-extended boundary field.

        Δu = (1-x)∂²_y ul + x∂²_y ur + (1-y)∂²_x ub + y∂²_x ut

        Args:
            dx (float): Infinitesimal distance in the x direction.
            dy (float): Infinitesimal distance in the y direction.

        Returns:
            torch.Tensor: Laplacian of the bilinear-extended field obtained
            using `boundary_extension_bilinear`.
                └── (..., nl, nx-2, ny-2)-shaped
        """
        # Compute grid
        x_in = self._x[1:-1]
        y_in = self._y[1:-1]
        xx_in, yy_in = torch.meshgrid(x_in, y_in, indexing="ij")
        xx_in = xx_in[None, :, :]  # (1, nx, ny)-shaped
        yy_in = yy_in[None, :, :]  # (1, nx, ny)-shaped
        ## Variables
        ut = self._boundaries.top[..., :, 0]
        ub = self._boundaries.bottom[..., :, 0]
        ul = self._boundaries.left[..., 0, :]
        ur = self._boundaries.right[..., 0, :]
        # 1D laplacian of x-wise boundary conditions
        lap_ub = laplacian1D(ub, dx)
        lap_ut = laplacian1D(ut, dx)
        # 1D laplacian of y-wise boundary conditions
        lap_ul = laplacian1D(ul, dy)
        lap_ur = laplacian1D(ur, dy)
        # Reshape
        lap_ub = lap_ub[..., :, :, None]  # (..., nl, nx, 1)-shaped
        lap_ut = lap_ut[..., :, :, None]  # (..., nl, nx, 1)-shaped
        lap_ul = lap_ul[..., :, None, :]  # (..., nl, 1, ny)-shaped
        lap_ur = lap_ur[..., :, None, :]  # (..., nl, 1, ny)-shaped

        return (
            (1 - xx_in) * lap_ul
            + xx_in * lap_ur
            + (1 - yy_in) * lap_ub
            + yy_in * lap_ut
        )  # (..., nl, nx, ny)-shaped

    @classmethod
    def from_tensors(
        cls,
        top: torch.Tensor,
        bottom: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Self:
        """Instantiate the boundary interpolation using tensors.

        Args:
            top (torch.Tensor): Boundary condition at the top (y=y_max)
                └── (..., nl, ny)-shaped
            bottom (torch.Tensor): Boundary condition at the bottom (y=y_min)
                └── (..., nl, ny)-shaped
            left (torch.Tensor): Boundary condition on the left (x=x_min)
                └── (..., nl, nx)-shaped
            right (torch.Tensor): Boundary condition on the right (x=x_max)
                └── (..., nl, nx)-shaped

        Returns:
            Self: BilinearExtendedBoundary
        """
        return cls(Boundaries(top=top, bottom=bottom, left=left, right=right))
