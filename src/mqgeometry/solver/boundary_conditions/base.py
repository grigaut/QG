"""Base classes for boundary conditions."""

from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from dataclasses import dataclass
from functools import cached_property
from typing import Literal, overload

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass(frozen=True)
class Boundaries:
    """Named tuple for boundary conditions.

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

    top: torch.Tensor
    bottom: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    def __add__(self, other: Boundaries) -> Boundaries:
        """Add two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top + other.top,
            bottom=self.bottom + other.bottom,
            left=self.left + other.left,
            right=self.right + other.right,
        )

    def __radd__(self, other: Boundaries) -> Boundaries:
        """Add two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top + other.top,
            bottom=self.bottom + other.bottom,
            left=self.left + other.left,
            right=self.right + other.right,
        )

    def __sub__(self, other: Boundaries) -> Boundaries:
        """Subtract two boundary conditions."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return Boundaries(
            top=self.top - other.top,
            bottom=self.bottom - other.bottom,
            left=self.left - other.left,
            right=self.right - other.right,
        )

    def __neg__(self) -> Boundaries:
        """Implements behavior for negation using the - operator."""
        return Boundaries(
            top=self.top.__neg__(),
            bottom=self.bottom.__neg__(),
            left=self.left.__neg__(),
            right=self.right.__neg__(),
        )

    def __rmul__(self, scalar: float) -> Boundaries:
        """Multiply boundary condition with scalar."""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return Boundaries(
            top=self.top * scalar,
            bottom=self.bottom * scalar,
            left=self.left * scalar,
            right=self.right * scalar,
        )

    def __mul__(self, scalar: float) -> Boundaries:
        """Multiply boundary condition with scalar."""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return Boundaries(
            top=self.top * scalar,
            bottom=self.bottom * scalar,
            left=self.left * scalar,
            right=self.right * scalar,
        )

    def __eq__(self, other: object) -> bool:
        """Check if two boundary conditions are equal."""
        if not isinstance(other, Boundaries):
            return NotImplemented
        return (
            torch.equal(self.top, other.top)
            and torch.equal(self.bottom, other.bottom)
            and torch.equal(self.left, other.left)
            and torch.equal(self.right, other.right)
        )

    __hash__ = None

    def __truediv__(self, scalar: float) -> Boundaries:
        """Multiply boundary condition with scalar."""
        if not isinstance(scalar, (float, int)):
            return NotImplemented
        return Boundaries(
            top=self.top / scalar,
            bottom=self.bottom / scalar,
            left=self.left / scalar,
            right=self.right / scalar,
        )

    def __post_init__(self) -> None:
        """Post initialization method.

        Raises:
            ValueError: If the boundary shapes are not compatible.
            ValueError: If the boundary shapes are not compatible.
        """
        if self.bottom.shape != self.top.shape:
            msg = "Both bottom and top boundaries must have the same shape."
            raise ValueError(msg)
        if self.left.shape != self.right.shape:
            msg = "Both left and right boundaries must have the same shape."
            raise ValueError(msg)
        if self.left.shape[-2] != self.bottom.shape[-1]:
            msg = (
                "The width of left/right boundaries must match "
                "the width of top/bottom boundaries."
            )
            raise ValueError(msg)

    def get_band(self, index: int = 0) -> Boundaries:
        """Get a specific band of the boundary conditions.

        Index 0 is the inner band.

        Args:
            index (int, optional): Index of the band to get. Defaults to 0.

        Returns:
            Boundaries: The band of the boundary conditions.
        """
        if self.width == 1:
            return self
        w = self.width
        i = index
        if (i + 1) > w:
            msg = f"Index {i} is out of bounds for width {w}."
            raise ValueError(msg)
        offset = w - i - 1
        s = slice(offset, -offset if offset != 0 else None)
        top = self.top[..., s, i]
        bottom = self.bottom[..., s, -i - 1]
        left = self.left[..., -i - 1, s]
        right = self.right[..., i, s]
        return Boundaries(
            top=top[..., :, None],
            bottom=bottom[..., :, None],
            left=left[..., None, :],
            right=right[..., None, :],
        )

    def __getitem__(self, s: slice | int | tuple[slice | int]) -> Boundaries:
        """Slice boundaries.

        Args:
            s (slice | int | list[slice  |  int]): Slicing arguments.

        Raises:
            ValueError: If one of the two last dimensions is sliced.
            ValueError: If one of the two last dimensions is sliced.

        Returns:
            Boundaries: Sliced boundary.
        """
        d = self.top.dim() - 2
        if isinstance(s, tuple):
            while s[-1] is ...:
                s = s[:-1]
            invalid_1 = s[-1] != slice(None)
            invalid_2 = len(s) > 2 and s[-2] != slice(None)  # noqa: PLR2004
            invalid_end = invalid_1 or invalid_2
            if invalid_end and (... in s or len(s) > d):
                msg = f"Slicing is only allowed on the {d} first dimensions."
                raise ValueError(msg)
        elif d == 0:
            msg = f"Slicing is only allowed on the {d} first dimensions."
            raise ValueError(msg)
        return Boundaries(
            top=self.top.__getitem__(s),
            bottom=self.bottom.__getitem__(s),
            left=self.left.__getitem__(s),
            right=self.right.__getitem__(s),
        )

    @cached_property
    def nx(self) -> int:
        """Number of points in the x direction."""
        return self.top.shape[-2]

    @cached_property
    def ny(self) -> int:
        """Number of points in the y direction."""
        return self.left.shape[-1]

    @cached_property
    def width(self) -> int:
        """Width of the boundary conditions."""
        return self.top.shape[-1]

    @classmethod
    def _compute_ij(
        cls,
        field: torch.Tensor,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
    ) -> tuple[int, int, int, int]:
        nx, ny = field.shape[-2:]
        return (
            (nx + 1) * (imin < 0) + imin,
            (nx + 1) * (imax < 0) + imax,
            (ny + 1) * (jmin < 0) + jmin,
            (ny + 1) * (jmax < 0) + jmax,
        )

    @classmethod
    def extract(
        cls,
        field: torch.Tensor,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
        width: int = 1,
    ) -> Self:
        """Extract boundary conditions from a field.

        A good rule of thumb to not mess up with index is that,
        if data has been slice using [imin:imax,jmin:jmax], one needs to
        perform Boundaries.extract(imin-1,imax+1,jmin-1,jmax+1), to be able
        to expand the data slice.

        Example:
        >>> x = torch.rand((25,30))
        >>> imin=5
        >>> imax=10
        >>> jmin=10
        >>> jmax=20
        >>> x_slice = x[imin:imax,jmin:jmax]
        >>> bc = Boundaries.extract(x,imin-1,imax+1,jmin-1,jmax+1)
        >>> bc.expand(x_slice)

        Args:
            field (torch.Tensor): Field to extract from.
                └── (..., Nx, Ny)-shaped
            imin (int): Minimum index in the x direction.
            imax (int): Maximum index in the x direction.
            jmin (int): Minimum index in the y direction.
            jmax (int): Maximum index in the y direction.
            width (int, optional): Width of the boundary. Defaults to 1.

        Returns:
            Self: Extracted boundaries.
                ├── top: (..., width, imax-imin+2*width)-shaped
                ├── bottom: (..., width, imax-imin+2*width)-shaped
                ├── left: (..., jmax-jmin+2*width, width)-shaped
                └── right: (..., jmax-jmin+2*width, width)-shaped
        """
        w = width
        imin, imax, jmin, jmax = cls._compute_ij(field, imin, imax, jmin, jmax)
        return cls(
            top=field[..., imin - (w - 1) : imax + (w - 1), jmax - 1 : jmax + w - 1],
            bottom=field[
                ...,
                imin - (w - 1) : imax + (w - 1),
                jmin - (w - 1) : jmin + 1,
            ],
            left=field[
                ...,
                imin - (w - 1) : imin + 1,
                jmin - (w - 1) : jmax + (w - 1),
            ],
            right=field[
                ...,
                imax - 1 : imax + w - 1,
                jmin - (w - 1) : jmax + (w - 1),
            ],
        )

    @overload
    def set_to(
        self,
        field: torch.Tensor,
        *,
        offset: int,
        inplace: Literal[True],
    ) -> None: ...
    @overload
    def set_to(
        self,
        field: torch.Tensor,
        *,
        offset: int,
        inplace: Literal[False],
    ) -> torch.Tensor: ...
    def set_to(
        self,
        field: torch.Tensor,
        *,
        offset: int = 0,
        inplace: bool = True,
    ) -> torch.Tensor | None:
        """Add boundary to a given Tensor.

        Args:
            field (torch.Tensor): Field to add boundary to.
            offset (int, optional): Offset. If 0, the outer most border
                will be completed. Defaults to 0.
            inplace (bool, optional): Whether to perform operation inplace or
                not. Defaults to True.

        Returns:
            torch.Tensor | None: Tensor with boundary if inplace is True,
                otherwise nothing.
        """
        w = self.width
        o = offset

        if not inplace:
            field = torch.clone(field)

        nx, ny = field.shape[-2:]
        lx = nx - o * 2
        ly = ny - o * 2

        if lx != self.nx:
            msg = "Mismatching size at top/bottom boundary."
            raise ValueError(msg)
        if ly != self.ny:
            msg = "Mismatching size at left/right boundary."
            raise ValueError(msg)

        s_left = slice(o, o + w)
        s_right = slice(-(o + w), -o if o != 0 else None)

        field_left = field[..., s_left, :].narrow(-1, o, ly)
        field_left[:] = self.left

        field_right = field[..., s_right, :].narrow(-1, o, ly)
        field_right[:] = self.right

        field_bottom = field[..., :, s_left].narrow(-2, o, lx)
        field_bottom[:] = self.bottom

        field_top = field[..., :, s_right].narrow(-2, o, lx)
        field_top[:] = self.top

        return field if not inplace else None

    def expand(self, field: torch.Tensor) -> torch.Tensor:
        """Expand a tensor by adding the boundary.

        Args:
            field (torch.Tensor): Tensor to expand.

        Returns:
            torch.Tensor: Expanded tensor.
        """
        w = self.width
        return self.set_to(F.pad(field, (w, w, w, w)), inplace=False)


@dataclass(frozen=True)
class TimedBoundaries:
    """Named tuple for time-dependent boundary conditions."""

    time: float
    boundaries: Boundaries

    @classmethod
    def from_tensors(
        cls,
        time: float,
        top: torch.Tensor,
        bottom: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Self:
        """Instantiate the boundaries using tensors.

        Args:
            time (float): Time.
            top (torch.Tensor): Boundary condition at the top (y=y_max)
                └── (..., nl, ny)-shaped
            bottom (torch.Tensor): Boundary condition at the bottom (y=y_min)
                └── (..., nl, ny)-shaped
            left (torch.Tensor): Boundary condition on the left (x=x_min)
                └── (..., nl, nx)-shaped
            right (torch.Tensor): Boundary condition on the right (x=x_max)
                └── (..., nl, nx)-shaped

        Returns:
            Self: TimedBoundary.
        """
        return cls(
            time=time,
            boundaries=Boundaries(top=top, bottom=bottom, left=left, right=right),
        )

    @classmethod
    def extract(
        cls,
        time: float,
        field: torch.Tensor,
        imin: int,
        imax: int,
        jmin: int,
        jmax: int,
        width: int = 1,
    ) -> Self:
        """Extract boundary conditions from a field.

        Args:
            time (float): Time of the boundary conditions.
            field (torch.Tensor): Field to extract from.
                └── (..., Nx, Ny)-shaped
            imin (int): Minimum index in the x direction.
            imax (int): Maximum index in the x direction.
            jmin (int): Minimum index in the y direction.
            jmax (int): Maximum index in the y direction.
            width (int, optional): Width of the boundary. Defaults to 1.

        Returns:
            Self: Extracted boundaries.
                ├── top: (..., imax-imin+1)-shaped
                ├── bottom: (..., imax-imin+1)-shaped
                ├── left: (..., jmax-jmin+1)-shaped
                └── right: (..., jmax-jmin+1)-shaped
        """
        return cls(
            time=time,
            boundaries=Boundaries.extract(field, imin, imax, jmin, jmax, width),
        )
