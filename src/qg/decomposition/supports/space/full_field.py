"""Full field."""

from __future__ import annotations

import torch

from qg.decomposition.supports.space.base import SpaceSupportFunction
from qg.specs import defaults


class FullFieldSpaceSupport(SpaceSupportFunction):
    """Full field space support."""

    def __init__(
        self,
        nx: torch.Tensor,
        ny: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Full field.

        Args:
            nx (torch.Tensor): Number of points in the x direction.
            ny (torch.Tensor): Number of points in the y direction.
            dtype (torch.dtype | None, optional): Data type. Defaults to None.
            device (torch.device | None, optional): Device. Defaults to None.
        """
        self._nx = nx
        self._ny = ny
        self._specs = defaults.get(dtype=dtype, device=device)

    def _compute(self) -> torch.Tensor:
        return torch.ones((1, self._nx, self._ny), **self._specs)
