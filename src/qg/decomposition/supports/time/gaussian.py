"""Gaussian time supports."""

# ruff: noqa: D107, D102
from typing import Any

import torch
from torch._tensor import Tensor

from qgsw import specs
from qg.decomposition.supports.time.base import TimeSupportFunction


class GaussianTimeSupport(TimeSupportFunction):
    """Gaussian time support."""

    def __init__(
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, Tensor],
    ) -> None:
        self._space = space_fields
        self.params = time_params
        self.unfreeze_normalization()

    __init__.__doc__ = TimeSupportFunction.__init__.__doc__

    def normalize(self, t: torch.Tensor, lvl: int, exp: torch.Tensor) -> torch.Tensor:
        """Compute normalization.

        Args:
            t (torch.Tensor): Time to normalize at.
            lvl (int): Lvl to normalize at.
            exp (torch.Tensor): Field.

        Returns:
            torch.Tensor: Normalized field.
        """
        if self.norms and (t >= self.freeze_t).cpu().item():
            return exp / self.norms[lvl]
        return exp / exp.sum(dim=0)

    def freeze_normalization(self, t: torch.Tensor) -> None:
        """Freeze normalization after a given time.

        Args:
            t (torch.Tensor): Time to freeze normalizing at.
        """
        self.norms = {}
        self.freeze_t = t.clone()
        tspecs = specs.from_tensor(t)
        for lvl, params in self.params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]

            tc = torch.tensor(centers, **tspecs)

            self.norms[lvl] = torch.exp(-((t - tc) ** 2) / (st) ** 2).sum(dim=0)

    def unfreeze_normalization(self) -> None:
        """Unfreeze normalization."""
        self.norms = {}

    def decompose(self, t: Tensor) -> dict[int, Tensor]:
        fields = {}
        tspecs = specs.from_tensor(t)
        for lvl, params in self.params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]

            tc = torch.tensor(centers, **tspecs)

            exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)
            exp_ = self.normalize(t, lvl, exp)

            fields[lvl] = torch.einsum("t,txy->xy", exp_, self._space[lvl])
        return fields

    decompose.__doc__ = TimeSupportFunction.decompose.__doc__

    def __call__(self, t: Tensor) -> Tensor:
        fields = self.decompose(t)
        return sum(fields.values()) / len(self.params)

    __call__.__doc__ = TimeSupportFunction.__call__.__doc__

    def dt_normalized(
        self,
        t: torch.Tensor,
        lvl: int,
        exp: torch.Tensor,
        dt_exp: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize time derivative of the field.

        Args:
            t (torch.Tensor): Time to normalize at.
            lvl (int): Lvl to normalize at.
            exp (torch.Tensor): Field.
            dt_exp (torch.Tensor): Time derivative of the field.

        Returns:
            torch.Tensor: Time derivative of normalized field.
        """
        if self.norms and (t >= self.freeze_t).cpu().item():
            return dt_exp / self.norms[lvl]
        exp_s = exp.sum(dim=0)

        return (dt_exp * exp_s - exp * dt_exp.sum(dim=0)) / exp_s**2

    def decompose_dt(self, t: Tensor) -> dict[int, Tensor]:
        fields = {}
        tspecs = specs.from_tensor(t)
        for lvl, params in self.params.items():
            centers = params["centers"]
            st: float = params["sigma_t"]
            space = self._space[lvl]

            tc = torch.tensor(centers, **tspecs)

            exp = torch.exp(-((t - tc) ** 2) / (st) ** 2)
            dt_exp = -2 * (t - tc) / st**2 * exp

            dt_e = self.dt_normalized(t, lvl, exp, dt_exp)

            fields[lvl] = torch.einsum("t,txy->xy", dt_e, space)
        return fields

    decompose_dt.__doc__ = TimeSupportFunction.decompose_dt.__doc__

    def dt(self, t: Tensor) -> Tensor:
        fields = self.decompose_dt(t)
        return sum(fields.values()) / len(self.params)

    dt.__doc__ = TimeSupportFunction.dt.__doc__
