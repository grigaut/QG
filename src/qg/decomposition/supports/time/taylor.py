"""Taylor series time supports."""

# ruff: noqa: D102, D107
from math import factorial
from typing import Any

import torch
from torch._tensor import Tensor

from qg.decomposition.supports.time.base import TimeSupportFunction


class TaylorSeriesTimeSupport(TimeSupportFunction):
    """Time support made from Taylor series."""

    def __init__(
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, Tensor],
    ) -> None:
        self._space = space_fields
        self.params = time_params

    __init__.__doc__ = TimeSupportFunction.__init__.__doc__

    def decompose(self, t: Tensor) -> dict[int, Tensor]:
        fields = {}
        for lvl in self.params:
            fact_lvl = factorial(lvl)

            fields[lvl] = torch.einsum("t,txy->xy", t**lvl / fact_lvl, self._space[lvl])
        return fields

    decompose.__doc__ = TimeSupportFunction.decompose.__doc__

    def __call__(self, t: Tensor) -> Tensor:
        fields = self.decompose(t)
        return sum(fields.values())

    __call__.__doc__ = TimeSupportFunction.__call__.__doc__

    def decompose_dt(self, t: Tensor) -> dict[int, Tensor]:
        fields = {}
        for lvl in self.params:
            if lvl == 0:
                fields[lvl] = torch.zeros_like(self._space[lvl][0])
                continue
            fact_lvl = factorial(lvl - 1)
            fields[lvl] = torch.einsum(
                "t,txy->xy", t ** (lvl - 1) / fact_lvl, self._space[lvl]
            )
        return fields

    decompose_dt.__doc__ = TimeSupportFunction.decompose_dt.__doc__

    def dt(self, t: Tensor) -> Tensor:
        fields = self.decompose_dt(t)
        return sum(fields.values())

    dt.__doc__ = TimeSupportFunction.dt.__doc__
