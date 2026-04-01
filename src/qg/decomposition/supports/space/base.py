"""Base class for space supports functions."""

from abc import ABC, abstractmethod
from functools import cached_property

import torch


class SpaceSupportFunction(ABC):
    """Support for space functions."""

    @cached_property
    def field(self) -> torch.Tensor:
        """Field."""
        return self._compute()

    @cached_property
    def dx(self) -> torch.Tensor:
        """X-derivative."""
        return self._compute_dx()

    @cached_property
    def dy(self) -> torch.Tensor:
        """Y-derivative."""
        return self._compute_dy()

    @cached_property
    def dx2(self) -> torch.Tensor:
        """Second X-derivative."""
        return self._compute_dx2()

    @cached_property
    def dydx(self) -> torch.Tensor:
        """X-Y-derivative."""
        return self._compute_dydx()

    @cached_property
    def dy2(self) -> torch.Tensor:
        """Second Y-derivative."""
        return self._compute_dy2()

    @cached_property
    def dxdy(self) -> torch.Tensor:
        """Y-X-derivative."""
        return self._compute_dxdy()

    @cached_property
    def dx3(self) -> torch.Tensor:
        """Third X-derivative."""
        return self._compute_dx3()

    @cached_property
    def dy3(self) -> torch.Tensor:
        """Third Y-derivative."""
        return self._compute_dy3()

    @cached_property
    def dydx2(self) -> torch.Tensor:
        """X-X-Y-derivative."""
        return self._compute_dydx2()

    @cached_property
    def dxdy2(self) -> torch.Tensor:
        """Y-Y-X-derivative."""
        return self._compute_dxdy2()

    @abstractmethod
    def _compute(self) -> torch.Tensor:
        """Method to compute field."""

    def _compute_dx(self) -> torch.Tensor:
        msg = "This space support does not implement x-derivative."
        raise NotImplementedError(msg)

    def _compute_dy(self) -> torch.Tensor:
        msg = "This space support does not implement y-derivative."
        raise NotImplementedError(msg)

    def _compute_dx2(self) -> torch.Tensor:
        msg = "This space support does not implement xx-derivative."
        raise NotImplementedError(msg)

    def _compute_dy2(self) -> torch.Tensor:
        msg = "This space support does not implement yy-derivative."
        raise NotImplementedError(msg)

    def _compute_dydx(self) -> torch.Tensor:
        msg = "This space support does not implement xy-derivative."
        raise NotImplementedError(msg)

    def _compute_dxdy(self) -> torch.Tensor:
        msg = "This space support does not implement yx-derivative."
        raise NotImplementedError(msg)

    def _compute_dx3(self) -> torch.Tensor:
        msg = "This space support does not implement xxx-derivative."
        raise NotImplementedError(msg)

    def _compute_dy3(self) -> torch.Tensor:
        msg = "This space support does not implement yyy-derivative."
        raise NotImplementedError(msg)

    def _compute_dydx2(self) -> torch.Tensor:
        msg = "This space support does not implement yxx-derivative."
        raise NotImplementedError(msg)

    def _compute_dxdy2(self) -> torch.Tensor:
        msg = "This space support does not implement xyy-derivative."
        raise NotImplementedError(msg)
