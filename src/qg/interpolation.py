"""Interpolation tools."""

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


class _Interpolation(Generic[T], metaclass=ABCMeta):
    """Interpolation base class."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: float) -> T:
        """Get the y value at a specific x.

        Args:
            x (float): X value.

        Returns:
            T: Interpolation of ys evaluated at x.
        """


class LinearInterpolation(_Interpolation[T], Generic[T]):
    """Linear Interpolation."""

    def __init__(
        self,
        xs: Iterator[float],
        ys: list[T],
        *,
        remove_offset: bool = True,
    ) -> None:
        """Instantiate the linear interpolation.

        Args:
            xs (Iterator[float]): X values.
            ys (list[T]): Y values.
            remove_offset (bool, optional): Whether to remove an eventual x
            offset or not. Defaults to True.
        """
        xx = torch.tensor(list(xs), dtype=torch.float64)
        argsort = torch.argsort(xx)
        self._xs = xx[argsort]
        if remove_offset:
            self._xs = self._xs - self._xs[0]
        self._xmin = self._xs[0]
        self._xmax = self._xs[-1]
        self._ys = ys

    def __call__(self, x: float) -> T:
        """Get the y value at a specific x.

        Args:
            x (float): X value.

        Returns:
            T: Interpolation of ys evaluated at x.
        """
        if x < self._xmin or x > self._xmax:
            msg = (
                f"X must be greater than {self._xmin} "
                f"and lower than {self._xmax}, got {x}."
            )
            raise ValueError(msg)
        # Find the two surrounding time points
        i = torch.searchsorted(self._xs, x, right=True)
        if i == len(self._xs):
            return self._ys[-1]
        x0, x1 = self._xs[i - 1].item(), self._xs[i].item()
        y0, y1 = self._ys[i - 1], self._ys[i]
        # Linear interpolation
        alpha = (x - x0) / (x1 - x0)
        return y0 * (1 - alpha) + y1 * alpha


class ConstantInterpolation(_Interpolation[T], Generic[T]):
    """Linear Interpolation."""

    def __init__(
        self,
        y: T,
    ) -> None:
        """Instantiate the linear interpolation.

        Args:
            y (T): Y value.
        """
        self._y = y

    def __call__(self, x: float) -> T:  # noqa: ARG002
        """Get the y value at a specific x.

        Args:
            x (float): X value, only for compatibility.

        Returns:
            T: Interpolation of ys evaluated at x.
        """
        return self._y


class QuadraticInterpolation(_Interpolation[T], Generic[T]):
    """Quadratic interpolation."""

    def __init__(
        self,
        xs: Iterator[float],
        ys: list[T],
        *,
        remove_offset: bool = True,
    ) -> None:
        """Instantiate the quadratic interpolation.

        Args:
            xs (Iterator[float]): X values.
            ys (list[T]): Y values.
            remove_offset (bool, optional): Whether to remove an eventual x
            offset or not. Defaults to True.
        """
        xx = torch.tensor(list(xs), dtype=torch.float64)
        argsort = torch.argsort(xx)
        self._xs = xx[argsort]
        if remove_offset:
            self._xs = self._xs - self._xs[0]
        self._xmin = self._xs[0]
        self._xmax = self._xs[-1]
        self._ys = ys

    def __call__(self, x: float) -> T:
        """Get the y value at a specific x.

        Args:
            x (float): X value.

        Returns:
            T: Interpolation of ys evaluated at x.
        """
        if x < self._xmin or x > self._xmax:
            msg = (
                f"X must be greater than {self._xmin} "
                f"and lower than {self._xmax}, got {x}."
            )
            raise ValueError(msg)
        # Find the two surrounding time points
        i = torch.searchsorted(self._xs, x, right=True)
        if i == len(self._xs):
            return self._ys[-1]
        x0, x1 = self._xs[i - 1].item(), self._xs[i].item()
        alpha = (x - x0) / (x1 - x0)
        if i < 2 or i > len(self._xs) - 2:  # noqa: PLR2004
            y0, y1 = self._ys[i - 1], self._ys[i]
            # Linear interpolation
            return y0 * (1 - alpha) + y1 * alpha
        y_1, y0, y1, y2 = (
            self._ys[i - 2],
            self._ys[i - 1],
            self._ys[i],
            self._ys[i + 1],
        )
        a: T = 1 / 6 * (y2 - 3 * y1 + 3 * y0 - y_1)
        b: T = 1 / 2 * (y1 - 2 * y0 + y_1)
        c: T = 1 / 6 * (-1 * y2 + 6 * y1 - 3 * y0 - 2 * y_1)
        d: T = y0
        return a * alpha**3 + b * alpha**2 + c * alpha + d
