"""Full domain masks."""

from __future__ import annotations

import torch
from torch import Tensor

from qg.logging.core import getLogger
from qg.logging.utils import sec2text, tree
from qg.observations.base import BaseObservationMask

logger = getLogger(__name__)


class FullDomainMask(BaseObservationMask):
    """Mask for full domain observations."""

    _track_validity_time: float = 3600

    @property
    def track_validity_time(self) -> float:
        """Validity time of the track (seconds).

        A track is considered for a short amount of time after its passage.
        """
        return self._track_validity_time

    @property
    def dt(self) -> float:
        """Time interval between two consecutive tracks (seconds)."""
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        self._dt = dt
        msg = f"dt set to {sec2text(dt)}."
        logger.info(msg)

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        dt: float = 3600,
    ) -> None:
        """Instantiate the observing system.

        Args:
            x (torch.Tensor): X locations.
                └── (nx, ny) shaped
            y (torch.Tensor): Y locations.
                └── (nx, ny) shaped
            dt (float, optional): Time interval between consecutive tracks
                (seconds).

        Raises:
            ValueError: If theta is not in [0, pi/2).
        """
        super().__init__(x, y)
        self.dt = dt

    def at_time(self, time: Tensor) -> Tensor:
        """Compute observation mask at given time..

        Args:
            time (torch.Tensor): Time of the observation.

        Returns:
            torch.Tensor: Mask.
                └── (nx, ny) shaped
        """
        j = (time / self.dt).round().to(torch.int64)

        if (j * self.dt - time).abs() > self.track_validity_time / 2:
            return torch.zeros_like(self._y, dtype=torch.bool)

        return torch.ones_like(self._x, dtype=torch.bool)

    def get_repr_parts(self) -> list[str | list]:
        """String representations parts.

        Returns:
            list[str | list]: String representation parts.
        """
        return [self.__class__.__name__, [f"dt: {sec2text(self._dt)}"]]

    def __repr__(self) -> str:
        """String representation."""
        return tree(*self.get_repr_parts())
