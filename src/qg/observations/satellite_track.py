"""Satellite-track-like observation system."""

from __future__ import annotations

import numpy as np
import torch

from qg.logging.core import getLogger
from qg.logging.utils import meters2text, sec2text, tree
from qg.observations.base import BaseObservationMask
from qg.specs import defaults

logger = getLogger(__name__)


class SatelliteTrackMask(BaseObservationMask):
    """Satellite-track-like observation system."""

    _track_validity_time: float = 3600

    @property
    def track_width(self) -> float:
        """Track width (meters)."""
        return self._track_width

    @property
    def track_interval(self) -> float:
        """Interval between two consecutive tracks (meters)."""
        return self._track_interval

    @property
    def width(self) -> float:
        """Total width (track + interval)."""
        return self.track_width + self.track_interval

    @property
    def theta(self) -> float:
        """Track orientation (radians)."""
        return self._theta

    @property
    def dt(self) -> float:
        """Time interval between two consecutive tracks (seconds)."""
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        self._dt = dt
        self._full_coverage_time = self._dt * self._full_coverage_nb_ite
        msg = (
            f"dt set to {sec2text(dt)}, full coverage time"
            f" updated to {sec2text(self._full_coverage_time)}."
        )
        logger.info(msg)

    @property
    def full_coverage_time(self) -> float:
        """Time required for (first) full coverage (seconds)."""
        return self._full_coverage_time

    @full_coverage_time.setter
    def full_coverage_time(self, time: float) -> None:
        self._full_coverage_time = time
        self._dt = self._compute_dt_from_full_coverage_ite(time)
        msg = (
            f"Full coverage time set to {sec2text(time)}, "
            f"dt updated to {sec2text(self._dt)}."
        )
        logger.info(msg)

    @property
    def track_validity_time(self) -> float:
        """Validity time of the track (seconds).

        A track is considered for a short amount of time after its passage.
        """
        return self._track_validity_time

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        track_width: float = 100000,
        track_interval: float = 500000,
        theta: float = torch.pi / 12,
        full_coverage_time: float = 20 * 3600 * 24,
    ) -> None:
        """Instantiate the observing system.

        Args:
            x (torch.Tensor): X locations.
                └── (nx, ny) shaped
            y (torch.Tensor): Y locations.
                └── (nx, ny) shaped
            track_width (float, optional): Track width (meters).
                Defaults to 100000.
            track_interval (float, optional): Interval between consecutive
                tracks (meters). Defaults to 500000.
            theta (float, optional): Track orientation (radians).
                Defaults to torch.pi/12.
            full_coverage_time (float, optional): Time required for (first)
                full coverage (seconds). Defaults to 20*3600*24.

        Raises:
            ValueError: If theta is not in [0, pi/2).
        """
        super().__init__(x, y)

        self._y_origin = y[0, 0]

        self._track_width = track_width
        self._track_interval = track_interval
        if theta >= torch.pi / 2 or theta < 0:
            msg = "Theta must be greater than or equal to 0 and less than pi/2 radians."
            raise ValueError(msg)
        self._theta = theta
        self._tan_theta = torch.tan(torch.tensor(theta, **defaults.get()))

        x_range = self._compute_acceptable_xs(x[:, 0], y[0, :])
        self._x_offset = x_range[0]
        self._x_range = x_range - self._x_offset
        self._x_max = self._x_range[-1]
        self._assert_valid_params()
        self._full_coverage_nb_ite = self._compute_full_coverage_nb_ite()
        self.full_coverage_time = full_coverage_time

    def get_repr_parts(self) -> list[str | list]:
        """String representations parts.

        Returns:
            list[str | list]: String representation parts.
        """
        return [
            self.__class__.__name__,
            [
                f"Track width: {meters2text(self.track_width)}",
                f"Interval width: {meters2text(self.track_interval)}",
                f"Orientation: {self.theta / torch.pi:.2f} π",
                f"dt: {sec2text(self._dt)}",
            ],
        ]

    def __repr__(self) -> str:
        """String representation."""
        return tree(
            *self.get_repr_parts(),
        )

    def _compute_acceptable_xs(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute acceptable X positions to draw from to generate tracks.

        The acceptable X positions are such that tracks centered on
        these positions can cover the whole domain.

        It is offseted to account for track orientation.

        Args:
            x (torch.Tensor): X locations.
                └── (nx,) shaped
            y (torch.Tensor): Y locations.
                └── (ny,) shaped

        Returns:
            torch.Tensor: Acceptable X positions.
        """
        dx = x[1] - x[0]
        x_offset = self._tan_theta * (y[-1] - y[0])
        x_o_grid = (torch.div(x_offset, dx, rounding_mode="floor") + 1) * dx
        x_min = x[0] - x_o_grid
        x_max = x[-1]
        return torch.arange(x_min, x_max + dx, step=dx)

    def _assert_valid_params(self, max_ite: int = 500) -> None:
        """Check that parameters allow for full coverage.

        This method does not actually compute the number of iterations,
        it only checks that full coverage can be achieved. It compute the set
        of all starting positions and ensure that the maximum between two
        consecutive position is smaller that the track width.

        Args:
            max_ite (int, optional): Max number of iterations
                to have full coverage. Defaults to 500.

        Raises:
            ValueError: If full coverage can not be achieved in
                max_ite iterations.
        """
        starts = set()
        for i in range(max_ite):
            start = (i * (self.width)) % self._x_max
            starts.add(start)

            if i == 0:
                continue

            starts_sorted = np.array(sorted(starts))

            if np.max(starts_sorted[1:] - starts_sorted[:-1]) < self.track_width:
                msg = "Specified track parameters ensure full coverage."
                logger.info(msg)
                return
        msg = "Specified track parameters do not ensure full coverage."
        raise ValueError(msg)

    def _compute_full_coverage_nb_ite(self, max_ite: int = 500) -> int:
        """Compute number of iterations for full coverage.

        Args:
            max_ite (int, optional): Max number of iterations
                to have full coverage. Defaults to 500.

        Raises:
            ValueError: If full coverage is not achieved in max_ite iterations.

        Returns:
            int: Number of iterations.
        """
        tracks = torch.zeros_like(self._y, dtype=torch.bool)
        for j in range(max_ite):
            tracks = self.at_index(j) | tracks

            if tracks.all():
                msg = f"Full coverage achieved in {j + 1} iterations."
                logger.info(msg)
                return j + 1
        msg = f"Full coverage not achieved in {max_ite} iterations."
        raise ValueError(msg)

    def _compute_dt_from_full_coverage_ite(self, full_coverage_time: float) -> float:
        return full_coverage_time / self._full_coverage_nb_ite

    def at_index(self, j: int) -> torch.Tensor:
        """Compute observation mask at given passage number.

        Args:
            j (int): Passage number.

        Returns:
            torch.Tensor: Mask.
                └── (nx, ny) shaped
        """
        xc = ((self._x_range[0] + j * (self.width)) % self._x_max) + self._x_offset
        x0 = xc - self.track_width / 2
        x1 = xc + self.track_width / 2

        y_rel = self._y - self._y_origin
        x_expanded = self._x[:, :1]

        is_below_track = y_rel <= (x_expanded - x0) / self._tan_theta
        is_above_track = y_rel >= (x_expanded - x1) / self._tan_theta
        return (is_below_track & is_above_track).to(torch.bool)

    def at_time(self, time: torch.Tensor) -> torch.Tensor:  # noqa: D102
        j = (time / self.dt).round().to(torch.int64)

        if (j * self.dt - time).abs() > self.track_validity_time / 2:
            return torch.zeros_like(self._y, dtype=torch.bool)

        return self.at_index(j.item())

    at_time.__doc__ = BaseObservationMask.at_time.__doc__
