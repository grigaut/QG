"""Input/output utilities for boundary conditions."""

from __future__ import annotations

from pathlib import Path

import torch

from qgsw.solver.boundary_conditions.base import Boundaries, TimedBoundaries
from qgsw.specs import defaults


class BoundaryConditionSaver:
    """Saver for boundary conditions."""

    time_key = "time"
    top_key = "top"
    bottom_key = "bottom"
    left_key = "left"
    right_key = "right"

    _times: torch.Tensor | None = None
    _tops: torch.Tensor | None = None
    _bottoms: torch.Tensor | None = None
    _left: torch.Tensor | None = None
    _right: torch.Tensor | None = None

    def __init__(self, file: Path | str) -> None:
        """Instantiate the saving tool.

        Args:
            file (Path | str): The file path to save the boundary conditions.

        Raises:
            ValueError: If the file extension is not .pt.
        """
        self._file = Path(file)
        if self._file.suffix != ".pt":
            msg = "File must have a .pt extension."
            raise ValueError(msg)
        # Create the output directory
        self._file.parent.mkdir(parents=True, exist_ok=True)

    def _create(self, time_boundary: TimedBoundaries) -> None:
        self._times = torch.tensor([time_boundary.time])
        self._tops = time_boundary.boundaries.top.unsqueeze(0)
        self._bottoms = time_boundary.boundaries.bottom.unsqueeze(0)
        self._left = time_boundary.boundaries.left.unsqueeze(0)
        self._right = time_boundary.boundaries.right.unsqueeze(0)

    def append(self, time_boundary: TimedBoundaries) -> None:
        """Append boundaries to the saver.

        Args:
            time_boundary (TimedBoundaries): The time boundary to append.

        Returns:
            None
        """
        if self._times is None:
            return self._create(time_boundary)
        time = torch.tensor([time_boundary.time])
        top = time_boundary.boundaries.top.unsqueeze(0)
        bottom = time_boundary.boundaries.bottom.unsqueeze(0)
        left = time_boundary.boundaries.left.unsqueeze(0)
        right = time_boundary.boundaries.right.unsqueeze(0)
        self._times = torch.cat([self._times, time])
        self._tops = torch.cat([self._tops, top])
        self._bottoms = torch.cat([self._bottoms, bottom])
        self._left = torch.cat([self._left, left])
        self._right = torch.cat([self._right, right])
        return None

    def append_tensors(
        self,
        *,
        time: float,
        top: torch.Tensor,
        bottom: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> None:
        """Append boundaries to the saver using tensors.

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
            None
        """
        return self.append(
            TimedBoundaries.from_tensors(
                time=time, top=top, bottom=bottom, left=left, right=right
            )
        )

    def save(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Save the boundary conditions to a file.

        Args:
            dtype (torch.dtype | None, optional): Data type to save with.
                Defaults to None.
            device (torch.device | None, optional): Device to save on.
                Defaults to None.
        """
        save_specs = defaults.get_save_specs(dtype=dtype, device=device)
        to_save = {
            self.time_key: self._times.to(**save_specs),
            self.top_key: self._tops.to(**save_specs),
            self.bottom_key: self._bottoms.to(**save_specs),
            self.left_key: self._left.to(**save_specs),
            self.right_key: self._right.to(**save_specs),
        }
        torch.save(to_save, self._file)


class BoundaryConditionLoader:
    """Loader for boundary conditions."""

    time_key = BoundaryConditionSaver.time_key
    top_key = BoundaryConditionSaver.top_key
    bottom_key = BoundaryConditionSaver.bottom_key
    left_key = BoundaryConditionSaver.left_key
    right_key = BoundaryConditionSaver.right_key

    def __init__(self, file: Path | str) -> None:
        """Instantiate the loading tool.

        Args:
            file (Path | str): The file path to load the boundary conditions.

        Raises:
            ValueError: If the file extension is not .pt.
        """
        self._file = Path(file)
        if self._file.suffix != ".pt":
            msg = "File must have a .pt extension."
            raise ValueError(msg)

    def load(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> tuple[list[float], list[Boundaries]]:
        """Load the boundary conditions from a file.

        Returns:
            list[list[float], list[Boundaries]]: The loaded time
                and boundary conditions.
        """
        load_specs = defaults.get(dtype=dtype, device=device)
        data = torch.load(self._file)
        return zip(
            *[
                (
                    time,
                    Boundaries(
                        top=top.to(**load_specs),
                        bottom=bottom.to(**load_specs),
                        left=left.to(**load_specs),
                        right=right.to(**load_specs),
                    ),
                )
                for time, top, bottom, left, right in zip(
                    data[self.time_key],
                    data[self.top_key],
                    data[self.bottom_key],
                    data[self.left_key],
                    data[self.right_key],
                )
            ]
        )
