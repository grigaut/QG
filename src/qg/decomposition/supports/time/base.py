"""Base time support function."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class TimeSupportFunction(ABC):
    """Base class for time support functions."""

    @abstractmethod
    def __init__(
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, torch.Tensor],
    ) -> None:
        """Instantiate the time support function.

        Args:
            time_params (dict[int, dict[str, Any]]): Parameters.
            space_fields (dict[int, torch.Tensor]): Already computed
                space fields.
        """

    @abstractmethod
    def decompose(self, t: torch.Tensor) -> dict[int, torch.Tensor]:
        """Compute level-wise space-time fields.

        Args:
            t (Tensor): Time.

        Returns:
            dict[int, Tensor]: Lvl -> space-time field.
        """

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the total resulting space-time field.

        Args:
            t (Tensor): Time.

        Returns:
            Tensor: Space-time field.
        """

    @abstractmethod
    def decompose_dt(self, t: torch.Tensor) -> dict[int, torch.Tensor]:
        """Compute level-wise time-derivated space-time fields.

        Args:
            t (Tensor): Time.

        Returns:
            dict[int, Tensor]: Lvl -> space-time field.
        """

    @abstractmethod
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the total resulting time derivatived field.

        Args:
            t (Tensor): Time.

        Returns:
            Tensor: Space-time field.
        """
