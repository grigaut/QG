"""Specs utils."""

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from typing import TypedDict

import torch


class Device:
    """Pytorch Device Object."""

    def __init__(self, device: str) -> None:
        """Instantiate the device.

        Args:
            device (str): Device value.
        """
        self._device = torch.device(device)

    def __repr__(self) -> str:
        """String representation of the variable.

        Returns:
            str: String representation of torch.device
        """
        return repr(self._device)

    def set_manually(self, device: str) -> None:
        """Manually set the device.

        Args:
            device (str): Ddevice type.

        Raises:
            TypeError: _description_
        """
        if not isinstance(device, str):
            msg = "'device' must be a string."
            raise TypeError(msg)

        self._device = torch.device(device)

    def use_cpu(self) -> None:
        """Set cpu as device."""
        return self.set_manually("cpu")

    def use_cuda(self) -> None:
        """Set cuda as device.

        Raises:
            ValueError: If cuda is not available.
        """
        if not torch.cuda.is_available():
            return self.set_manually("cuda")
        msg = "CUDA not available."
        raise ValueError(msg)

    def get(self) -> torch.device:
        """Get the device type.

        Returns:
            torch.device: Device type.
        """
        return self._device

    @classmethod
    def set_automatically(cls) -> Self:
        """Automatically sets the device to 'cuda' if possible.

        Returns:
            Self: Device.
        """
        return Device("cuda") if torch.cuda.is_available() else Device("cpu")


class TensorSpecs(TypedDict):
    """Defaults specs."""

    dtype: torch.dtype
    device: torch.device
