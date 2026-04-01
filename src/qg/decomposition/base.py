"""Base classes for space-time decompositions."""

from __future__ import annotations

from qg.logging.core import getLogger

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qg.decomposition.supports.space.base import SpaceSupportFunction
from qg.decomposition.supports.time.base import TimeSupportFunction
from qg.specs import defaults

if TYPE_CHECKING:
    import torch

    from qg.decomposition.coefficients import DecompositionCoefs
TimeSupport = TypeVar("TimeSupport", bound=TimeSupportFunction)
SpaceSupport = TypeVar("SpaceSupport", bound=SpaceSupportFunction)

logger = getLogger(__name__)


class SpaceTimeDecomposition(ABC, Generic[SpaceSupport, TimeSupport]):
    """Space-time decomposition."""

    type: str

    @cached_property
    def order(self) -> int:
        """Decomposition order."""
        return len(self._space.keys())

    def __init__(
        self,
        space_params: dict[int, dict[str, Any]],
        time_params: dict[int, dict[str, Any]],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Instantiate the decomposition.

        Args:
            space_params (dict[int, dict[str, Any]]): Space parameters.
            time_params (dict[int, dict[str, Any]]): Time parameters.
            dtype (torch.dtype | None, optional): Data type.
                Defaults to None.
            device (torch.device | None, optional): Ddevice.
                Defaults to None.
        """
        self._check_validity(space_params, time_params)
        msg = f"Instantiating {self.__class__.__name__}."
        logger.info(msg)

        self._specs = defaults.get(dtype=dtype, device=device)
        self._space = space_params
        self._time = time_params

    def _check_validity(
        self,
        space_params: dict[str, Any],
        time_params: dict[str, Any],
    ) -> None:
        """Check parameters validity."""
        if space_params.keys() != time_params.keys():
            msg = "Mismatching keys between space and time parameters."
            raise ValueError(msg)

    @abstractmethod
    def numel(self) -> int:
        """Total number of elements."""

    @abstractmethod
    def generate_random_coefs(self) -> DecompositionCoefs:
        """Generate random coefficient.

        Useful to properly instantiate coefs.

        Returns:
            DecompositionCoefs: Level -> coefficients.
                ├── 0: coefficents for level 0.
                ├── ...
                └── order-1: coefficient for level order-1
        """

    def set_coefs(self, coefs: DecompositionCoefs) -> None:
        """Set coefficients values.

        To ensure consistent coefficients shapes, best is to use
        self.generate_random_coefs().

        Args:
            coefs (DecompositionCoefs): Level -> coefficients.
                ├── 0: coefficents for level 0.
                ├── ...
                └── order-1: coefficient for level order-1
        """
        self._coefs = coefs

    @abstractmethod
    def generate_time_support(
        self,
        time_params: dict[int, dict[str, Any]],
        space_fields: dict[int, torch.Tensor],
    ) -> TimeSupport:
        """Generate time support.

        Args:
            time_params (dict[int, dict[str, Any]]): Time parameters.
            space_fields (dict[int, torch.Tensor]): Space fields.

        Returns:
            TimeSupport: Gaussian time support.
        """

    def localize(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_dx(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets x-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space_dx(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_dx2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets second order x-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space_dx2(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_dx3(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets third order x-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space_dx3(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_dydx2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets x-x-y derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space_dydx2(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_dy(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space_dy(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_dy2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets second order y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space_dy2(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_dy3(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets third order y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space_dy3(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_dxdy2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets y-y-x derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        space_fields = self._build_space_dxdy2(xx=xx, yy=yy)

        return self.generate_time_support(self._time, space_fields)

    def localize_laplacian(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets second order y-derivative.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        dx2 = self._build_space_dx2(xx=xx, yy=yy)
        dy2 = self._build_space_dy2(xx=xx, yy=yy)
        space_fields = {k: dx2[k] + dy2[k] for k in dx2}

        return self.generate_time_support(self._time, space_fields)

    def localize_dx_laplacian(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets x derivative of laplacian.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        dx3 = self._build_space_dx3(xx=xx, yy=yy)
        dxdy2 = self._build_space_dxdy2(xx=xx, yy=yy)
        space_fields = {k: dx3[k] + dxdy2[k] for k in dx3}

        return self.generate_time_support(self._time, space_fields)

    def localize_dy_laplacian(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        """Localize wavelets x derivative of laplacian.

        Args:
            xx (torch.Tensor): X locations.
            yy (torch.Tensor): Y locations.

        Returns:
            TimeSupport: Time support function.
        """
        dydx2 = self._build_space_dydx2(xx=xx, yy=yy)
        dy3 = self._build_space_dy3(xx=xx, yy=yy)
        space_fields = {k: dydx2[k] + dy3[k] for k in dydx2}

        return self.generate_time_support(self._time, space_fields)

    @abstractmethod
    def _build_space(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport: ...

    def _build_space_dx(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement x-derivative."
        raise NotImplementedError(msg)

    def _build_space_dy(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement y-derivative."
        raise NotImplementedError(msg)

    def _build_space_dx2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement xx-derivative."
        raise NotImplementedError(msg)

    def _build_space_dy2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement yy-derivative."
        raise NotImplementedError(msg)

    def _build_space_dx3(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement xxx-derivative."
        raise NotImplementedError(msg)

    def _build_space_dy3(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement yyy-derivative."
        raise NotImplementedError(msg)

    def _build_space_dydx2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement xxy-derivative."
        raise NotImplementedError(msg)

    def _build_space_dxdy2(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement yyx-derivative."
        raise NotImplementedError(msg)

    def _build_space_laplacian(self, xx: torch.Tensor, yy: torch.Tensor) -> TimeSupport:
        msg = "This decomposition does not implement laplacian."
        raise NotImplementedError(msg)

    def _build_space_dx_laplacian(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> TimeSupport:
        msg = "This decomposition does not implement dx laplacian."
        raise NotImplementedError(msg)

    def _build_space_dy_laplacian(
        self, xx: torch.Tensor, yy: torch.Tensor
    ) -> TimeSupport:
        msg = "This decomposition does not implement dy laplacian."
        raise NotImplementedError(msg)

    def get_params(self) -> dict[str, Any]:
        """Return decomposition params as dict.

        Returns:
            dict[str, Any]: Decomposition params.
        """
        return {"type": self.type, "space": self._space, "time": self._time}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        """Build class from params dict.

        Args:
            params (dict[str, Any]): Decomposition params.

        Returns:
            Self: Instance of class.
        """
        return cls(space_params=params["space"], time_params=params["time"])
