"""Class for decomposition coefficients."""

from collections.abc import ItemsView, Iterator, KeysView, ValuesView

import torch

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


class DecompositionCoefs:
    """Decomposition coefficients."""

    def __init__(self, *coefs: torch.Tensor) -> None:
        """Instantiate the decomposition coefs.

        Args:
            *coefs (torch.Tensor): Tensors to use as coefs.
        """
        self._coefs = dict(enumerate(coefs))

    def numel(self) -> int:
        """Number of coefficients."""
        tot = 0
        for c in self._coefs.values():
            tot += c.numel()
        return tot

    def __repr__(self) -> str:
        """String representation."""
        text = "Decomposition coefficients"
        for lvl, c in self._coefs.items():
            text += f"\n\tLevel {lvl}: {c.shape}-shaped coefficients"
        return text

    def __getitem__(self, index: int) -> torch.Tensor:
        """Implement __getitem__."""
        return self._coefs[index]

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Use dict's __iter__ method."""
        return iter(self._coefs)

    def items(self) -> ItemsView[int, torch.Tensor]:
        """Return coefs items."""
        return self._coefs.items()

    def values(self) -> ValuesView[torch.Tensor]:
        """Return coefs values."""
        return self._coefs.values()

    def keys(self) -> KeysView[int]:
        """Return coefs keys."""
        return self._coefs.keys()

    def scale(self, *scales: float) -> "DecompositionCoefs":
        """Scale coeffcients.

        Args:
            *scales (float): Values to use for scaling.
        """
        coefs = {k: v * scales[k] for k, v in self._coefs.items()}
        return DecompositionCoefs.from_dict(coefs)

    def to_dict(self) -> dict[int, torch.Tensor]:
        """Convert to dictionary.

        Returns:
            dict[int, torch.Tensor]: Level -> Coef.
        """
        return self._coefs

    def requires_grad_(
        self, *, requires_grad: bool = True
    ) -> "DecompositionCoefs":
        """Call requires_grad_ an coefs.

        Args:
            requires_grad (bool, optional): If autograd should record
                operations on this tensor.. Defaults to True.

        Returns:
            DecompositionCoefs: Coefs.
        """
        coefs = {
            k: v.requires_grad_(requires_grad) for k, v in self._coefs.items()
        }
        return DecompositionCoefs.from_dict(coefs)

    @classmethod
    def from_dict(cls, coefs_dict: dict[int, torch.Tensor]) -> Self:
        """Instantiate coefs from dictionary.

        Args:
            coefs_dict (dict[int, torch.Tensor]): Dictionary coef.

        Returns:
            Self: DecompositionCoefs.
        """
        coefs = [coefs_dict[k] for k in sorted(coefs_dict.keys())]
        return cls(*coefs)

    @classmethod
    def zeros_like(cls, coefs: "DecompositionCoefs") -> Self:
        """Set all coefs to 0.

        Args:
            coefs (DecompositionCoefs): Decompositon coefs.

        Returns:
            Self: DecompositionCoefs.
        """
        coefs_zeros = {k: torch.zeros_like(v) for k, v in coefs.items()}

        return cls.from_dict(coefs_zeros)

    @classmethod
    def ones_like(cls, coefs: "DecompositionCoefs") -> Self:
        """Set all coefs to 1.

        Args:
            coefs (DecompositionCoefs): Decompositon coefs.

        Returns:
            Self: DecompositionCoefs.
        """
        coefs_ones = {k: torch.ones_like(v) for k, v in coefs.items()}

        return cls.from_dict(coefs_ones)
