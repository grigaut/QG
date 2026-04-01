"""Utils for dictionaries with tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from qg.specs import defaults

if TYPE_CHECKING:
    from collections.abc import Callable


def iterate_over_dict(
    d: dict[str, Any], f: Callable[[torch.Tensor], Any]
) -> dict[str, Any]:
    """Iterate tensor operations on the tensors of a given dictionary.

    Args:
        d (dict[str, Any]): Dictionary.
        f (Callable[[torch.Tensor], Any]): Function to apply
            to tensors.

    Returns:
        dict[str, Any]: Resulting dictionary.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out[k] = f(v)
        elif isinstance(v, dict):
            out[k] = iterate_over_dict(v, f)
        else:
            out[k] = v
    return out


def change_specs(
    d: dict[str, Any],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Change specs of tensors in dictionnary.

    Args:
        d (dict[str, Any]): Dictionary to modify (inplace).
        dtype (torch.dtype | None, optional): Data type. Defaults to None.
        device (torch.device | None, optional): Device. Defaults to None.

    Returns:
        dict[str, Any]: Modified dictionary.
    """
    specs = defaults.get(dtype=dtype, device=device)
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(**specs)
        elif isinstance(v, dict):
            d[k] = change_specs(v, **specs)
    return d
