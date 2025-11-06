"""Wrapper for some of matplotlib.pyplot's functions."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import figure
from matplotlib import pyplot as plt
from typing_extensions import ParamSpec

Param = ParamSpec("Param")

DEFAULT_CMAP = "RdBu_r"


def imshow(
    data: torch.Tensor | np.ndarray,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    **kwargs: Param.kwargs,
) -> None:
    """Wrapper for plt.imshow.

    Args:
        data (torch.Tensor | np.ndarray): 2D array to plot.
        ax (plt.Axes | None, optional): Axes to plot on. Defaults to None.
        title (str | None, optional): Title. Defaults to None.
        **kwargs: optional arguments to pass to plt.imshow.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    kwargs.setdefault("vmax", np.max(np.abs(data)))
    kwargs.setdefault("vmin", -kwargs["vmax"])
    kwargs.setdefault("cmap", DEFAULT_CMAP)
    kwargs.setdefault("origin", "lower")
    if ax is None:
        ax = plt.subplot()

    cbar = ax.imshow(data.T, **kwargs)
    ax.figure.colorbar(cbar, ax=ax)
    if title is not None:
        ax.set_title(title)


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Param.kwargs,
) -> tuple[figure.Figure, np.ndarray]:
    """Wrapper for plt.subplots.

    Args:
        nrows (int, optional): Number of rows. Defaults to 1.
        ncols (int, optional): Number of columns. Defaults to 1.
        **kwargs: optional arguments to pass to plt.subplots.

    Returns:
        tuple[mpl.figure.Figure, np.ndarray]: Figure, Axes array.
    """
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("constrained_layout", True)
    kwargs.setdefault("figsize", ((4 * ncols, 4 * nrows + 1)))
    return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)


def show(**kwargs: Param.kwargs) -> None:
    """Wrapper for plt.show."""
    return plt.show(**kwargs)


def set_coltitles(
    colnames: list[str],
    axs: np.ndarray,
    *,
    pad: int = 5,
    **kwargs: Param.kwargs,
) -> None:
    """Set column titles.

    Args:
        colnames (list[str]): Column names.
        axs (np.ndarray): Axes array.
        pad (int, optional): Padding below text. Defaults to 5.
        **kwargs (Param.kwargs): Keywords arguments to pass to ax.annotate

    Raises:
        ValueError: If length mismatch between colnames and axs.
    """
    if len(colnames) != (n := axs.shape[1]):
        msg = f"There must be exactly {n} column names."
        raise ValueError(msg)
    kwargs.setdefault("xy", (0.5, 1))
    kwargs.setdefault("xycoords", "axes fraction")
    kwargs.setdefault("textcoords", "offset points")
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "baseline")

    for ax, col in zip(axs[0], colnames):
        ax.annotate(col, xytext=(0, pad), **kwargs)


def set_rowtitles(
    rownames: list[str],
    axs: np.ndarray,
    *,
    pad: int = 5,
    **kwargs: Param.kwargs,
) -> None:
    """Set row titles.

    Args:
        rownames (list[str]): Row names.
        axs (np.ndarray): Axes array.
        pad (int, optional): Padding below text. Defaults to 5.
        **kwargs (Param.kwargs): Keywords arguments to pass to ax.annotate

    Raises:
        ValueError: If length mismatch between rownames and axs.
    """
    if len(rownames) != (n := axs.shape[0]):
        msg = f"There must be exactly {n} row names."
        raise ValueError(msg)

    kwargs.setdefault("xy", (0, 0.5))
    kwargs.setdefault("textcoords", "offset points")
    kwargs.setdefault("ha", "right")
    kwargs.setdefault("va", "center")
    kwargs.setdefault("rotation", 90)

    for ax, row in zip(axs[:, 0], rownames):
        ax.annotate(
            row,
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            **kwargs,
        )
