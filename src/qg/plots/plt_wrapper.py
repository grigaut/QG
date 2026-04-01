"""Wrapper for some of matplotlib.pyplot's functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import xarray

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.text import Annotation, Text


DEFAULT_CMAP = "RdBu_r"


def retrieve_imshow_data(
    data: torch.Tensor | np.ndarray | xarray.DataArray,
) -> np.ndarray:
    """Retrieve data for imshow plot.

    Args:
        data (torch.Tensor | np.ndarray): Original data.

    Returns:
        np.ndarray: Data as a numpy array, transposed for imshow compatibility.
    """
    if isinstance(data, torch.Tensor):
        if data.dtype == torch.bool:
            data = data.to(torch.int8)
        data = data.detach().cpu().numpy()
    elif isinstance(data, xarray.DataArray):
        data = data.to_numpy()
    return data.T


def default_clim(
    data: np.ndarray, *, robust: bool = False
) -> tuple[float, float]:
    """Compute default colorbar limit values.

    Args:
        data (np.ndarray): Data for which to compute colorbar limits.
        robust (bool, optional): If True, vmax will be computed from the
            0.98-th quantile of the absolute value of the data,
            and from its from the max of its absolute value if False.
            Defaults to False.
        robust

    Returns:
        tuple[float, float]: Colorbar limits as (vmin, vmax).
    """
    data_ = data[~np.isnan(data)]
    vmax = (
        np.quantile(np.abs(data_), 0.98) if robust else np.max(np.abs(data_))
    )
    return -vmax, vmax


class CbarKwargs(TypedDict, total=False):
    """Non-exhaustives kwargs for plt.colorbar."""

    extendrect: bool


def retrieve_colorbar(
    im: AxesImage, ax: Axes, **kwargs: Unpack[CbarKwargs]
) -> Colorbar:
    """Retrieve colorbar axes from axes.

    Args:
        im (AxesImage): Image to associate colorbar with.
        ax (Axes): Axes to retrieve colorbar from.
        **kwargs: optional arguments to pass to plt.colorbar.

    Returns:
        Axes: Colorbar axes.
    """
    try:
        return ax.cbar
    except AttributeError:
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad="3%")
        ax.cbar = ax.figure.colorbar(im, cax=cax, **kwargs)
        return ax.cbar


class ImshowKwargs(TypedDict, total=False):
    """Non-exhaustives kwargs for imshow."""

    cmap: str
    vmin: float
    vmax: float
    cbar_kwargs: CbarKwargs


def _imshow(
    data: np.ndarray,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_cbar: bool = True,
    **kwargs: Unpack[ImshowKwargs],
) -> AxesImage:
    """Wrapper for plt.imshow with not centered colorbar.

    Args:
        data (torch.Tensor | np.ndarray | xarray.DataArray): 2D array to plot.
        ax (Axes | None, optional): Axes to plot on. Defaults to None.
        title (str | None, optional): Title. Defaults to None.
        show_cbar (bool): Whether to show colorbar or not.
        **kwargs: optional arguments to pass to plt.imshow.
    """
    kwargs.setdefault("cmap", DEFAULT_CMAP)
    kwargs.setdefault("origin", "lower")
    cbar_kwargs = kwargs.pop("cbar_kwargs", {})
    if ax is None:
        ax = plt.subplot()

    im = ax.imshow(data, **kwargs)
    if show_cbar:
        cbar = retrieve_colorbar(im, ax, **cbar_kwargs)
        cbar.update_normal(im)
    if title is not None:
        ax.set_title(title)
    return im


def imshow(
    data: torch.Tensor | np.ndarray | xarray.DataArray,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_cbar: bool = True,
    robust: bool = False,
    **kwargs: Unpack[ImshowKwargs],
) -> AxesImage:
    """Wrapper for plt.imshow.

    Args:
        data (torch.Tensor | np.ndarray | xarray.DataArray): 2D array to plot.
        ax (Axes | None, optional): Axes to plot on. Defaults to None.
        title (str | None, optional): Title. Defaults to None.
        show_cbar (bool, optional): Whether to show colorbar or not.
            Defaults to True.
        robust (bool, optional): If True, vmax will be computed from the
            0.98-th quantile of the absolute value of the data,
            and from its from the max of its absolute value if False.
            Defaults to False.
        **kwargs: optional arguments to pass to plt.imshow.
    """
    data = retrieve_imshow_data(data)
    vmin, vmax = default_clim(data, robust=robust)
    kwargs.setdefault("vmax", vmax)
    kwargs.setdefault("vmin", vmin)
    return _imshow(
        data=data,
        ax=ax,
        title=title,
        show_cbar=show_cbar,
        **kwargs,
    )


def imshow_(
    data: torch.Tensor | np.ndarray | xarray.DataArray,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_cbar: bool = True,
    **kwargs: Unpack[ImshowKwargs],
) -> AxesImage:
    """Wrapper for plt.imshow with not centered colorbar.

    Args:
        data (torch.Tensor | np.ndarray | xarray.DataArray): 2D array to plot.
        ax (Axes | None, optional): Axes to plot on. Defaults to None.
        title (str | None, optional): Title. Defaults to None.
        show_cbar (bool): Whether to show colorbar or not.
        **kwargs: optional arguments to pass to plt.imshow.
    """
    data = retrieve_imshow_data(data)
    return _imshow(
        data=data,
        ax=ax,
        title=title,
        show_cbar=show_cbar,
        **kwargs,
    )


class SubplotsKwargs(TypedDict, total=False):
    """Non-exhaustives kwargs for subplots."""

    figsize: tuple[int, ...]


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Unpack[SubplotsKwargs],
) -> tuple[Figure, np.ndarray]:
    """Wrapper for plt.subplots.

    Args:
        nrows (int, optional): Number of rows. Defaults to 1.
        ncols (int, optional): Number of columns. Defaults to 1.
        **kwargs: optional arguments to pass to plt.subplots.

    Returns:
        tuple[mpl.Figure, np.ndarray]: Figure, Axes array.
    """
    kwargs.setdefault("squeeze", False)
    kwargs.setdefault("figsize", ((4 * ncols, 4 * nrows + 1)))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    fig.subplots_adjust(wspace=0.3)
    return fig, axs


class ShowKwargs(TypedDict, total=False):
    """Non-exhaustives kwargs for show."""


def show(*, tight_layout: bool = True, **kwargs: Unpack[ShowKwargs]) -> None:
    """Wrapper for plt.show."""
    if tight_layout:
        plt.tight_layout()
    return plt.show(**kwargs)


class AnnotateKwargs(TypedDict, total=False):
    """Non-exhaustives kwargs for show."""


def set_coltitles(
    colnames: list[str],
    axs: np.ndarray,
    *,
    pad: int = 5,
    **kwargs: Unpack[AnnotateKwargs],
) -> list[Annotation]:
    """Set column titles.

    Args:
        colnames (list[str]): Column names.
        axs (np.ndarray): Axes array.
        pad (int, optional): Padding below text. Defaults to 5.
        **kwargs: Keywords arguments to pass to ax.annotate

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
    col_titles: list[Annotation] = []
    for ax, col in zip(axs[0], colnames):
        ax: Axes
        col_titles.append(ax.annotate(col, xytext=(0, pad), **kwargs))
    return col_titles


def set_rowtitles(
    rownames: list[str],
    axs: np.ndarray,
    *,
    pad: int = 5,
    **kwargs: Unpack[AnnotateKwargs],
) -> list[Annotation]:
    """Set row titles.

    Args:
        rownames (list[str]): Row names.
        axs (np.ndarray): Axes array.
        pad (int, optional): Padding below text. Defaults to 5.
        **kwargs: Keywords arguments to pass to ax.annotate

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
    row_titles: list[Annotation] = []
    for ax, row in zip(axs[:, 0], rownames):
        ax: Axes
        row_titles.append(
            ax.annotate(
                row,
                xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label,
                **kwargs,
            )
        )
    return row_titles


def clamp_ylims(bottom: float, top: float, ax: Axes) -> None:
    """Clamp y lims.

    Args:
        bottom (float): Bottom value.
        top (float): Top value.
        ax (Axes): Axes.
    """
    ax.relim()
    ax.autoscale_view()

    _, my = ax.margins()

    y0 = max(bottom, ax.dataLim.ymin)
    y1 = min(top, ax.dataLim.ymax)

    dy = y1 - y0
    pad = my * dy

    ax.set_ylim(y0 - pad, y1 + pad)


def set_ylims(bottom: float, top: float, ax: Axes) -> None:
    """Set y lims.

    Args:
        bottom (float): Bottom value.
        top (float): Top value.
        ax (Axes): Axes.
    """
    ax.relim()
    ax.autoscale_view()

    _, my = ax.margins()

    y0 = bottom
    y1 = top

    dy = y1 - y0
    pad = my * dy

    ax.set_ylim(y0 - pad, y1 + pad)


class SuptitleKwargs(TypedDict, total=False):
    """Non-exhaustives kwargs for suptitle."""


def blittable_suptitle(
    text: str,
    fig: Figure,
    ax: Axes,
    **kwargs: Unpack[SuptitleKwargs],
) -> Text:
    """Set the figure suptitle through ax.text.

    This allow the returned text to be modified in animation
    with blit = True.

    Args:
        text (str): Suptitle text.
        fig (Figure): Figure to add suptitle to.
        ax (Axes): Ax to use for the suptitle.
        **kwargs: Keywords arguments to pass to ax.text.

    Returns:
        Text: Suptitle text.
    """
    temp_suptitle = fig.suptitle(text, **kwargs)

    # Get ALL text properties using get_fontproperties() and other getters
    position = temp_suptitle.get_position()
    font_properties = temp_suptitle.get_fontproperties()
    color = temp_suptitle.get_color()
    ha = temp_suptitle.get_ha()
    va = temp_suptitle.get_va()
    alpha = temp_suptitle.get_alpha()
    rotation = temp_suptitle.get_rotation()
    bbox = temp_suptitle.get_bbox_patch()

    # Remove the suptitle
    temp_suptitle.remove()
    fig._suptitle = None  # noqa: SLF001

    suptitle = ax.text(
        position[0],
        position[1],
        text,
        ha=ha,
        va=va,
        fontproperties=font_properties,
        color=color,
        alpha=alpha,
        rotation=rotation,
        transform=fig.transFigure,
    )
    if bbox:
        suptitle.set_bbox(
            {
                "boxstyle": bbox.get_boxstyle(),
                "facecolor": bbox.get_facecolor(),
                "edgecolor": bbox.get_edgecolor(),
                "alpha": bbox.get_alpha(),
            }
        )
    return suptitle


def close(fig: int | str | Figure | None = None) -> None:
    """Close a figure window, and unregister it from pyplot.

    Args:
        fig (int | str | Figure | None, optional): Figure to close.
            Defaults to None.
    """
    plt.close(fig)
