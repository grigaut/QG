"""eNATL60 data loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def open_dataset(filepath: Path) -> xr.Dataset:
    """Open a xarray dataset from a single file.

    Args:
        filepath (Path): Filepath to read the dataset from.

    Raises:
        ValueError: If the dataset suffix is not recognized.

    Returns:
        xr.Dataset: Dataset.
    """
    if filepath.suffix == ".zarr":
        return xr.open_zarr(filepath)
    if filepath.suffix == ".nc":
        return xr.open_dataset(filepath)

    msg = "Unrecognised filepath type."
    raise ValueError(msg)


def open_datasets(
    *filepaths: Path, concat_dim: str = "time_counter"
) -> xr.Dataset:
    """Open multiple datasets and combine them into a single one.

    Args:
        *filepaths (Path): Filepaths.
        concat_dim (str, optional): Dimension along which to concatenate.
            Defaults to "time_counter".

    Returns:
        xr.Dataset: Dataset.
    """
    return xr.combine_nested(
        [open_dataset(f) for f in filepaths],
        concat_dim=concat_dim,
    ).sortby(concat_dim)


def format_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Format a dataset.

    Args:
        ds (xr.Dataset): Dataset to format.

    Returns:
        xr.Dataset: Formatted dataset.
    """
    ds = ds.set_coords(["nav_lat", "nav_lon"])
    if "axis_nbounds" in ds.dims:
        ds = ds.drop_dims("axis_nbounds")
    if "time_centered" in ds.coords:
        ds = ds.reset_coords("time_centered", drop=True)
    return ds


def sort_files_by_dates(*filepaths: Path) -> np.ndarray:
    """Sort file names by dates."""
    times = retrieve_dates(*filepaths).to_numpy()
    args = np.argsort(times)
    return np.array(filepaths)[args]  # .tolist()


def retrieve_dates(*filepaths: Path) -> pd.DatetimeIndex:
    """Parse files dates.

    Returns:
        list[int]: List of dates from the filepaths.
    """
    return pd.to_datetime([f.name[-20:-12] for f in filepaths])  # .to_numpy()


def load_datasets(
    *filepaths: Path,
    concat_dim: str = "time_counter",
    format_func: Callable[[xr.Dataset], xr.Dataset] = format_dataset,
) -> xr.Dataset:
    """Load (open and format) a dataset.

    Args:
        *filepaths (Path): Filepaths.
        concat_dim (str, optional): Dimension along which to concatenate
            datasets. Defaults to "time_counter".
        format_func (Callable[[xr.Dataset],xr.Dataset]): formating function.

    Returns:
        xr.Dataset: Dataset.
    """
    ds = open_datasets(*filepaths, concat_dim=concat_dim)
    return format_func(ds)
