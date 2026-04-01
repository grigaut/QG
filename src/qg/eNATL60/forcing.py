"""Forcing-related tools."""

from pathlib import Path

import numpy as np
import xarray as xr

from qg.eNATL60.var_keys import ATMOS_PRESSURE, LATITUDE, LONGITUDE, TIME


def load_era_interim_oneyear(folder: Path, year: int = 2010) -> xr.Dataset:
    """Load atmospheric fields corresponding to one year of eNATL60 simulation.

    Args:
        folder (Path): Path to folder containing data.
        year (int, optional): Year to load. Defaults to 2010.

    Returns:
        xr.Dataset: Dataset.
    """
    varlist = ["msl_ERAinterim", "u10_DFS5.2", "v10_DFS5.2"]
    chks = {TIME: 1, LONGITUDE: -1, LATITUDE: -1}
    # open all 3 datasets, force coordinates to align (floating point errors)
    # and have same name (lon vs lon0, eg)
    var0_file = folder / f"drowned_{varlist[0]}_y{year}.nc"
    ds = xr.open_dataset(var0_file, chunks=chks)
    ds = ds.rename({"lon": LONGITUDE, "lat": LATITUDE, "MSL": ATMOS_PRESSURE})
    ds[LONGITUDE] = (ds[LONGITUDE] + 180) % 360 - 180  # longitude in [-180, 180]
    dsu = xr.merge(
        [
            xr.open_dataset(folder / f"drowned_{v}_y{year}.nc", chunks=chks)
            for v in varlist[1:]
        ]
    )
    dsu = dsu.rename({"lon0": LONGITUDE, "lat0": LATITUDE})
    dsu[LONGITUDE] = ds[LONGITUDE]
    dsu[LATITUDE] = ds[LATITUDE]
    ds = ds.merge(dsu).sortby(LONGITUDE).sortby(LATITUDE)
    ds[TIME] = xr.date_range(f"{year}-01-01T03:00", periods=365 * 8, freq="3h")
    return ds.set_coords([TIME, LONGITUDE, LATITUDE])


def load_era_interim(folder: Path, *years: int) -> xr.Dataset:
    """Load atmospheric field over multiple years.

    Args:
        folder (Path): Path to folder containing data.
        *years (int): years to load.

    Returns:
        xr.Dataset: Dataset.
    """
    return xr.concat(
        [load_era_interim_oneyear(folder, y) for y in years],
        dim=TIME,
    )


def slice_time(ds_era: xr.Dataset, times: xr.DataArray) -> xr.Dataset:
    """Slice ERA dataset given time range.

    Args:
        ds_era (xr.Dataset): ERA dataset.
        times (xr.DataArray): Timerange to match.

    Returns:
        xr.Dataset: Sliced ERA dataset.
    """
    era_dt = np.timedelta64(3, "h")
    t_min = times.min() - era_dt
    t_max = times.max() + era_dt
    return ds_era.sel(time=slice(t_min, t_max))


def slice_space(
    ds_era: xr.Dataset,
    lons: xr.DataArray,
    lats: xr.DataArray,
) -> xr.Dataset:
    """Perform space-slice of ERA dataset.

    Args:
        ds_era (xr.Dataset): ERA dataset.
        lons (xr.DataArray): Longitudes to match.
        lats (xr.DataArray): Latitudes to match.

    Returns:
        xr.Dataset: Sliced ERA dataset.
    """
    era_dlon = np.abs(
        ds_era[LONGITUDE].values[1:] - ds_era[LONGITUDE].values[:-1]
    ).max()
    lon_min = lons.min() - era_dlon
    lon_max = lons.max() + era_dlon
    era_dlat = np.abs(ds_era[LATITUDE].values[1:] - ds_era[LATITUDE].values[:-1]).max()
    lat_min = lats.min() - era_dlat
    lat_max = lats.max() + era_dlat
    return ds_era.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))


def interpolate_era_da(
    da_era: xr.DataArray,
    ds: xr.Dataset,
) -> xr.DataArray:
    """Interpolate atmospheric variable "da_era" on the grid of "ds".

    Args:
        da_era (xr.DataArray): ERA DataArray.
        ds (xr.Dataset): Dataset to match coordinates.

    Returns:
        xr.DataArray: Interpolated DataArray.
    """
    if TIME in ds.dims and len(ds[TIME]) == 1:  # time before space
        da = da_era.interp({TIME: ds[TIME]})
        return da.interp({LONGITUDE: ds[LONGITUDE], LATITUDE: ds[LATITUDE]})
    da = da_era.interp({LONGITUDE: ds[LONGITUDE], LATITUDE: ds[LATITUDE]})
    return da.interp({TIME: ds[TIME]})
