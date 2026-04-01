"""Data interpolation tools."""

import numpy as np
import xarray as xr
import xesmf as xe

from qg.eNATL60.var_keys import LATITUDE, LONGITUDE
from qg.physics.constants import EARTH_RADIUS


def compute_lonlat_bounds(
    lons: xr.DataArray, lats: xr.DataArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute longitude and latitude bounds.

    Args:
        lons (xr.DataArray): Longitudes (2D).
        lats (xr.DataArray): Latitiudes (2D).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: lon_min,
            lon_max, lat_min, lat_max
    """
    lon_min = np.deg2rad(lons[0, :].max().values)
    lon_max = np.deg2rad(lons[-1, :].min().values)
    lat_min = np.deg2rad(lats[:, 0].max().values)
    lat_max = np.deg2rad(lats[:, -1].min().values)
    return (lon_min, lon_max, lat_min, lat_max)


def compute_lonlat_from_regular_xy_grid(
    lons: xr.DataArray,
    lats: xr.DataArray,
    dx: float = 10000,
    dy: float = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute longitude / latitude grids with regular cartesian spacing.

    Args:
        lons (xr.DataArray): Longitude values.
        lats (xr.DataArray): Latitude values.
        dx (float, optional): Spacing along x in cartesian coordinates.
            Defaults to 10000.
        dy (float, optional): Spacing along y in cartesian coordinates.
            Defaults to 10000.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    lon_min, lon_max, lat_min, lat_max = compute_lonlat_bounds(lons, lats)

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    dlat = dy / EARTH_RADIUS

    nlat = lat_range // dlat

    lat_offset = (lat_range - nlat * dlat) / 2

    lats_ = np.arange(
        lat_min + lat_offset,
        lat_max,
        dlat,
    )

    dlons = dx / EARTH_RADIUS / np.cos(lats_)
    dlon_max = np.max(dlons)
    nlon = lon_range // dlon_max
    dlon_offsets = (lon_range - nlon * dlons) / 2

    lons_ = (
        np.arange(0, nlon + 1).reshape((-1, 1)) * dlons.reshape((1, -1))
        + dlon_offsets.reshape((1, -1))
        + lon_min
    )
    return lons_, np.tile(lats_.reshape(1, -1), (int(nlon) + 1, 1))


def lonlat_to_xy(lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat grid to xy grid.

    Args:
        lons (np.ndarray): Longitudes.
        lats (np.ndarray): Latitudes.

    Returns:
        tuple[np.ndarray, np.ndarray]: Xs, Ys
    """
    lat0 = (lats.max() + lats.min()) / 2

    ys = EARTH_RADIUS * (lats - lat0)

    xs = EARTH_RADIUS * lons * np.cos(lats)

    return xs, ys


def build_regridder(
    ds: xr.Dataset,
    lons: np.ndarray,
    lats: np.ndarray,
) -> xe.Regridder:
    """Build regridder for interpolation.

    Args:
        ds (xr.Dataset): Dataset with reference lon / lat.
        lons (np.ndarray): Longitudes to interpolate onto.
        lats (np.ndarray): Latitudes to interpolate onto.

    Returns:
        xe.Regridder: Regridder.
    """
    ds_out = xr.Dataset(
        {
            "lon": (["i", "j"], np.rad2deg(lons)),
            "lat": (["i", "j"], np.rad2deg(lats)),
        }
    )
    if (len(ds[LONGITUDE].shape) == 1) and (len(ds[LATITUDE].shape) == 1):
        lons_, lats_ = xr.broadcast(ds[LONGITUDE], ds[LATITUDE])
        lons_ref = np.ascontiguousarray(lons_.T).T
        lats_ref = np.ascontiguousarray(lats_.T).T
    elif (len(ds[LONGITUDE].shape) == 2) and (len(ds[LATITUDE].shape) == 2):
        lons_ref = np.ascontiguousarray(ds[LONGITUDE].T).T
        lats_ref = np.ascontiguousarray(ds[LATITUDE].T).T

    else:
        msg = "Uncompatible lon/lat shapes."
        raise ValueError(msg)
    ds_in = xr.Dataset(
        {
            "lon": (["i", "j"], lons_ref),
            "lat": (["i", "j"], lats_ref),
        }
    )
    return xe.Regridder(
        ds_in,
        ds_out,
        "bilinear",
        periodic=False,
    )
