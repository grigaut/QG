"""Fields computations."""

import xarray as xr

from qg.eNATL60.forcing import interpolate_era_da
from qg.eNATL60.var_keys import (
    ATMOS_PRESSURE,
    SSH,
    TIME,
)


def compute_streamfunction_with_atmospheric_pressure(
    ds: xr.Dataset,
    ds_era: xr.Dataset,
    rho0: float = 1026.0,
    g: float = 9.81,
    *,
    remove_avgs: bool = False,
) -> xr.DataArray:
    """Compute surface streamfunction.

    Args:
        ds (xr.Dataset): Sea surface height.
        ds_era (xr.Dataset): Atmospheric datas.
        rho0 (float, optional): Sea density. Defaults to 1026.0.
        g (float, optional): Gravity constant. Defaults to 9.81.
        remove_avgs (bool, optional): Whether to remove atmospheric pressure
            average (spatial average evaluated at every time) and ssh average
            (both in time and space) or not. Defauts to False

    Returns:
        xr.DataArray: _description_
    """
    atmos_p = interpolate_era_da(ds_era[ATMOS_PRESSURE], ds)
    ssh = ds[SSH]
    if remove_avgs:
        atmos_p_avg = atmos_p.mean(dim=[d for d in atmos_p.dims if d != TIME])
        atmos_p = atmos_p - atmos_p_avg
        ssh_avg = ssh.mean()
        ssh = ssh - ssh_avg
    return atmos_p / rho0 + g * ssh


def compute_stream_function_ssh_only(
    ds: xr.Dataset,
    g: float = 9.81,
    *,
    remove_avg: bool = False,
) -> xr.DataArray:
    """Compute surface streamfunction.

    Args:
        ds (xr.Dataset): Sea surface height.
        ds_era (xr.Dataset): Atmospheric datas.
        rho0 (float, optional): Sea density. Defaults to 1026.0.
        g (float, optional): Gravity constant. Defaults to 9.81.
        remove_avg (bool, optional): Whether to remove ssh average or not.
            Defauts to False

    Returns:
        xr.DataArray: _description_
    """
    ssh = ds[SSH]
    if remove_avg:
        ssh_avg = ssh.mean()
        ssh = ssh - ssh_avg
    return g * ssh
