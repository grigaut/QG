"""
Double-gyre on regular domain.
"""

from math import sqrt
from pathlib import Path
import shutil
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import xarray as xr
from qg.constants import EARTH_ANGULAR_ROTATION, EARTH_RADIUS
from qg.decomposition.coefficients import DecompositionCoefs
from qg.decomposition.wavelets.core import WaveletBasis
from qg.decomposition.wavelets.param_generators import dyadic_decomposition
from qg.eNATL60 import seasons
from qg.eNATL60.fields_computations import (
    compute_streamfunction_with_atmospheric_pressure,
)
from qg.eNATL60.forcing import load_era_interim, slice_space, slice_time
from qg.eNATL60.interpolation import (
    build_regridder,
    compute_lonlat_from_regular_xy_grid,
    lonlat_to_xy,
)
from qg.eNATL60.loading import load_datasets, retrieve_dates, sort_files_by_dates
from qg.eNATL60.var_keys import LATITUDE, LONGITUDE, SSH, STREAMFUNCTION, TIME
from qg.forced import Forced
from qg.logging import setup_root_logger, getLogger
from qg.cli import ScriptArgs
from qg.config import (
    load_model_config,
    load_optimization_config,
    load_output_config,
    load_regularization_config,
    load_season_config,
    load_simulation_config,
)
from qg.interpolation import QuadraticInterpolation
from qg.logging.utils import box, sec2text, step
from qg.observations.satellite_track import SatelliteTrackMask
from qg.optim.callbacks import LRChangeCallback
from qg.optim.utils import EarlyStop, RegisterParams
from qg.pv import compute_q1_interior
from qg.solver.boundary_conditions.base import Boundaries
from qg.space import compute_xy_q
from qg.specs import defaults
from qg.utils.cropping import crop
from qg.utils.storage import get_path_from_env

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

args = ScriptArgs.from_cli(config_default=Path("configs/va_enatl60_forced_summer.toml"))
specs = defaults.get()

setup_root_logger(args.verbose)
logger = getLogger(__name__)

# Configuratiosn

## Model
config = load_model_config(args.config)

sigma_bc = 16
sigma_ic = 16


## Areas
n_ens = config["n_ens"]
nx = config["xv"].shape[0] - 1
ny = config["yv"].shape[0] - 1
yv = config["yv"]
xv = config["xv"]

Lx = xv[-1] - xv[0]
Ly = yv[-1] - yv[0]

dx = Lx / nx
dy = Ly / ny
dt = config["dt"]
n_year = int(365 * 24 * 3600 / dt)

## Load eNATL60 grid
n_file_per_cycle = 20

### Data folder

data_folder = get_path_from_env(key="eNATL60_FOLDER")
files = list((data_folder / "MEANDERS" / "gridT").glob("*.nc"))

files = sort_files_by_dates(*files)

season = {
    "summer": seasons.SUMMER,
    "autumn": seasons.AUTUMN,
    "winter": seasons.WINTER,
    "spring": seasons.SPRING,
}
season_value = load_season_config(args.config)["season"]
in_season = retrieve_dates(*files.tolist()).month.isin(season[season_value])
if ((in_season[1:]) & (~in_season[:-1])).sum() + int(in_season[0]) > 1:
    msg = "Non-time-contiguous data for this season in provided dataset."
    raise ValueError(msg)
files = files[in_season]


def format_ds(ds: xr.Dataset) -> xr.Dataset:
    """Format Dataset."""
    # Drop useless variables
    if "axis_nbounds" in ds.dims:
        ds = ds.drop_dims("axis_nbounds")
    if "time_centered" in ds.coords:
        ds = ds.reset_coords("time_centered", drop=True)
    # Rename
    ds = ds.rename(
        {
            "time_counter": TIME,
            "nav_lon": LONGITUDE,
            "nav_lat": LATITUDE,
            "x": "i",
            "y": "j",
            "sossheig": SSH,
        }
    )
    ds = ds.transpose(TIME, "i", "j")
    return ds.set_coords([LONGITUDE, LATITUDE])


### Load only one file to access grid informations

ds = load_datasets(files[0], format_func=format_ds)

### Compute longitude / latitudes

lons, lats = compute_lonlat_from_regular_xy_grid(
    ds[LONGITUDE],
    ds[LATITUDE],
    dx=dx.item(),
    dy=dy.item(),
)
xs, ys = lonlat_to_xy(lons, lats)

### Compute β-plane parameters

lat0 = (lats.max() + lats.min()) / 2

f0 = 2 * EARTH_ANGULAR_ROTATION * np.sin(lat0)
beta = 2 * EARTH_ANGULAR_ROTATION * np.cos(lat0) / EARTH_RADIUS

### Build regridder

psi_regridder = build_regridder(ds, lons, lats)


msg_model = "Variational assimilation of eNATL60 data."

## Simulation
sim_config = load_simulation_config(args.config)
n_steps = sim_config["duration"]

duration = sec2text(sim_config["duration"] * dt)
msg_sim = f"Cycle duration: {duration} ({n_steps} steps)"

## Optimization
optim_config = load_optimization_config(args.config)
n_optim = optim_config["optimization_steps"]
n_cycles = optim_config["cycles"]
comparison_interval = optim_config["comparison_interval"]

separation = int(dt * optim_config["separation"] / 3600 / 24)

msg_optim = (
    f"Performing {n_cycles} cycles with up to {n_optim} optimization steps.\n"
    f"Loss will be evaluated every {sec2text(comparison_interval * dt)}."
)

if optim_config["separation"] != 0:
    msg_sim += f"\nCycles are separated by {sec2text(optim_config['separation'] * dt)}."

## Regularization
reg_config = load_regularization_config(args.config)
with_reg = reg_config["gamma"] is not None

gamma = reg_config["gamma"] / comparison_interval

if with_reg:
    msg_reg = f"Using ɣ = {gamma:#8.3g} to weight regularization"  # noqa: RUF001
    if gamma != reg_config["gamma"]:
        msg_reg += (
            f" (rescaled from ɣ = {reg_config['gamma']:#5.3g} to"  # noqa: RUF001
            " account for observations sparsity)."
        )
    else:
        msg_reg += "."
else:
    msg_reg = "No regularization."

## Ouput
output_config = load_output_config(args.config)

prefix = output_config["prefix"]
filename = f"{prefix}.pt"
folder: Path = output_config["folder"]

if folder.is_dir():
    msg = f"{folder} already exists and will be overidden."
    logger.warning(msg)
    shutil.rmtree(folder, ignore_errors=True)
else:
    folder.mkdir()

    gitignore = folder.joinpath(".gitignore")
    with gitignore.open("w") as file:
        file.write("*")

output_file: Path = folder.joinpath(filename)

msg_output = f"Outputs will be save at {output_file}"

## Observations

X, Y = torch.meshgrid(xv, yv, indexing="ij")

bc = 4
xx, yy = crop(X, bc), crop(Y, bc)

obs_mask = SatelliteTrackMask(
    xx,
    yy,
    track_width=100000,
    track_interval=600000,
    theta=torch.pi / 12,
    full_coverage_time=20 * 3600 * 24,
)
if comparison_interval != 1:
    msg = "Using Satellite track, comparison interval inferred from tracks trajectory."
    logger.warning(box(msg, style="="))
n_obs = obs_mask.compute_obs_nb(n_steps, dt)
msg_obs = f"Surface observed along satellite tracks, {n_obs} pixels observed."

logger.info(
    box(
        msg_model,
        msg_sim,
        msg_obs,
        msg_optim,
        msg_reg,
        msg_output,
        style="=",
    )
)

msg = f"Running code using {specs['device']}"
logger.info(msg)

Ly = yv[-1] - yv[0]


# Width for boundaries


config_model = {
    "xv": config["xv"][bc:-bc],
    "yv": config["yv"][bc:-bc],
    "n_ens": config["n_ens"],
    "mask": config["mask"][bc:-bc, bc:-bc],
    "flux_stencil": config["flux_stencil"],
    "H": config["H"][:1] * config["H"][1:2] / (config["H"][:1] + config["H"][1:2]),
    "g_prime": config["g_prime"][1:2],
    "f0": f0,
    "beta": beta,
    "bottom_drag_coef": 0,
    "device": specs["device"],
    "dt": config["dt"],  # time-step (s)
}

H1, H2 = config["H"][0, 0, 0], config["H"][1, 0, 0]
g1, g2 = config["g_prime"][0, 0, 0], config["g_prime"][1, 0, 0]


qg = Forced(config_model)
L: float = dx
beta_effect_w = config["beta"] * ((yv[:-1] + yv[1:]) / 2 - qg.y0)


def update_loss(
    loss: torch.Tensor,
    f: torch.Tensor,
    f_ref: torch.Tensor,
    time: torch.Tensor,
    *,
    variance: float | torch.Tensor = 1,
) -> torch.Tensor:
    """Update loss."""
    mask = obs_mask.at_time(time)
    if not mask.any():
        return loss
    f_sliced = f.flatten()[mask.flatten()]
    f_ref_sliced = f_ref.flatten()[mask.flatten()]
    return loss + (f_sliced - f_ref_sliced).square().sum() / variance


# PV computation


def compute_q_rg(psi1: torch.Tensor) -> torch.Tensor:
    return compute_q1_interior(
        psi1,
        torch.zeros_like(psi1),
        1 / H1 / H2 * (H1 + H2) / g2,
        0,
        dx,
        dy,
        config["f0"],
        beta_effect_w[..., 1:-1],
    )


outputs = []
for c in range(n_cycles):
    torch.cuda.reset_peak_memory_stats()

    start_cycle = c * n_file_per_cycle + c * separation
    end_cycle = (c + 1) * n_file_per_cycle + c * separation

    if end_cycle > len(files):
        msg = f"Not enough files to perform cycle {c} and above."
        logger.warning(msg)
        break

    files_for_cycle = files[start_cycle:end_cycle]

    ds = load_datasets(*files_for_cycle, format_func=format_ds)

    msg = f"Cycle {step(c + 1, n_cycles)}: eNATL60 data loaded."
    logger.info(box(msg, style="round"))

    with logger.timeit("Loading ERA data"):
        dates = retrieve_dates(*files_for_cycle.tolist())
        years = dates.year.unique().to_list()
        if dates.min().month == 1 and dates.min().day == 1:
            years.insert(0, dates.min().year - 1)
        msg = f"Loading data for years: {', '.join([str(y) for y in years])}"
        logger.info(msg)
        ds_era = load_era_interim(data_folder / "misc", *years)

        ds_era = slice_time(ds_era, ds[TIME])
        ds_era = slice_space(ds_era, ds[LONGITUDE], ds[LATITUDE])

    ds[STREAMFUNCTION] = compute_streamfunction_with_atmospheric_pressure(
        ds,
        ds_era,
        config["rho0"],
        config["g_prime"][0, 0, 0].item(),
        remove_avgs=True,
    )

    with logger.timeit("Filtering stream function"):
        msg = f"Using σ={sigma_ic} for initial condition"  # noqa: RUF001
        logger.info(msg)
        psi0_filt_da = xr.apply_ufunc(
            gaussian_filter,
            ds[STREAMFUNCTION][0].load(),
            kwargs={"sigma": sigma_ic},
            input_core_dims=[["i", "j"]],
            output_core_dims=[["i", "j"]],
            vectorize=True,
        )
        msg = f"Using σ={sigma_bc} for boundary conditions"  # noqa: RUF001
        ds["psi_filt"] = xr.apply_ufunc(
            gaussian_filter,
            ds[STREAMFUNCTION].load(),
            kwargs={"sigma": sigma_bc},
            input_core_dims=[["i", "j"]],
            output_core_dims=[["i", "j"]],
            vectorize=True,
        )
        logger.info(msg)

    with logger.timeit("Interpolating stream function"):
        regridded_psi: xr.DataArray = psi_regridder(ds[STREAMFUNCTION])
        regridded_psi_filt: xr.DataArray = psi_regridder(ds["psi_filt"])
        ds_interp = xr.Dataset(
            {
                LONGITUDE: (["i", "j"], lons),
                LATITUDE: (["i", "j"], lats),
                STREAMFUNCTION: ([TIME, "i", "j"], regridded_psi.data),
                "psi_filt": ([TIME, "i", "j"], regridded_psi_filt.data),
            },
            regridded_psi_filt.coords,
        )
        ds_interp = ds_interp.set_coords([LONGITUDE, LATITUDE])
        ds_interp = ds_interp.load()

    psis_ref = [
        torch.tensor(p, **specs).unsqueeze(0).unsqueeze(0) / f0
        for p in ds_interp[STREAMFUNCTION].to_numpy()
    ]

    var_ref = torch.stack([crop(psi[0, 0], bc) for psi in psis_ref]).var()

    with logger.timeit("Computing psi boundaries"):
        psis_filt = [
            torch.tensor(p, **specs).unsqueeze(0).unsqueeze(0) / f0
            for p in ds_interp["psi_filt"].to_numpy()
        ]
        psi_bcs = [psi[..., bc:-bc, bc:-bc] for psi in psis_filt]

    t0 = ds_interp[TIME][0]
    times = (ds_interp[TIME] - t0).dt.total_seconds().to_numpy()
    times = torch.tensor(times, **specs)

    psi0 = (
        torch.tensor(psi_regridder(psi0_filt_da).data, **specs)
        .unsqueeze(0)
        .unsqueeze(0)
        / f0
    )
    psi0_mean: float = psi0.mean()

    U: float = psi0_mean / L
    T = L / U

    msg = f"Cycle {step(c + 1, n_cycles)}: eNATL60 data loaded and processed."
    logger.info(box(msg, style="round"))

    psi_bcs = [
        Boundaries.extract(psi, bc, -bc - 1, bc, -bc - 1, 2) for psi in psis_filt
    ]
    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)
    qs = (compute_q_rg(p1) for p1 in psis_filt)
    q_bcs = [Boundaries.extract(q, bc - 2, -(bc - 1), bc - 2, -(bc - 1), 3) for q in qs]
    q_bc_interp = QuadraticInterpolation(times, q_bcs)

    space_params, time_params = dyadic_decomposition(
        order=5,
        xx_ref=xx,
        yy_ref=yy,
        Lxy_max=900_000,
        Lt_max=n_steps * dt,
    )

    basis = WaveletBasis(space_params, time_params)
    basis.n_theta = 7

    msg = f"Using basis of order {basis.order}"
    logger.info(msg)

    coefs = basis.generate_random_coefs()
    coefs = DecompositionCoefs.zeros_like(coefs)
    coefs = coefs.requires_grad_()

    numel = coefs.numel()
    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(
        [
            {
                "params": list(coefs.values()),
                "lr": 1e-2,
                "name": "Wavelet coefs",
            }
        ]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    lr_callback = LRChangeCallback(optimizer)
    early_stop = EarlyStop()

    coefs_scaled = coefs.scale(*(U**2 / L**2 for _ in range(basis.order)))

    register_params = RegisterParams(coefs=coefs_scaled.to_dict())

    for o in range(n_optim):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        qg = Forced(config_model)
        qg.wind_scaling = H1.item()
        qg.reset_time()
        qg.set_boundary_maps(psi_bc_interp, q_bc_interp)

        with torch.enable_grad():
            qg.set_psiq(crop(psi0, bc), crop(compute_q_rg(psi0), bc - 1))

            coefs_scaled = coefs.scale(*(U**2 / L**2 for _ in range(basis.order)))
            basis.set_coefs(coefs_scaled)

            wv = basis.localize(*compute_xy_q(xx, yy))

            loss = torch.tensor(0, **defaults.get())

            loss = update_loss(
                loss,
                qg.psi[0, 0],
                crop(psis_ref[0][0, 0], bc),
                qg.time,
                variance=var_ref,
            )

            for n in range(1, n_steps):
                qg.forcing = wv(qg.time)[None, None, ...]
                qg.step()

                loss = update_loss(
                    loss,
                    qg.psi[0, 0],
                    crop(psis_ref[n][0, 0], bc),
                    qg.time,
                    variance=var_ref,
                )
            if with_reg:
                for lvl, coef in coefs.items():
                    sigma_x = space_params[lvl]["sigma_x"] / dx
                    sigma_y = space_params[lvl]["sigma_y"] / dy
                    loss += (
                        gamma
                        * sqrt(sigma_x * sigma_y) ** (-5 / 3)
                        * coef.square().mean()
                    )

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        if torch.isnan(qg.psi).any():
            msg = "Streamfunction has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params.step(loss, coefs=coefs_scaled.to_dict())

        if early_stop.step(loss):
            msg = f"Convergence reached after {o + 1} iterations."
            logger.info(msg)
            break

        loss_ = loss.cpu().item()

        msg = (
            f"Cycle {step(c + 1, n_cycles)} | "
            f"Optimization step {step(o + 1, n_optim)} | "
            f"Loss: {loss_:>#10.5g} | "
            f"Best loss: {register_params.best_loss:>#10.5g}"
        )
        logger.info(msg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(coefs.values()), max_norm=1e0)

        optimizer.step()
        scheduler.step(loss)
        lr_callback.step()

    best_loss = register_params.best_loss
    msg = f"Optimization completed with loss: {best_loss:>#10.5g}"
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    msg_mem = f"Max memory allocated: {max_mem:.1f} MB."
    logger.info(box(msg, msg_mem, style="round"))
    output = {
        "cycle": c,
        "config": {
            "comparison_interval": comparison_interval,
            "gamma": reg_config["gamma"] if with_reg else 0,
            "basis": basis.get_params(),
            "numel": numel,
            "separation_steps": optim_config["separation"],
        },
        "optim": {
            "max_steps": n_optim,
            "nb_steps": o + 1,
            "loss": best_loss,
        },
        "specs": {"max_memory_allocated": max_mem},
        "coefs": register_params.params["coefs"],
    }
    outputs.append(output)

    torch.save(outputs, output_file)
    msg = f"Outputs saved to {output_file}"
    logger.info(box(msg, style="="))
