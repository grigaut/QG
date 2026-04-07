"""
Double-gyre on regular domain.
"""

from collections.abc import Callable
from pathlib import Path
import shutil
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import xarray as xr
from qg.constants import EARTH_ANGULAR_ROTATION, EARTH_RADIUS
from qg.decomposition.base import SpaceTimeDecomposition
from qg.decomposition.coefficients import DecompositionCoefs
from qg.decomposition.exp_exp.core import GaussianExpBasis
from qg.decomposition.exp_exp.param_generator import gaussian_exp_field
from qg.decomposition.supports.space.base import SpaceSupportFunction
from qg.decomposition.supports.time.base import TimeSupportFunction
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
from qg.logging import setup_root_logger, getLogger
from qg.cli import ScriptArgs
from qg.config import (
    load_minimal_model_config,
    load_optimization_config,
    load_output_config,
    load_regularization_config,
    load_season_config,
    load_simulation_config,
)
from qg.fd import grad, interp_TP
from qg.interpolation import QuadraticInterpolation
from qg.logging.utils import box, sec2text, step
from qg.observations.satellite_track import SatelliteTrackMask
from qg.optim.callbacks import LRChangeCallback
from qg.optim.utils import EarlyStop, RegisterParams
from qg.pv import compute_q1_interior
from qg.solver.boundary_conditions.base import Boundaries
from qg.space import compute_xy_q
from qg.specs import defaults
from qg.stretching_matrix import compute_A_tilde
from qg.rgsi import RGSI
from qg.utils.cropping import crop
from qg.utils.storage import get_path_from_env

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

args = ScriptArgs.from_cli(config_default=Path("configs/va_enatl60_rgsi_summer.toml"))
specs = defaults.get()

setup_root_logger(args.verbose)
logger = getLogger(__name__)

# Configuratiosn

## Model
config = load_minimal_model_config(args.config)

sigma_bc = 16
sigma_ic = 16


## Areas
n_ens = config["n_ens"]
dx = config["dx"]
dy = config["dy"]
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
    dx=dx,
    dy=dy,
)
xs, ys = lonlat_to_xy(lons, lats)

# Width for boundaries

bc = 4

# Space

xv = torch.arange(xs.shape[0], **specs) * dx
yv = torch.arange(ys.shape[1], **specs) * dy

nx = xv.shape[0] - 1
ny = yv.shape[0] - 1
mask = torch.ones(nx, ny, **specs)

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

separation = int(optim_config["separation"] * dt / 3600 / 24)

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
    folder.mkdir()
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

config_model = {
    "xv": xv[bc:-bc],
    "yv": yv[bc:-bc],
    "n_ens": config["n_ens"],
    "mask": mask[bc:-bc, bc:-bc],
    "flux_stencil": config["flux_stencil"],
    "H": config["H"][:2],
    "g_prime": config["g_prime"][:2],
    "f0": f0,
    "beta": beta,
    "bottom_drag_coef": 0,
    "device": specs["device"],
    "dt": config["dt"],  # time-step (s)
}

H1, H2 = config["H"][0, 0, 0], config["H"][1, 0, 0]
g1, g2 = config["g_prime"][0, 0, 0], config["g_prime"][1, 0, 0]


qg = RGSI(config_model)
L: float = dx

beta_effect_w = beta * ((yv[:-1] + yv[1:]) / 2 - qg.y0)


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


## Regularization


def compute_regularization_func(
    psi2_basis: SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction],
    alpha: torch.Tensor,
    scale: float,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build regularization function.

    Args:
        psi2_basis (SpaceTimeDecomposition): Basis.
        alpha (torch.Tensor) : Baroclinic radius perturbation.
        scale (float): Regularizaiton scaling value.

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
            Regularization function.
    """
    A_tilde = compute_A_tilde(
        config["H"][:2, 0, 0],
        config["g_prime"][:2, 0, 0],
        alpha,
        **specs,
    )
    A_21 = A_tilde[1:2, :1]
    A_22 = A_tilde[1:2, 1:2]

    xq, yq = compute_xy_q(xx, yy)
    x = crop(xq, 1)
    y = crop(yq, 1)

    fpsi2 = psi2_basis.localize(x, y)
    fdx_psi2 = psi2_basis.localize_dx(x, y)
    fdy_psi2 = psi2_basis.localize_dy(x, y)
    flap_psi2 = psi2_basis.localize_laplacian(x, y)
    fdx_lap_psi2 = psi2_basis.localize_dx_laplacian(x, y)
    fdy_lap_psi2 = psi2_basis.localize_dy_laplacian(x, y)

    def compute_reg(
        psi1: torch.Tensor,
        dpsi1: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute regularization term.

        Args:
            psi1 (torch.Tensor): Top layer stream function.
            dpsi1 (torch.Tensor): Top layer stream function derivative.
            time (torch.Tensor): Time.

        Returns:
            torch.Tensor: ∂ₜq₂ + J(ѱ₂, q₂)
        """
        dt_lap_psi2 = flap_psi2.dt(time)
        dt_psi2 = fpsi2.dt(time)

        dt_q2 = dt_lap_psi2 - f0**2 * (
            A_22 * dt_psi2 + A_21 * interp_TP(crop(dpsi1, 1))
        )

        dx_psi1, dy_psi1 = grad(psi1, dx, dy)

        dx_psi1_i = (dx_psi1[..., 1:] + dx_psi1[..., :-1]) / 2
        dy_psi1_i = (dy_psi1[..., 1:, :] + dy_psi1[..., :-1, :]) / 2

        dx_psi2 = fdx_psi2(time)
        dy_psi2 = fdy_psi2(time)

        dy_q2 = (
            fdy_lap_psi2(time) - f0**2 * (A_22 * dy_psi2 + A_21 * crop(dy_psi1_i, 1))
        ) + beta

        dx_q2 = fdx_lap_psi2(time) - f0**2 * (
            A_22 * dx_psi2 + A_21 * crop(dx_psi1_i, 1)
        )

        adv_q2 = -dy_psi2 * dx_q2 + dx_psi2 * dy_q2
        return ((dt_q2 + adv_q2) / scale).square().sum()

    return compute_reg


# PV computation


def build_compute_q_rg(A11, A12):
    return lambda psi1: compute_q1_interior(
        psi1,
        torch.zeros_like(psi1),
        A11,
        A12,
        dx,
        dy,
        f0,
        beta_effect_w[..., 1:-1],
    )


def extract_psi_bc(psi: torch.Tensor) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, bc, -bc - 1, bc, -bc - 1, 2)


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
        psi_bcs = [extract_psi_bc(psi) for psi in psis_filt]

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

    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

    space_params, time_params = gaussian_exp_field(
        0,
        3,
        xx,
        yy,
        n_steps * dt,
        n_steps / 6 * 7200,
    )
    basis = GaussianExpBasis(space_params, time_params)
    coefs = DecompositionCoefs.zeros_like(basis.generate_random_coefs())
    coefs = coefs.requires_grad_()

    kappa: torch.Tensor = torch.tensor(0, **specs, requires_grad=True)
    numel = kappa.numel() + coefs.numel()
    params = [
        {"params": [kappa], "lr": 1e-2, "name": "κ"},
        {
            "params": list(coefs.values()),
            "lr": 1e0,
            "name": "Decomposition coefs",
        },
    ]

    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    lr_callback = LRChangeCallback(optimizer)
    early_stop = EarlyStop()

    coefs_scaled = coefs.scale(
        *(1e-1 * psi0_mean / (n_steps * dt) ** k for k in range(basis.order))
    )
    epsilon = 0.1
    register_params = RegisterParams(
        alpha=torch.exp(epsilon * kappa + kappa * kappa.abs()) - 1,
        coefs=coefs_scaled.to_dict(),
    )
    for o in range(n_optim):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        qg = RGSI(config_model)
        qg.reset_time()

        with torch.enable_grad():
            alpha = torch.exp(epsilon * kappa + kappa * kappa.abs()) - 1
            coefs_scaled = coefs.scale(
                *(1e-1 * psi0_mean / (n_steps * dt) ** k for k in range(basis.order))
            )

            basis.set_coefs(coefs_scaled)

            qg.basis = basis
            qg.alpha = alpha

            compute_reg = compute_regularization_func(basis, alpha, scale=1 / T**2)

            compute_q_rg = build_compute_q_rg(
                qg.A[:1, :1],
                qg.A[:1, 1:2],
            )
            q0 = crop(compute_q_rg(psi0), bc - 1)

            qs = (compute_q_rg(p1) for p1 in psis_filt)
            q_bcs = [
                Boundaries.extract(q, bc - 2, -(bc - 1), bc - 2, -(bc - 1), 3)
                for q in qs
            ]
            q_bc_interp = QuadraticInterpolation(times, q_bcs)

            qg.set_psiq(crop(psi0[:, :1], bc), q0)
            qg.set_boundary_maps(psi_bc_interp, q_bc_interp)

            loss = torch.tensor(0, **defaults.get())

            loss = update_loss(
                loss,
                qg.psi[0, 0],
                crop(psis_ref[0][0, 0], bc),
                qg.time,
                variance=var_ref,
            )

            for n in range(1, n_steps):
                psi1_ = qg.psi
                time = qg.time.clone()

                qg.step()

                psi1 = qg.psi

                if with_reg:
                    dpsi1_ = (psi1 - psi1_) / dt
                    reg = gamma * compute_reg(psi1_, dpsi1_, time)
                    loss += reg

                loss = update_loss(
                    loss,
                    psi1[0, 0],
                    crop(psis_ref[n][0, 0], bc),
                    qg.time,
                    variance=var_ref,
                )

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        if torch.isnan(qg.psi).any():
            msg = "Streamfunction has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params.step(
            loss,
            alpha=alpha,
            coefs=coefs_scaled.to_dict(),
        )

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

        torch.nn.utils.clip_grad_value_([kappa], clip_value=1.0)

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
            "season": season_value,
        },
        "optim": {
            "max_steps": n_optim,
            "nb_steps": o + 1,
            "loss": best_loss,
        },
        "specs": {"max_memory_allocated": max_mem},
        "alpha": register_params.params["alpha"],
        "coefs": register_params.params["coefs"],
    }
    outputs.append(output)

    torch.save(outputs, output_file)
    msg = f"Outputs saved to {output_file}"
    logger.info(box(msg, style="="))
