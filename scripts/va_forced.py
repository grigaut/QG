"""
Double-gyre on regular domain.
"""

from math import sqrt
from pathlib import Path
import torch
from qg.decomposition.coefficients import DecompositionCoefs
from qg.decomposition.wavelets.core import WaveletBasis
from qg.decomposition.wavelets.param_generators import dyadic_decomposition
from qg.forced import Forced
from qg.io import SaveState
from qg.logging import setup_root_logger, getLogger
from qg.cli import ScriptArgs
from qg.config import (
    load_model_config,
    load_optimization_config,
    load_output_config,
    load_regularization_config,
    load_simulation_config,
    load_subdomain_config,
)
from qg.fd import grad_perp
from qg.interpolation import QuadraticInterpolation
from qg.logging.utils import box, sec2text, step
from qg.observations.satellite_track import SatelliteTrackMask
from qg.optim.callbacks import LRChangeCallback
from qg.optim.utils import EarlyStop, RegisterParams
from qg.pv import compute_q1_interior
from qg.qgm import QGFV
from qg.solver.boundary_conditions.base import Boundaries
from qg.space import compute_xy_q
from qg.specs import defaults
from qg.utils.cropping import crop
from qg.wind import compute_double_gyre_wind_curl

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

args = ScriptArgs.from_cli(config_default=Path("configs/va_forced.toml"))
specs = defaults.get()

setup_root_logger(args.verbose)
logger = getLogger(__name__)

# Configuratiosn

## Model
config = load_model_config(args.config)

n_ens = config["n_ens"]
nx = config["xv"].shape[0] - 1
ny = config["yv"].shape[0] - 1
dt = config["dt"]
n_year = int(365 * 24 * 3600 / dt)

msg_model = "Variational assimilation of double gyre simulation."

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

msg_optim = (
    f"Performing {n_cycles} cycles with up to {n_optim} optimization steps.\n"
    f"Loss will be evaluated every {sec2text(comparison_interval * dt)}."
)

if optim_config["separation"] != 0:
    msg_sim += f"\nCycles are separated by {sec2text(optim_config['separation'] * dt)}."

## Regularization
reg_config = load_regularization_config(args.config)
with_reg = reg_config["gamma"] != 0

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

## Subdomain
subdomain_config = load_subdomain_config(args.config)
imin: int = subdomain_config["imin"]
imax: int = subdomain_config["imax"]
jmin: int = subdomain_config["jmin"]
jmax: int = subdomain_config["jmax"]

msg_subdomain = f"Focusing on i in [{imin}, {imax}] and j in [{jmin}, {jmax}]"

## Ouput
output_config = load_output_config(args.config)

prefix = output_config["prefix"]
filename = f"{prefix}_{imin}_{imax}_{jmin}_{jmax}.pt"
output_file: Path = output_config["folder"].joinpath(filename)

msg_output = f"Outputs will be save at {output_file}"

## Observations
yv = config["yv"]
xv = config["xv"]

Lx = xv[-1] - xv[0]
Ly = yv[-1] - yv[0]

dx = Lx / nx
dy = Ly / ny

X, Y = torch.meshgrid(xv, yv, indexing="ij")

xx, yy = X[imin : imax + 1, jmin : jmax + 1], Y[imin : imax + 1, jmin : jmax + 1]

obs_mask = SatelliteTrackMask(
    xx,
    yy,
    track_width=100000,
    track_interval=500000,
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
        msg_subdomain,
        msg_output,
        style="=",
    )
)

msg = f"Running code using {specs['device']}"
logger.info(msg)

Ly = yv[-1] - yv[0]

# forcing
curl_tau = compute_double_gyre_wind_curl(config.pop("tau0"), xv, yv, config["n_ens"])

qg_3l = QGFV(config)
qg_3l.set_wind_forcing(curl_tau)

if (f := sim_config["startup_file"]) is not None:
    startup = torch.load(f)
    qg_3l.set_psiq(
        startup["psi"].to(**specs),
        startup["q"].to(**specs),
    )
saver = SaveState(output_config["folder"])
saver.save("ic.pt", psi=qg_3l.psi, q=qg_3l.q)
saver.copy_config(args.config)

u, v = grad_perp(qg_3l.psi[0, 0], qg_3l.dx, qg_3l.dy)
U: float = (u.abs().max() ** 2 + v.abs().max() ** 2).sqrt().item()
L: float = qg_3l.dx.item()
T: float = L / U

# Width for boundaries
bc = 4

psi_slice_w = [slice(imin - bc, imax + 1 + bc), slice(jmin - bc, jmax + 1 + bc)]

config_sliced = {
    "xv": config["xv"][imin : imax + 1],
    "yv": config["yv"][jmin : jmax + 1],
    "n_ens": config["n_ens"],
    "mask": config["mask"][imin:imax, jmin:jmax],
    "flux_stencil": config["flux_stencil"],
    "H": config["H"][:1] * config["H"][1:2] / (config["H"][:1] + config["H"][1:2]),
    "g_prime": config["g_prime"][1:2],
    "f0": config["f0"],
    "beta": config["beta"],
    "bottom_drag_coef": 0,
    "device": specs["device"],
    "dt": config["dt"],  # time-step (s)
}

H1, H2 = config["H"][0, 0, 0], config["H"][1, 0, 0]
g1, g2 = config["g_prime"][0, 0, 0], config["g_prime"][1, 0, 0]

beta_effect = config["beta"] * (
    (yv[1 + jmin : jmax + 1] + yv[jmin:jmax]) / 2 - qg_3l.y0
)
beta_effect_w = config["beta"] * (
    (yv[1 + jmin - bc : jmax + bc + 1] + yv[jmin - bc : jmax + bc]) / 2 - qg_3l.y0
)


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
        1 / H1 / H2 * (H1 + H1) / g2,
        0,
        dx,
        dy,
        config["f0"],
        beta_effect_w[..., 1:-1],
    )


outputs = []
for c in range(n_cycles):
    qg_3l.reset_time()
    times = [qg_3l.time]
    psis = [qg_3l.psi[:, :1, psi_slice_w[0], psi_slice_w[1]]]
    psi0 = psis[0]
    psi0_mean = crop(psis[0][:, :1], bc).mean()

    var_ref = torch.stack([crop(psi[0, 0], bc) for psi in psis]).var()

    ## Scaling parameters

    # time integration
    for n in range(1, n_steps + 1):
        qg_3l.step()  # one RK3 integration step
        times.append(qg_3l.time.item())
        psis.append(qg_3l.psi[:, :1, psi_slice_w[0], psi_slice_w[1]])

    msg = f"Cycle {step(c + 1, n_cycles)}: Model spin-up completed."
    logger.info(box(msg, style="round"))

    psi_bcs = [Boundaries.extract(psi, bc, -bc - 1, bc, -bc - 1, 2) for psi in psis]
    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)
    qs = (compute_q_rg(p1) for p1 in psis)
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
        qg = Forced(config_sliced)
        qg.wind_scaling = H1.item()
        qg.y0 = qg_3l.y0
        qg.set_wind_forcing(curl_tau[..., imin:imax, jmin:jmax])
        qg.reset_time()
        qg.set_boundary_maps(psi_bc_interp, q_bc_interp)

        with torch.enable_grad():
            qg.set_psiq(crop(psi0, bc), crop(compute_q_rg(psi0), bc - 1))

            coefs_scaled = coefs.scale(*(U**2 / L**2 for _ in range(basis.order)))
            basis.set_coefs(coefs_scaled)

            wv = basis.localize(*compute_xy_q(xx, yy))

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps):
                qg.forcing = wv(qg.time)[None, None, ...]
                qg.step()

                loss = update_loss(
                    loss,
                    qg.psi[0, 0],
                    crop(psis[n][0, 0], bc),
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
        "coords": (imin, imax, jmin, jmax),
        "coefs": register_params.params["coefs"],
    }
    outputs.append(output)

    torch.save(outputs, output_file)
    msg = f"Outputs saved to {output_file}"
    logger.info(box(msg, style="="))

    for _ in range(optim_config["separation"]):
        qg_3l.step()
    if optim_config["separation"] > 0:
        msg = f"Performed {optim_config['separation']} steps before next cycle."
        logger.info(box(msg, style="="))
