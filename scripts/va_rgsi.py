"""
Double-gyre on regular domain.
"""

from collections.abc import Callable
from pathlib import Path
import shutil
import torch
from qg.decomposition.base import SpaceTimeDecomposition
from qg.decomposition.coefficients import DecompositionCoefs
from qg.decomposition.exp_exp.core import GaussianExpBasis
from qg.decomposition.exp_exp.param_generator import gaussian_exp_field
from qg.decomposition.supports.space.base import SpaceSupportFunction
from qg.decomposition.supports.time.base import TimeSupportFunction
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
from qg.fd import grad, grad_perp, interp_TP
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
from qg.stretching_matrix import compute_A_tilde
from qg.rgsi import RGSI
from qg.utils.cropping import crop
from qg.wind import compute_double_gyre_wind_curl

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

args = ScriptArgs.from_cli(config_default=Path("configs/va_rgsi_z2.toml"))
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
filename = f"{prefix}.pt"
folder: Path = output_config["folder"]

if folder.is_dir():
    msg = f"{folder} already exists and will be overidden."
    logger.warning(msg)
    shutil.rmtree(folder, ignore_errors=True)

output_file: Path = folder.joinpath(filename)

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

# if (f := sim_config["startup_file"]) is not None:
startup = torch.load("data/data.pt")
qg_3l.set_psiq(
    startup["psi"].to(**specs),
    startup["q"].to(**specs),
)
saver = SaveState(folder)
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
    "H": config["H"][:2],
    "g_prime": config["g_prime"][:2],
    "f0": config["f0"],
    "beta": config["beta"],
    "bottom_drag_coef": 0,
    "device": specs["device"],
    "dt": config["dt"],  # time-step (s)
}

H1, H2 = config["H"][0, 0, 0], config["H"][1, 0, 0]
g1, g2 = config["g_prime"][0, 0, 0], config["g_prime"][1, 0, 0]

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


def compute_regularization_func(
    psi2_basis: SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction],
    alpha: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build regularization function.

    Args:
        psi2_basis (SpaceTimeDecomposition): Basis.
        alpha (torch.Tensor) : Baroclinic radius perturbation.
        space (SpaceDiscretization2D): Space.

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

    f0: float = config["f0"]
    beta: float = config["beta"]

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
        return ((dt_q2 + adv_q2) / U * L * T).square().sum()

    return compute_reg


# PV computation


def build_compute_q_rg(A11, A12):
    return lambda psi1: compute_q1_interior(
        psi1,
        torch.zeros_like(psi1),
        A11,
        A12,
        qg_3l.dx,
        qg_3l.dy,
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

    ## Scaling parameters

    # time integration
    for n in range(1, n_steps):
        qg_3l.step()  # one RK3 integration step
        times.append(qg_3l.time.item())
        psis.append(qg_3l.psi[:, :1, psi_slice_w[0], psi_slice_w[1]])
    msg = f"Cycle {step(c + 1, n_cycles)}: Model spin-up completed."
    logger.info(box(msg, style="round"))

    var_ref = torch.stack([crop(psi[0, 0], bc) for psi in psis]).var()

    psi_bcs = [Boundaries.extract(psi, bc, -bc - 1, bc, -bc - 1, 2) for psi in psis]
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
        {"params": [kappa], "lr": 1e-1, "name": "κ"},
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
        qg = RGSI(config_sliced)
        qg.y0 = qg_3l.y0
        qg.set_wind_forcing(curl_tau[..., imin:imax, jmin:jmax])
        qg.reset_time()

        with torch.enable_grad():
            alpha = torch.exp(epsilon * kappa + kappa * kappa.abs()) - 1
            coefs_scaled = coefs.scale(
                *(1e-1 * psi0_mean / (n_steps * dt) ** k for k in range(basis.order))
            )

            basis.set_coefs(coefs_scaled)

            qg.basis = basis
            qg.alpha = alpha

            compute_reg = compute_regularization_func(basis, alpha)

            compute_q_rg = build_compute_q_rg(
                qg.A[:1, :1],
                qg.A[:1, 1:2],
            )
            q0 = crop(compute_q_rg(psi0), bc - 1)

            qs = (compute_q_rg(p1) for p1 in psis)
            q_bcs = [
                Boundaries.extract(q, bc - 2, -(bc - 1), bc - 2, -(bc - 1), 3)
                for q in qs
            ]
            q_bc_interp = QuadraticInterpolation(times, q_bcs)

            qg.set_psiq(crop(psi0[:, :1], bc), q0)
            qg.set_boundary_maps(psi_bc_interp, q_bc_interp)

            loss = torch.tensor(0, **defaults.get())

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
                    crop(psis[n][0, 0], bc),
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
        },
        "optim": {
            "max_steps": n_optim,
            "nb_steps": o + 1,
            "loss": best_loss,
        },
        "specs": {"max_memory_allocated": max_mem},
        "coords": (imin, imax, jmin, jmax),
        "alpha": register_params.params["alpha"],
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
