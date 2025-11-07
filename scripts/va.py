"""
Double-gyre on regular domain.
"""

from pathlib import Path
import torch
from qg import logging
from qg.io import SaveState
from qg.logging import setup_root_logger, getLogger
from qg.cli import ScriptArgs
from qg.config import (
    load_model_config,
    load_optimization_config,
    load_output_config,
    load_simulation_config,
    load_subdomain_config,
)
from qg.fd import grad_perp
from qg.flux import div_flux_5pts_no_pad
from qg.interpolation import QuadraticInterpolation
from qg.logging.utils import box, sec2text, step
from qg.optim.utils import EarlyStop, RegisterParams
from qg.pv import compute_q1_interior, compute_q2_2l_interior
from qg.qg_mixed import QGMixed
from qg.qgm import QGFV
from qg.solver.boundary_conditions.base import Boundaries
from qg.specs import defaults
from qg.utils.cropping import crop

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

args = ScriptArgs.from_cli(config_default=Path("configs/double_gyre.toml"))
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

logger.info(box(msg_model, msg_sim, msg_optim, msg_subdomain, msg_output, style="="))

msg = f"Running code using {specs['device']}"
logger.info(msg)

yv = config["yv"]
Ly = yv[-1] - yv[0]

# forcing
yc = 0.5 * (yv[1:] + yv[:-1])  # cell centers
tau0 = config.pop("tau0")
curl_tau = -tau0 * 2 * torch.pi / Ly * torch.sin(2 * torch.pi * yc / Ly).tile((nx, 1))
curl_tau = curl_tau.unsqueeze(0).repeat(n_ens, 1, 1, 1)

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
p = 4

psi_slice_w = [slice(imin - p, imax + 1 + p), slice(jmin - p, jmax + 1 + p)]

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

beta_effect = config["beta"] * (yv[jmin:jmax] - qg_3l.y0)
beta_effect_w = config["beta"] * (yv[jmin - p : jmax + p] - qg_3l.y0)


def compute_dtq2(dpsi1: torch.Tensor, dpsi2: torch.Tensor) -> torch.Tensor:
    return compute_q2_2l_interior(
        dpsi1,
        dpsi2,
        H2,
        g2,
        qg_3l.dx,
        qg_3l.dy,
        config["f0"],
        torch.zeros_like(beta_effect[..., 1:-1]),
    )


def compute_q2(psi1: torch.Tensor, psi2: torch.Tensor) -> torch.Tensor:
    return compute_q2_2l_interior(
        psi1,
        psi2,
        H2,
        g2,
        qg_3l.dx,
        qg_3l.dy,
        config["f0"],
        beta_effect[..., 1:-1],
    )


def rmse(f: torch.Tensor, f_ref: torch.Tensor) -> float:
    """RMSE."""
    return (f - f_ref).square().mean().sqrt() / f_ref.square().mean().sqrt()


def regularization(
    psi1: torch.Tensor,
    psi2: torch.Tensor,
    dpsi1: torch.Tensor,
    dpsi2: torch.Tensor,
) -> torch.Tensor:
    """Compute regularization.

    Args:
        psi1 (torch.Tensor): Top layer stream function.
        psi2 (torch.Tensor): Bottom layer stream function.
        dpsi1 (torch.Tensor): Top layer stream function derivative.
        dpsi2 (torch.Tensor): Bottom layer stream function derivative

    Returns:
        torch.Tensor: ||∂_t q₂ + J(ѱ₂,q₂)||² (normalized by U / LT)
    """
    dtq2 = compute_dtq2(dpsi1, dpsi2)[..., 1:-1, 1:-1]
    q2 = compute_q2(psi1, psi2)

    u2, v2 = grad_perp(psi2[..., 1:-1, 1:-1], qg_3l.dx, qg_3l.dy)

    dq_2 = div_flux_5pts_no_pad(
        q2, u2[..., 1:-1, :], v2[..., :, 1:-1], qg_3l.dx, qg_3l.dy
    )
    return ((dtq2 + dq_2) * T**2).square().sum()


gamma = 10 / comparison_interval

# PV computation


def compute_q_psi2(psi1, psi2) -> torch.Tensor:
    return compute_q1_interior(
        psi1,
        psi2,
        H1,
        g1,
        g2,
        qg_3l.dx,
        qg_3l.dy,
        config["f0"],
        beta_effect_w[..., 1:-1],
    )


for c in range(n_cycles):
    qg_3l.reset_time()
    times = [qg_3l.time]
    psis = [qg_3l.psi[:, :1, psi_slice_w[0], psi_slice_w[1]]]

    ## Scaling parameters

    # time integration
    for n in range(1, n_steps + 1):
        qg_3l.step()  # one RK3 integration step
        times.append(qg_3l.time.item())
        psis.append(qg_3l.psi[:, :1, psi_slice_w[0], psi_slice_w[1]])

    msg = f"Cycle {step(c + 1, n_cycles)}: Model spin-up completed."
    logger.info(box(msg, style="round"))

    psi_bcs = [Boundaries.extract(psi, p, -p - 1, p, -p - 1, 2) for psi in psis]
    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

    psi0 = psis[0]
    psi0_mean = psi0[:, :1].mean()

    alpha = torch.tensor(0.5, **specs, requires_grad=True)
    psi2_adim = (torch.rand_like(psi0) * 1e-1).requires_grad_()
    dpsi2 = (torch.rand_like(psi2_adim) * 1e-3).requires_grad_()

    numel = alpha.numel() + psi2_adim.numel() + dpsi2.numel()
    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(
        [
            {"params": [alpha], "lr": 1e-1},
            {"params": [psi2_adim], "lr": 1e-1},
            {"params": [dpsi2], "lr": 1e-3},
        ],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    early_stop = EarlyStop()
    register_params_mixed = RegisterParams(
        alpha=alpha,
        psi2=psi2_adim * psi0_mean,
        dpsi2=dpsi2,
    )
    outputs = []
    for o in range(n_optim):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        qg = QGMixed(config_sliced)
        qg.y0 = qg_3l.y0
        qg.set_wind_forcing(curl_tau[..., imin:imax, jmin:jmax])
        qg.reset_time()

        with torch.enable_grad():
            psi2 = psi2_adim * psi0_mean
            q0 = crop(compute_q_psi2(psi0, psi2 + alpha * psi0), p - 1)
            psis_ = (
                (p[:, :1], psi2 + n * dt * dpsi2 + alpha * p[:, :1])
                for n, p in enumerate(psis)
            )
            qs = (compute_q_psi2(p1, p2) for p1, p2 in psis_)
            q_bcs = [
                Boundaries.extract(q, p - 2, -(p - 1), p - 2, -(p - 1), 3) for q in qs
            ]

            qg.set_psiq(crop(psi0[:, :1], p), q0)
            q_bc_interp = QuadraticInterpolation(times, q_bcs)
            qg.alpha = torch.ones_like(qg.psi) * alpha
            qg.set_boundary_maps(psi_bc_interp, q_bc_interp)
            qg.dpsi2 = crop(dpsi2, p)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps + 1):
                psi1_ = qg.psi
                psi2_ = crop(psi2 + (n - 1) * dt * dpsi2, p) + alpha * psi1_

                qg.step()

                psi1 = qg.psi
                dpsi1_ = (psi1 - psi1_) / dt
                dpsi2_ = crop(dpsi2, p) + alpha * (psi1 - psi1_) / dt
                reg = gamma * (regularization(psi1_, psi2_, dpsi1_, dpsi2_))
                loss += reg

                if n % comparison_interval == 0:
                    loss += rmse(psi1[0, 0], crop(psis[n][0, 0], p))

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params_mixed.step(
            loss,
            alpha=alpha,
            psi2=psi2,
            dpsi2=dpsi2,
        )

        if early_stop.step(loss):
            msg = f"Convergence reached after {o + 1} iterations."
            logger.info(msg)
            break

        loss_ = loss.cpu().item()

        msg = (
            f"Cycle {step(c + 1, n_cycles)} | "
            f"Optimization step {step(o + 1, n_optim)} | "
            f"Loss: {loss_:3.5f}"
        )
        logger.info(msg)

        loss.backward()

        lr_alpha = optimizer.param_groups[0]["lr"]
        grad_alpha = alpha.grad.item()
        torch.nn.utils.clip_grad_value_([alpha], clip_value=1.0)
        grad_alpha_ = alpha.grad.item()

        lr_psi2 = optimizer.param_groups[1]["lr"]
        norm_grad_psi2 = psi2_adim.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([psi2_adim], max_norm=1e-1)
        norm_grad_psi2_ = psi2_adim.grad.norm().item()

        lr_dpsi2 = optimizer.param_groups[2]["lr"]
        norm_grad_dpsi2 = dpsi2.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([dpsi2], max_norm=1e-1)
        norm_grad_dpsi2_ = dpsi2.grad.norm().item()
        with logger.section("ɑ parameters:", level=logging.DETAIL):  # noqa: RUF001
            msg = f"Learning rate {lr_alpha:.1e}"
            logger.detail(msg)
            msg = f"Gradient: {grad_alpha:.1e} -> {grad_alpha_:.1e}"
            logger.detail(msg)

        with logger.section("ѱ₂ parameters:", level=logging.DETAIL):
            msg = f"Learning rate {lr_psi2:.1e}"
            logger.detail(msg)
            msg = f"Gradient norm: {norm_grad_psi2:.1e} -> {norm_grad_psi2_:.1e}"
            logger.detail(msg)
        with logger.section("dѱ₂ parameters:", level=logging.DETAIL):
            msg = f"Learning rate {lr_dpsi2:.1e}"
            logger.detail(msg)
            msg = f"Gradient norm: {norm_grad_dpsi2:.1e} -> {norm_grad_dpsi2_:.1e}"
            logger.detail(msg)

        optimizer.step()
        scheduler.step(loss)

    best_loss = register_params_mixed.best_loss
    msg = f"ɑ, dɑ, ѱ₂ and dѱ₂ optimization completed with loss: {best_loss:3.5f}"  # noqa: RUF001
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    msg_mem = f"Max memory allocated: {max_mem:.1f} MB."
    logger.info(box(msg, msg_mem, style="round"))
    output = {
        "cycle": c,
        "config": {
            "comparison_interval": comparison_interval,
            "optimization_steps": [n_optim],
        },
        "specs": {"max_memory_allocated": max_mem},
        "coords": (imin, imax, jmin, jmax),
        "alpha": register_params_mixed.params["alpha"].detach().cpu(),
        "psi2": register_params_mixed.params["psi2"].detach().cpu(),
        "dpsi2": register_params_mixed.params["dpsi2"].detach().cpu(),
    }
outputs.append(output)
torch.save(outputs, output_file)
msg = f"Outputs saved to {output_file}"
logger.info(box(msg, style="="))
