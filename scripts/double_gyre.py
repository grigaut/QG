"""
Double-gyre on regular domain.
"""

from pathlib import Path
import torch

from mqgeometry import logging
from mqgeometry.cli import ScriptArgs
from mqgeometry.config import (
    load_model_config,
    load_output_config,
    load_simulation_config,
)
from mqgeometry.io import SaveState
from mqgeometry.logging.utils import box, sec2text
from mqgeometry.qgm import QGFV
from mqgeometry.specs import defaults

torch.backends.cudnn.deterministic = True

args = ScriptArgs.from_cli(config_default=Path("configs/double_gyre.toml"))
logging.setup_root_logger(args.verbose)
logger = logging.getLogger(__name__)
specs = defaults.get()


config = load_model_config(args.config)
output_config = load_output_config(args.config)
sim_config = load_simulation_config(args.config)

n_ens = config["n_ens"]
nx = config["xv"].shape[0] - 1
ny = config["yv"].shape[0] - 1
dt = config["dt"]

duration = sec2text(sim_config["duration"] * dt)
msg = (
    f"Running double gyre simulation on a {nx}x{ny} grid for a duration of {duration}."
)
logger.info(box(msg, style="="))

msg = f"Running code using {specs['device']}"
logger.info(msg)

yv = config["yv"]
Ly = yv[-1] - yv[0]

# forcing
yc = 0.5 * (yv[1:] + yv[:-1])  # cell centers
tau0 = config.pop("tau0")
curl_tau = -tau0 * 2 * torch.pi / Ly * torch.sin(2 * torch.pi * yc / Ly).tile((nx, 1))
curl_tau = curl_tau.unsqueeze(0).repeat(n_ens, 1, 1, 1)

qg = QGFV(config)
qg.set_wind_forcing(curl_tau)

saver = SaveState(output_config["folder"])
saver.register_tensors(psi=qg.psi, q=qg.q)
saver.save("ic.pt")
saver.copy_config(args.config)

# time params
n_steps = sim_config["duration"]

# time integration
for n in range(1, n_steps + 1):
    qg.step()  # one RK3 integration step
    if n % 500 == 0 and torch.isnan(qg.psi).any():
        msg = "NaN appeared while computing."
        raise ValueError(msg)
    if n % output_config["interval"] == 0:
        saver.save(f"step_{n}.pt")

if n % output_config["interval"] != 0:
    saver.save(f"step_{n}.pt")
