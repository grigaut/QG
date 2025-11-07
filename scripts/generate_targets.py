"""Generate observations."""

from pathlib import Path
import torch
from qg.io import SaveState
from qg.logging import setup_root_logger, getLogger
from qg.cli import ScriptArgs
from qg.config import (
    load_model_config,
    load_output_config,
    load_simulation_config,
)
from qg.logging.utils import box, sec2text
from qg.qgm import QGFV
from qg.specs import defaults
from qg.wind import compute_double_gyre_wind_curl

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
n_day = int(24 * 3600 / dt)
obs_delay = 5 * n_day

msg_model = "Generating observations from a double gyre simulation."

## Simulation
sim_config = load_simulation_config(args.config)

## Ouput
output_config = load_output_config(args.config)

logger.info(box(msg_model, style="="))

msg = f"Running code using {specs['device']}"
logger.info(msg)

yv = config["yv"]
Ly = yv[-1] - yv[0]

# forcing
yv = config["yv"]
xv = config["xv"]

# forcing
curl_tau = compute_double_gyre_wind_curl(config.pop("tau0"), xv, yv, config["n_ens"])

qg = QGFV(config)
qg.set_wind_forcing(curl_tau)

if (f := sim_config["startup_file"]) is not None:
    startup = torch.load(f)
    qg.set_psiq(
        startup["psi"].to(**specs),
        startup["q"].to(**specs),
    )
saver = SaveState(output_config["folder"], dtype=torch.float32)
saver.save("ic.pt", psi=qg.psi, q=qg.q)
saver.copy_config(args.config)

# Training set

n_obs = 500

duration = sec2text(n_obs * obs_delay * dt)
msg = f"[TRAINING] Generating set over {duration}..."
with logger.section(msg):
    for n in range(1, n_obs * obs_delay + 1):
        qg.step()
        if n % (obs_delay) == 0:
            saver.save(f"train_step_{n}.pt", psi=qg.psi, q=qg.q)

logger.info("One year evolution without data sampling...")
## Let 1 year pass
for n in range(1, n_year + 1):
    qg.step()
logger.info("Completed")

# Validation set

n_obs = 100

duration = sec2text(n_obs * obs_delay * dt)
msg = f"[VALIDATION] Generating set over {duration}..."
with logger.section(msg):
    for n in range(1, n_obs * obs_delay + 1):
        qg.step()
        if n % (obs_delay) == 0:
            saver.save(f"validate_step_{n}.pt", psi=qg.psi, q=qg.q)

logger.info("One year evolution without data sampling...")
## Let 1 year pass
for n in range(1, n_year + 1):
    qg.step()
logger.info("Completed")

# Testing set

n_obs = 100

duration = sec2text(n_obs * obs_delay * dt)
msg = f"[TEST] Generating test set over {duration}..."
with logger.section(msg):
    for n in range(1, n_obs * obs_delay + 1):
        qg.step()
        if n % (obs_delay) == 0:
            saver.save(f"test_step_{n}.pt", psi=qg.psi, q=qg.q)

msg = f"Dataset completed and saved to {saver.folder}"
logger.info(round(msg, style="="))
