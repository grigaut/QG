"""Concatenate targets into a single file."""

from qg.logging import getLogger
from pathlib import Path

import torch
from qg.config import load_model_config
from qg.logging.utils import sec2text
from qg.utils.parsing import sort_files

logger = getLogger(__name__)


def load_psi(file: Path) -> torch.Tensor:
    psi = torch.load(file)["psi"]
    return psi[:, :1]


config = load_model_config("output/g5k/targets/_config.toml")
dt = config["dt"]
input_dir = Path("output/g5k/targets")

output_dir = Path("output/targets")
if not output_dir.is_dir():
    output_dir.mkdir()
    gitignore = output_dir.joinpath(".gitignore")
    with gitignore.open("w") as file:
        file.write("*")

for prefix in ["train", "validate", "test"]:
    with logger.section(f"[{prefix.upper()}] Concatenating data"):
        files = input_dir.rglob(f"{prefix}_*")

        steps, files = sort_files(files, f"{prefix}_step_", ".pt")

        logger.info(f"{len(files)} files sorted.")

        times = torch.tensor([dt * s for s in steps], dtype=torch.int32)

        logger.info(
            f"Time ranges from {sec2text(times[0].item())} to {sec2text(times[-1].item())}."
        )

        torch.save(
            {"times": times, "psi_tops": torch.stack([load_psi(f) for f in files])},
            output_dir.joinpath(f"{prefix}.pt"),
        )
        logger.info(
            f"Times and top stream function values saved to {output_dir.joinpath(f'{prefix}.pt')}"
        )
logger.info("Completed")
