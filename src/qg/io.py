"""Input / Output related methods."""

from __future__ import annotations

import os
from pathlib import Path
import toml
import torch

from qg import logging

logger = logging.getLogger(__name__)


def get_storage_path(key: str = "STORAGE") -> Path:
    """Read .env to find storage path.

    Returns:
        Path: $STORAGE from .env or $PWD if no STORAGE environment variable.
    """
    if key in os.environ:
        return Path(os.environ[key]).absolute()
    msg = f"Impossible to read the {key} from environment variables."
    raise ValueError(msg)


def get_absolute_storage_path(path: Path) -> Path:
    """Make an absolute stroage path.

    Args:
        path (Path): Path to use to save data.

    Returns:
        Path: Absolute storage path.
    """
    if path.is_absolute():
        if path.is_relative_to(get_storage_path()):
            return path
        msg = f"Path {path} is absolute, use relative path instead."
        raise ValueError(msg)
    return get_storage_path().joinpath(path).absolute()


class SaveState:
    """Class to save model state."""

    def __init__(
        self,
        output_folder: str | Path,
    ) -> None:
        """Satet saver.

        Args:
            output_folder (str | Path): Output folder to save in.
        """
        self.folder = get_absolute_storage_path(Path(output_folder))
        logger.info(f"Output will be saved under {self.folder}")
        if not self.folder.exists():
            self.folder.mkdir()
            gitignore = self.folder.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")

        self.tensors = {}

    def register_tensors(self, **tensors: torch.Tensor) -> None:
        """Register tensors to save."""
        self.tensors = tensors

    def save(self, filename: str) -> None:
        """Save registered tensors.

        Args:
            filename (str): Name of the file to save in.

        Raises:
            ValueError: If no tensors were registered.
        """
        if not self.tensors:
            msg = "No tensors registered."
            raise ValueError(msg)
        path = self.folder.joinpath(filename)
        torch.save({k: v.detach().cpu() for k, v in self.tensors.items()}, path)
        msg = f"Saved tensors to {path}"
        logger.info(msg)

    def copy_config(self, config_path: str | Path) -> None:
        """Save registered tensors.

        Args:
            filename (str): Name of the file to save in.

        Raises:
            ValueError: If no tensors were registered.
        """
        path = self.folder.joinpath("_config.toml")
        toml.dump(toml.load(Path(config_path)), path.open("w"))
        msg = f"Saved configuration to {path}"
        logger.info(msg)
