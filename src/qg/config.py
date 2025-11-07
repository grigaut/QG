"""Configuration related tools."""

from __future__ import annotations

from pathlib import Path
import toml
from typing import Any

import torch

from qg.io import get_absolute_storage_path
from qg.specs import defaults


def load_model_config(file: str | Path) -> dict[str, Any]:
    """Load model configuration from toml file.

    Args:
        file (str | Path): Toml file.

    Returns:
        dict[str, Any]: Configuration.
    """
    config_data = toml.load(Path(file))
    specs = defaults.get()
    Lx = config_data["Lx"]
    nx = config_data["nx"]
    Ly = config_data["Ly"]
    ny = config_data["ny"]
    xv = torch.linspace(0, Lx, nx + 1, **specs)
    yv = torch.linspace(0, Ly, ny + 1, **specs)

    H = torch.tensor(config_data["H"], **specs)[:, None, None]
    g_prime = torch.tensor(config_data["g_prime"], **specs)[:, None, None]

    return {
        "xv": xv,
        "yv": yv,
        "n_ens": config_data.get("n_ens", 1),
        "mask": torch.ones(nx, ny, **specs),
        "flux_stencil": config_data.get("flux_stencil", 5),
        "H": H,
        "g_prime": g_prime,
        "tau0": config_data.get("tau0", 8e-5),
        "f0": config_data["f0"],
        "beta": config_data.get("beta", 0),
        "bottom_drag_coef": config_data.get("bottom_drag_coef", 0),
        "device": specs["device"],
        "dt": config_data["dt"],  # time-step (s)
    }


def load_output_config(file: str | Path) -> dict[str, Any]:
    """Load output configuration from toml file.

    Args:
        file (str | Path): Toml file.

    Returns:
        dict[str, Any]: Configuration.
    """
    config_data = toml.load(Path(file))
    return {
        "folder": get_absolute_storage_path(Path(config_data.get("folder", "output"))),
        "interval": config_data.get("interval", 1),
        "prefix": config_data.get("prefix", "results_"),
    }


def load_simulation_config(file: str | Path) -> dict[str, Any]:
    """Load output configuration from toml file.

    Args:
        file (str | Path): Toml file.

    Returns:
        dict[str, Any]: Configuration.
    """
    config_data = toml.load(Path(file))
    startup = config_data.get("startup_file", None)
    return {
        "duration": config_data.get("duration"),
        "startup_file": startup,
    }


def load_optimization_config(file: str | Path) -> dict[str, Any]:
    """Load optimization configuration from toml file.

    Args:
        file (str | Path): Toml file.

    Returns:
        dict[str, Any]: Configuration.
    """
    config_data = toml.load(Path(file))
    return {
        "optimization_steps": config_data.get("optimization_steps", 100),
        "comparison_interval": config_data.get("comparison_interval", 1),
        "cycles": config_data.get("cycles", 1),
    }


def load_subdomain_config(file: str | Path) -> dict[str, Any]:
    """Load subdomain configuration from toml file.

    Args:
        file (str | Path): Toml file.

    Returns:
        dict[str, Any]: Configuration.
    """
    config_data = toml.load(Path(file))
    imin = config_data["imin"]
    imax = config_data["imax"]
    jmin = config_data["jmin"]
    jmax = config_data["jmax"]

    if imax < imin:
        msg = "imin must be lower than imax."
        raise ValueError(msg)
    if jmax < jmin:
        msg = "jmin must be lower than jmax."
        raise ValueError(msg)

    return {
        "imin": imin,
        "imax": imax,
        "jmin": jmin,
        "jmax": jmax,
    }
