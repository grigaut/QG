"""Storage parsing."""

import os
from pathlib import Path


class StorageError(Exception):
    """Storage-related exception."""


def get_path_from_env(key: str = "STORAGE") -> Path:
    """Read .env to find storage path.

    Returns:
        Path: $STORAGE from .env or $PWD if no STORAGE environment variable.
    """
    if key in os.environ:
        return Path(os.environ[key]).absolute()
    msg = f"Impossible to read the {key} from environment variables."
    raise StorageError(msg)


def get_absolute_storage_path(path: Path) -> Path:
    """Make an absolute stroage path.

    Args:
        path (Path): Path to use to save data.

    Returns:
        Path: Absolute storage path.
    """
    if path.is_absolute():
        if path.is_relative_to(get_path_from_env()):
            return path
        msg = f"Path {path} is absolute, use relative path instead."
        raise StorageError(msg)
    return get_path_from_env().joinpath(path).absolute()
