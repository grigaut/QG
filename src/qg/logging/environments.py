"""Environment detection."""

import os


def in_notebook() -> bool:
    """Detect if code is run within a jupyter notebook.

    Returns:
        bool: True if run within a Jupyter notebook.
    """
    try:
        from IPython import get_ipython  # noqa: PLC0415

        return get_ipython() is not None
    except ImportError:
        return False


def in_oar() -> bool:
    """Detect if code is run by OAR.

    Returns:
        bool: True if run by OAR.
    """
    return "OAR_JOB_ID" in os.environ
