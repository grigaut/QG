"""Logging-related tools."""

from qg.logging._levels import CRITICAL, DEBUG, DETAIL, INFO, WARNING
from qg.logging.core import getLogger, setup_root_logger

__all__ = [
    "CRITICAL",
    "DEBUG",
    "DETAIL",
    "INFO",
    "WARNING",
    "getLogger",
    "setup_root_logger",
]
