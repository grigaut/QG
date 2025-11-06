"""Core of logging module."""

from __future__ import annotations

import logging
import sys

from qg.logging._levels import DEBUG, DETAIL, INFO, WARNING
from qg.logging.log_records import make_log_record

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme

    WITH_RICH = True
except ImportError:
    WITH_RICH = False

from qg.logging.environments import in_notebook, in_oar
from qg.logging.formatters import Formatter, RichFormatter
from qg.logging.logger import Logger


def setup_root_logger(verbose_level: int = 1) -> None:
    """Setup root logger."""
    logging.setLoggerClass(Logger)
    logging.setLogRecordFactory(make_log_record)
    logging.addLevelName(DETAIL, "DETAIL")
    level_map = {
        0: WARNING,
        1: INFO,
        2: DETAIL,
        3: DEBUG,
    }
    level = level_map.get(verbose_level, INFO)
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    oar = in_oar()
    notebook = in_notebook()

    handler = get_handler_no_rich() if (oar or not WITH_RICH) else get_handler_rich()
    handler.setLevel(level)
    logger.addHandler(handler)

    if not notebook:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)


def get_handler_rich() -> RichHandler:
    """Define Handler using rich.

    Returns:
        RichHandler: Handler.
    """
    custom_theme = Theme(
        {
            "logging.level.detail": "cyan",
            "logging.level.info": "blue",
            "logging.level.debug": "dim",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
            "logging.level.critical": "bold red",
        }
    )

    console = Console(file=sys.stdout, theme=custom_theme, force_jupyter=False)

    handler = RichHandler(
        console=console,
        rich_tracebacks=False,
        show_time=True,
        show_path=False,
        markup=False,
        show_level=True,
        log_time_format="[%H:%M:%S]",
    )
    handler.setFormatter(RichFormatter())
    return handler


def get_handler_no_rich() -> logging.StreamHandler:
    """Define Handler without using rich.

    Returns:
        logging.StreamHandler: Handler.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(Formatter())
    return handler


def getLogger(name: str | None = None) -> Logger:  # noqa: N802
    """Wrapper for logging.getLogger.

    Args:
        name (str | None, optional): Logger name. Defaults to None.

    Returns:
        Logger: Logger.
    """
    return logging.getLogger(name)
