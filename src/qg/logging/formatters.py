"""Custom formatters."""

import logging
from typing import ClassVar

from qg.logging._levels import (
    CRITICAL,
    DEBUG,
    DETAIL,
    ERROR,
    INFO,
    WARNING,
)
from qg.logging.log_records import LogRecord


class Formatter(logging.Formatter):
    """Custom formatter."""

    level_formats: ClassVar[dict[int, str]] = {
        CRITICAL: "[CRITICAL]",
        ERROR: "[ERROR   ]",
        WARNING: "[WARNING ]",
        INFO: "[INFO    ]",
        DETAIL: "[DETAIL  ]",
        DEBUG: "[DEBUG   ]",
    }

    def format(self, record: LogRecord) -> str:
        """Format the psecified record as text.

        Args:
            record (logging.LogRecord): Record.

        Returns:
            str: Text.
        """
        timestamp = self.formatTime(record, "%H:%M:%S")
        level_str = self.level_formats.get(record.levelno, "[LOG     ]")
        msg = record.getMessage()
        txt_indent = record.indent
        indent = f"[{timestamp}] {level_str} "
        full_indent = txt_indent + " " * len(indent)
        msg_parts = msg.split("\n")
        return "\n".join(
            [
                indent + txt_indent + msg_parts[0],
                *[full_indent + p for p in msg_parts[1:]],
            ]
        )


class RichFormatter(logging.Formatter):
    """Custom formatter."""

    def format(self, record: LogRecord) -> str:
        """Format the psecified record as text.

        Args:
            record (logging.LogRecord): Record.

        Returns:
            str: Text.
        """
        msg = record.getMessage()
        txt_indent = record.indent
        msg_parts = msg.split("\n")
        return "\n".join([txt_indent + p for p in msg_parts])
