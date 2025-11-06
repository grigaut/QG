"""LogRecords."""

import logging

from typing_extensions import ParamSpec

P = ParamSpec("P")


class LogRecord(logging.LogRecord):
    """LogRecord."""

    _indent = ""

    @property
    def indent(self) -> str:
        """Identation."""
        return self._indent

    @indent.setter
    def indent(self, indent: str) -> None:
        self._indent = indent


def make_log_record(*args: P.args, **kwargs: P.kwargs) -> LogRecord:
    """Factory method to create the LogRecord."""
    return LogRecord(*args, **kwargs)
