"""Logger class."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from typing_extensions import ParamSpec

from qg.logging._levels import DETAIL, INFO

if TYPE_CHECKING:
    from collections.abc import Generator

    from qg.logging.log_records import LogRecord

P = ParamSpec("P")


_indent_level: ContextVar[int] = ContextVar("_indent_level", default=0)


class Logger(logging.Logger):
    """Logger with the .detail method."""

    @property
    def _indent(self) -> str:
        return "    " * _indent_level.get()

    def detail(self, msg: object, *args: P.args, **kwargs: P.kwargs) -> None:
        """Implement the .detail method.

        Args:
            msg (object): Message.
            *args (P.args): Arguments.
            **kwargs (P.kwargs): Keyword arguments.
        """
        if self.isEnabledFor(DETAIL):
            self._log(DETAIL, msg, args, **kwargs)

    def makeRecord(  # noqa: N802
        self, *args: P.args, **kwargs: P.kwargs
    ) -> logging.LogRecord:
        """Create the LogRecord."""
        record: LogRecord = super().makeRecord(*args, **kwargs)
        record.indent = self._indent
        return record

    @contextmanager
    def timeit(self, message: str) -> Generator[None, None, None]:
        """Time the execution of some code.

        Args:
            message (str): Message to display.

        Yields:
            Generator[None, None, None]: Context manager.
        """
        self.info(f"{message}...")
        start = time.perf_counter()
        yield
        self.info(f"{message} done in {time.perf_counter() - start:.2f}s")

    @contextmanager
    def section(
        self,
        message: str | None = None,
        level: int = INFO,
    ) -> Generator[None, None, None]:
        """Create a section and indent all messages within.

        Args:
            message (str|None, optional): First message of the section
                (unindented). Defaults to None.
            level (int, optional): Level at which to print the section header.
                Defaults to INFO.

        Yields:
            Generator[None, None, None]: Context manager.
        """
        if message is not None:
            self.log(level, message)
        token = _indent_level.set(_indent_level.get() + 1)
        try:
            yield
        finally:
            _indent_level.reset(token)
