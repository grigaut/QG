"""Command Line Interface."""

import argparse
import pathlib
from dataclasses import dataclass
from pathlib import Path

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


@dataclass
class ScriptArgs:
    """Script arguments."""

    config: Path
    verbose: int

    @classmethod
    def from_cli(cls, *, config_default: Path) -> Self:
        """Instantiate script arguments from CLI.

        Returns:
            Self: ScriptArgs.
            default_config (str): Default configuration.
        """
        parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        cls._add_verbose(parser)
        cls._add_config(parser, config_default)
        return cls(**vars(parser.parse_args()))

    @classmethod
    def _add_config(cls, parser: argparse.ArgumentParser, config_default: Path) -> None:
        """Add configuration to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
            default_config (str): Default configuration.
        """
        parser.add_argument(
            "--config",
            type=pathlib.Path,
            default=config_default,
            help="Configuration File Path (path from root level)",
        )

    @classmethod
    def _add_verbose(cls, parser: argparse.ArgumentParser) -> None:
        """Add verbose to parser.

        Args:
            parser (argparse.ArgumentParser): Arguments parser.
        """
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Verbose level.",
        )
