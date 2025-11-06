"""Input / Output related methods."""

from pathlib import Path
import torch


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
        self.folder = Path(output_folder)
        if not self.folder.exists():
            self.folder.mkdir()
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
