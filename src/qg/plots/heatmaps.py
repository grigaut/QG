"""Plots from files."""

from __future__ import annotations

import plotly.colors as pco
import plotly.graph_objects as go
import torch

from qg.plots.animated_plots import BaseAnimatedPlot


class AnimatedHeatmaps(BaseAnimatedPlot[torch.Tensor]):
    """Animated Heatmap with shared colorscale."""

    _color_bar_text = ""
    _zmax = None
    _zmin = None
    _colorscale = pco.diverging.RdBu_r

    def __init__(
        self,
        datas: list[list[torch.Tensor]],
    ) -> None:
        """Instantiate the plot.

        Args:
            datas (list[list[torch.Tensor]]): Data list.
            frame_labels (list[str] | None, optional): Frames names.
            Defaults to None.
        """
        self._colorbar = self._create_colorbar()
        super().__init__(datas)

    @property
    def colorbar(self) -> go.heatmap.ColorBar:
        """Plot colorbar."""
        return self._colorbar

    def set_zbounds(self, zmin: float, zmax: float) -> None:
        """Set zmin and zmax.

        Args:
            zmin (float): zmin.
            zmax (float): zmax.
        """
        self._zmin = zmin
        self._zmax = zmax

    def _create_colorbar(self) -> go.heatmap.ColorBar:
        """Create the colorbar.

        Returns:
            go.heatmap.ColorBar: Colorbar.
        """
        return go.heatmap.ColorBar(
            exponentformat="e",
            showexponent="all",
            title={"text": self._color_bar_text, "side": "right"},
            thickness=50,
        )

    def set_colorbar_text(self, text: str) -> None:
        """Set the colorbar text.

        Args:
            text (str): Colorbar text.
        """
        self._color_bar_text = text
        self._colorbar = self._create_colorbar()

    def _add_traces(self) -> None:
        """Initialize the traces."""
        zmax = self._compute_zmax(self._datas[0][1])
        zmin = -zmax if self._zmin is None else self._zmin
        showscales = self._compute_showscales(self._datas[0][1])
        for subplot_index in range(self.n_subplots):
            row, col = self.map_subplot_index_to_subplot_loc(subplot_index)
            self.figure.add_trace(
                go.Heatmap(
                    z=self._datas[0][1][subplot_index],
                    showscale=showscales[subplot_index],
                    colorscale=self._colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=self.colorbar,
                ),
                row=row,
                col=col,
            )

    def _compute_frame(self, frame_index: int) -> go.Frame:
        """Compute a frame a at a given index.

        Args:
            frame_index (int): Frame index.

        Returns:
            go.Frame: Frame.
        """
        frame_arrays = self._datas[frame_index][1]
        zmax = self._compute_zmax(self._datas[0][1])
        zmin = -zmax if self._zmin is None else self._zmin
        showscales = self._compute_showscales(frame_arrays)
        return go.Frame(
            data=[
                go.Heatmap(
                    z=frame_arrays[subplot_index],
                    colorscale=self._colorscale,
                    showscale=showscales[subplot_index],
                    colorbar=self.colorbar,
                    zmin=zmin,
                    zmax=zmax,
                )
                for subplot_index in range(self.n_subplots)
            ],
            traces=list(range(self.n_subplots)),
            name=frame_index,
        )

    def _compute_step(self, frame_index: int) -> go.layout.slider.Step:
        """Compute a step at a given index.

        Args:
            frame_index (int): Frame index.

        Returns:
            go.layout.slider.Step: Step..
        """
        return go.layout.slider.Step(
            args=[
                [frame_index],
                {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            label=self._datas[frame_index][0],
            method="animate",
        )

    def _compute_showscales(self, arrays: list[torch.Tensor]) -> list[bool]:
        """Compute which trace to show scale of.

        Args:
            arrays (list[torch.Tensor]): List of data arrays.

        Returns:
            list[bool]: True on the index of scale to show, False elsewhere.
        """
        not_empty_arrays = torch.tensor(
            [~torch.isnan(array).all() for array in arrays],
        )
        find_first = torch.cumsum(not_empty_arrays, 0)
        remove_trailing = torch.cumsum(find_first, 0)
        return [bool(res) for res in (remove_trailing == 1)]

    def _compute_zmax(self, arrays: list[torch.Tensor]) -> float:
        """Compute maxiumal value over al ist of arrays, without nans.

        Args:
            arrays (list[torch.Tensor]): List of arrays to find the max of.

        Returns:
            float: Maximum values over the arrays, np.nan only if all array
            are entirely made of nans.
        """
        if self._zmax is not None:
            return self._zmax
        not_empty_arrays = [array for array in arrays if ~torch.isnan(array).all()]
        if not not_empty_arrays:
            return torch.nan
        not_empty_vals = [arr[~torch.isnan(arr)] for arr in not_empty_arrays]
        return max(torch.max(torch.abs(arr)) for arr in not_empty_vals).cpu().item()
