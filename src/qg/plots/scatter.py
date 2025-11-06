"""Line plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from qg.plots.base import BasePlot


class ScatterPlot(BasePlot[np.ndarray]):
    """Line plot."""

    def __init__(self, datas: list[np.ndarray]) -> None:
        """Instantiate the plot.

        Args:
            datas (list[np.ndarray]): List of data to use.
        """
        super().__init__(datas)
        self._datas = datas
        self._traces_name = [None for _ in range(self.n_traces)]
        self._xs = [list(range(len(d))) for d in self._datas]

    def set_xs(self, *xs: list[Any]) -> None:
        """Set the xs of thetraces.

        Args:
            *xs (list[Any]): List of x values.
        """
        if any(len(xs[k]) != len(d) for k, d in enumerate(self._datas)):
            msg = "xs's lengths must matchs the lengths of datas."
            raise ValueError(msg)
        self._xs = xs

    def _add_traces(self) -> None:
        for trace_nb in range(self.n_traces):
            self.figure.add_trace(
                go.Scatter(
                    x=self._xs[trace_nb],
                    y=self._datas[trace_nb],
                    name=self._traces_name[trace_nb],
                ),
            )

    def set_traces_name(self, *traces_name: str) -> None:
        """Set the name of the traces.

        Args:
            *traces_name (str): List of names.

        Raises:
            ValueError: If the number of names
            doesn't match the number of traces.
        """
        if len(traces_name) != self.n_traces:
            msg = f"There must be {self.n_traces} names."
            raise ValueError(msg)
        self._traces_name = traces_name
