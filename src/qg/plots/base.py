"""Base class for plots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import plotly.graph_objects as go

T = TypeVar("T")


class BasePlot(ABC, Generic[T]):
    """Base for all plots."""

    _is_set = False

    _xaxis_title = ""
    _yaxis_title = ""

    def __init__(
        self,
        datas: list[T],
    ) -> None:
        """Instantiate the plot.

        Args:
            datas (list[T]): List of datas to plot.
        """
        self._n_traces = len(datas)
        self._fig = self._create_figure()

    @property
    def figure(self) -> go.Figure:
        """Figure."""
        return self._fig

    @property
    def n_traces(self) -> int:
        """Number of traces."""
        return self._n_traces

    def _create_figure(self) -> go.Figure:
        """Create the Figure.

        Returns:
            go.Figure: Figure.
        """
        return go.Figure()

    def set_xaxis_title(self, text: str) -> None:
        """Set the xaxis title.

        Args:
            text (str): xaxis title.
        """
        self._xaxis_title = text

    def set_yaxis_title(self, text: str) -> None:
        """Set the yaxis title.

        Args:
            text (str): yaxis title.
        """
        self._yaxis_title = text

    def set_figure_size(self, width: int, height: int) -> None:
        """Set the figure size.

        Args:
            width (int): Figure width
            height (int): Figure geight
        """
        self.figure.update_layout(height=height, width=width)

    def _create_xaxis(self) -> go.layout.XAxis:
        return go.layout.XAxis(title=self._xaxis_title)

    def _create_yaxis(self) -> go.layout.YAxis:
        return go.layout.YAxis(title=self._yaxis_title)

    def _create_layout(self) -> go.Layout:
        return go.Layout()

    @abstractmethod
    def _add_traces(self) -> None:
        """Initialize the traces."""

    def _set_figure(self) -> None:
        """Set the figure traces."""
        if self._is_set:
            return
        self._is_set = True
        self._add_traces()
        self.figure.update_layout(self._create_layout())
        self.figure.update_xaxes(self._create_xaxis())
        self.figure.update_yaxes(self._create_yaxis())

    def show(self) -> None:
        """Show the Figure."""
        self.retrieve_figure().show()

    def retrieve_figure(self) -> go.Figure:
        """Retrieve the figure.

        Returns:
            go.Figure: Figure
        """
        self._set_figure()
        return self.figure
