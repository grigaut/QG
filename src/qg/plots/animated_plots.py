"""Base plots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Generic, TypeVar

import imageio
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from qg.plots.base import BasePlot

T = TypeVar("T")


class BaseAnimatedPlot(BasePlot, ABC, Generic[T]):
    """Animated Plot base class."""

    _slider_prefix: str = "Frame: "
    frame_duration = 300
    transition_duration = 500

    def __init__(
        self,
        datas: list[list[T]],
    ) -> None:
        """Instantiate the plot.

        Args:
            datas (list[list[T]]): Data list.
            frame_labels (list[str] | None, optional): Frames names.
            Defaults to None.

        Raises:
            ValueError: If frame_labels is incompatible with datas.
        """
        self._check_lengths(datas)
        super().__init__(datas=datas)
        self._n_frames = len(datas[0])
        # self._datas groups frames number and datas at the given step.
        self._datas = list(
            zip([str(i) for i in range(self.n_frames)], list(zip(*datas))),
        )

    @property
    def n_subplots(self) -> int:
        """Number of subplots."""
        return self.n_traces

    @cached_property
    def n_rows(self) -> int:
        """Number of rows."""
        return (self.n_subplots - 1) // self.n_cols + 1

    @cached_property
    def n_cols(self) -> int:
        """Number of columns."""
        return min(3, self.n_subplots)

    @cached_property
    def n_frames(self) -> int:
        """Number of steps."""
        return self._n_frames

    def _check_lengths(self, datas: list[T]) -> None:
        """Check than the data length are matching."""
        n_frames = len(datas[0])
        if all(len(data) == n_frames for data in datas):
            return
        msg = "Different lengths for datas."
        raise ValueError(msg)

    def _create_figure(
        self,
    ) -> go.Figure:
        """Create the Figure.

        Returns:
            go.Figure: Figure.
        """
        self.set_subplot_titles(None)
        return self._fig

    def set_slider_prefix(self, text: str) -> None:
        """Set the slider prefix.

        Args:
            text (str): Slider prefix.
        """
        self._slider_prefix = text

    def set_subplot_titles(self, subplot_titles: list[str]) -> None:
        """Set the subplot titles.

        Args:
            subplot_titles (list[str]): List of subplot titles.
        """
        self._fig = make_subplots(
            rows=self.n_rows,
            cols=self.n_cols,
            subplot_titles=subplot_titles,
        )

    def map_subplot_index_to_subplot_loc(
        self,
        subplot_index: int,
    ) -> tuple[int, int]:
        """Convert subplot index to subplot location.

        Args:
            subplot_index (int): Subplot index.

        Returns:
            tuple[int, int]: Row and column location within the plot.
        """
        return (subplot_index // self.n_cols + 1, subplot_index % 3 + 1)

    def map_subplot_loc_to_subplot_index(self, loc: tuple[int, int]) -> int:
        """Convert subplot location to subplot index.

        Args:
            loc (tuple[int, int]): Row and column location within the plot.

        Returns:
            int: Subplot index.
        """
        row, col = loc
        return (row - 1) * self.n_cols + (col - 1)

    def _create_play_button(self) -> go.layout.updatemenu.Button:
        """Create Play button.

        Returns:
            go.layout.updatemenu.Button: Play button.
        """
        return go.layout.updatemenu.Button(
            label="Play",
            method="animate",
            args=[
                None,
                {
                    "frame": {
                        "duration": self.frame_duration,
                        "redraw": True,
                    },
                    "fromcurrent": True,
                    "transition": {
                        "duration": self.transition_duration,
                        "easing": "quadratic-in-out",
                    },
                },
            ],
        )

    def _create_pause_button(self) -> go.layout.updatemenu.Button:
        """Create Pause button.

        Returns:
            go.layout.updatemenu.Button: Pause button.
        """
        return go.layout.updatemenu.Button(
            label="Pause",
            method="animate",
            args=[
                [None],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        )

    def _create_update_menu(self) -> go.layout.Updatemenu:
        """Create update menu.

        Returns:
            go.layout.Updatemenu: Update Menu.
        """
        return go.layout.Updatemenu(
            type="buttons",
            visible=True,
            buttons=[
                self._create_play_button(),
                self._create_pause_button(),
            ],
            direction="left",
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.1,
            xanchor="right",
            y=0,
            yanchor="top",
        )

    def _create_layout(self) -> go.Layout:
        """Create the layout.

        Returns:
            go.Layout: Layout.
        """
        layout = super()._create_layout()
        layout.update(
            sliders=[self._create_slider()],
            updatemenus=[self._create_update_menu()],
        )
        return layout

    def _create_slider_current_value(self) -> go.layout.slider.Currentvalue:
        """Create Slider current value.

        Returns:
            go.layout.slider.Currentvalue: Slider current value.
        """
        return go.layout.slider.Currentvalue(
            font={"size": 20},
            prefix=self._slider_prefix,
            visible=True,
            xanchor="center",
        )

    def _create_slider_transition(self) -> go.layout.slider.Transition:
        """Create Slider transition.

        Returns:
            go.layout.slider.Transition: Slider Transition.
        """
        return go.layout.slider.Transition(
            duration=self.transition_duration,
            easing="cubic-in-out",
        )

    def _create_slider(
        self,
    ) -> go.layout.Slider:
        """Create the slider.

        Returns:
            go.layout.Slider: Slider.
        """
        return go.layout.Slider(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=self._create_slider_current_value(),
            transition=self._create_slider_transition(),
            pad={"b": 10, "t": 50},
            len=0.9,
            x=0.1,
            y=0,
            steps=self._generate_steps(),
        )

    def _generate_frames(self) -> list[go.Frame]:
        """Generate the frames.

        Returns:
            list[dict]: Frames list.
        """
        return [self._compute_frame(frame) for frame in range(self.n_frames)]

    def _generate_steps(self) -> list[go.layout.slider.Step]:
        """Generate the steps.

        Returns:
            list[go.layout.slider.Step]: List of steps.
        """
        return [self._compute_step(frame) for frame in range(self.n_frames)]

    @abstractmethod
    def _compute_frame(self, frame_index: int) -> go.Frame:
        """Compute a frame at a given index.

        Args:
            frame_index (int): Frame index.

        Returns:
            go.Frame: Frame
        """

    @abstractmethod
    def _compute_step(self, frame_index: int) -> go.Frame:
        """Compute a step at a given index.

        Args:
            frame_index (int): Frame index.

        Returns:
            go.Frame: Step.
        """

    def _set_figure(self) -> None:
        """Set the figure traces."""
        super()._set_figure()
        self.figure.update(frames=self._generate_frames())

    def set_frame_labels(self, frame_labels: list[str]) -> None:
        """Set the frames labels.

        Args:
            frame_labels (list[str]): List of frame labels.

        Raises:
            ValueError: If the number of label is invalid.
        """
        if len(frame_labels) != self._n_frames:
            msg = (
                "Frames numbers must contains as many "
                f"value as datas, hence {self._n_frames}"
            )
            raise ValueError(msg)
        if len(set(frame_labels)) != self._n_frames:
            msg = "frame_labels must contain unique values."
        self._datas: list[tuple[str, list[T]]] = [
            (frame_labels[i], e[1]) for i, e in enumerate(self._datas)
        ]

    def save_frame(self, frame_index: int, output_folder: Path) -> None:
        """Save a given frame.

        Args:
            frame_index (int): Frame index.
            output_folder (Path): Folder to save output in.
        """
        self._set_figure()
        figure = self._create_figure()
        figure.update_xaxes(self._create_xaxis())
        figure.update_yaxes(self._create_yaxis())
        frame = self._compute_frame(frame_index)
        rows_cols = [
            self.map_subplot_index_to_subplot_loc(i) for i in range(self.n_subplots)
        ]
        figure.add_traces(
            frame.data,
            rows=[row for row, _ in rows_cols],
            cols=[col for _, col in rows_cols],
        )
        figure.write_image(
            output_folder.joinpath(
                f"frame_{self._datas[frame_index][0]}.png",
            ),
        )

    def save_video(
        self,
        output_file: str | Path,
        *,
        width: int = 500,
        height: int = 500,
        fps: int = 25,
    ) -> None:
        """Save the animated plot as a video.

        Args:
            output_file (str | Path): Output location.
            width (int, optional): Video width. Defaults to 500.
            height (int, optional): Video height. Defaults to 500.
            fps (int, optional): Frame per second. Defaults to 25.

        Raises:
            ValueError: If the output file is neither .gif or .mp4.
        """
        output = Path(output_file)
        if output.suffix not in [".gif", ".mp4"]:
            msg = (
                f"Unusuported output given ('{output.suffix}') type, "
                "only .gif or .mp4 are supported."
            )
            raise ValueError(msg)

        figure = self._create_figure()
        figure.update_xaxes(self._create_xaxis())
        figure.update_yaxes(self._create_yaxis())
        rows_cols = [
            self.map_subplot_index_to_subplot_loc(i) for i in range(self.n_subplots)
        ]
        frames = self._generate_frames()
        images = []
        for i, frame in enumerate(frames):
            # Apply frame data to the base figure
            figure.add_traces(
                frame.data,
                rows=[row for row, _ in rows_cols],
                cols=[col for _, col in rows_cols],
            )
            figure.update_layout(title=self._datas[i][0])
            # Render frame to image
            img_bytes = pio.to_image(
                figure,
                format="png",
                width=width,
                height=height,
                engine="kaleido",
            )
            images.append(imageio.v2.imread(img_bytes))
            figure.data = []
        imageio.v2.mimwrite(output, images, fps=fps)
