"""Visualize time supports."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_gaussian_time_params(time: dict[int, dict[str, Any]]) -> None:
    """Plot time centers.

    Args:
        time (dict[int, dict[str, Any]]): Time parameters.
    """
    minmax_t = None
    for v in time.values():
        tc_p = v["centers"]
        if minmax_t is None:
            minmax_t = (min(tc_p), max(tc_p))
        else:
            minmax_t = (min(*tc_p, minmax_t[0]), max(*tc_p, minmax_t[1]))
    t = np.linspace(minmax_t[0], minmax_t[1], 200)

    for k, v in time.items():
        tc_p = v["centers"]
        st = v["sigma_t"]
        exp_field = np.zeros_like(t)
        for tc in tc_p:
            exp_field = np.exp(-((t - tc) ** 2) / st**2)
            plt.plot(t, exp_field, zorder=0)
        for tc in tc_p:
            plt.scatter(tc, 0, zorder=1, marker="|", s=200)
        plt.hlines([0.5], [t[0]], [t[-1]], color="grey", linestyles="--")
        plt.legend()
        plt.title(f"Order: {k}")
        plt.show()
