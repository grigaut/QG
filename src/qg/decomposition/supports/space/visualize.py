"""Visualize space params."""

from typing import Any

import numpy as np
from matplotlib import pyplot as plt


def plot_gaussian_space_params(space: dict[int, dict[str, Any]]) -> None:
    """Plot space centers.

    Args:
        space (dict[int, dict[str, Any]]): Space parameters dictionnary.
    """
    minmax_x = None
    minmax_y = None
    for v in space.values():
        xyc_p = v["centers"]
        xs = [x for x, _ in xyc_p]
        ys = [y for _, y in xyc_p]

        if minmax_x is None:
            minmax_x = (min(xs), max(xs))
        else:
            minmax_x = (min(*xs, minmax_x[0]), max(*xs, minmax_x[1]))
        if minmax_y is None:
            minmax_y = (min(ys), max(ys))
        else:
            minmax_y = (min(*ys, minmax_y[0]), max(*ys, minmax_y[1]))

    x = np.linspace(minmax_x[0], minmax_x[-1])
    y = np.linspace(minmax_y[0], minmax_y[-1])
    xx, yy = np.meshgrid(x, y, indexing="ij")

    for k, v in space.items():
        xyc_p = v["centers"]
        xs = [x for x, _ in xyc_p]
        ys = [y for _, y in xyc_p]
        sx = v["sigma_x"]
        sy = v["sigma_y"]

        exp_field = np.zeros_like(xx)

        for xc, yc in zip(xs, ys):
            exp_field += np.exp(
                -((xx - xc) ** 2) / sx**2 - (yy - yc) ** 2 / sy**2
            )

        cbar = plt.imshow(exp_field.T, extent=[x[0], x[-1], y[0], y[-1]])
        plt.colorbar(cbar)
        plt.scatter(xs, ys, label=f"{k}", c="red")

        plt.title(f"Order: {k}")
        plt.show()
