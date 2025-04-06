from __future__ import annotations

import problem
import numpy as np

from matplotlib import pyplot as plt


def draw_arrow(axes: plt.Axes, begin: tuple[int, int], end: tuple[int, int]) -> None:
    (begin_y, begin_x), (end_y, end_x) = begin, end
    delta_x, delta_y = end_x - begin_x, end_y - begin_y
    axes.arrow(
        begin_x + 0.5,
        begin_y + 0.5,
        delta_x,
        delta_y,
        length_includes_head=True,
        head_width=0.8,
        head_length=0.8,
        fc="r",
        ec="r",
    )


def draw_episode(track: np.ndarray, positions: list[problem.Position], episode: int, directory: str) -> None:
    ax = plt.axes()
    ax.imshow(track)
    for i in range(len(positions) - 1):
        begin, end = positions[i], positions[i + 1]
        draw_arrow(ax, begin, end)
    plt.savefig(f"{directory}/track_{episode}.png", dpi=300)
    plt.clf()


def draw_penalties_plot(penalties: list[int], window_size: int, episode: int, directory: str) -> None:
    means = [np.mean(penalties[i : i + window_size]) for i in range(len(penalties) - window_size)]
    ax = plt.axes()
    ax.plot(means)
    plt.savefig(f"{directory}/penalties_{episode}.png", dpi=300)
    plt.clf()
