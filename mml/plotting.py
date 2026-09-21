"""Consistent matplotlib helpers.

Deliberately thin at setup. The plotting vocabulary this repo needs is built up as the chapters
demand it, and each helper is promoted here from a chapter's ``src/`` once it has been used twice:

* **Ch 2–3** — ``plot_vectors`` (2-D arrows from the origin), ``plot_subspace`` (a line or plane
  through the origin with its orthogonal complement), ``plot_projection`` (a point, its image
  under a projector, and the residual).
* **Ch 4** — ``plot_spectrum`` (singular or eigenvalue decay, log axis, cumulative variance).
* **Ch 4, 6, 10** — ``heatmap`` (a matrix with a diverging, zero-centred colour map).

Everything here should take an optional ``ax`` and return it, so plots compose into subplots.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

__all__ = ["use_repo_style", "new_axes"]

#: Small, readable defaults. Applied by :func:`use_repo_style`, not on import.
REPO_STYLE: dict[str, Any] = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 110,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "font.size": 11,
    "legend.frameon": False,
    "image.cmap": "RdBu_r",
}


def use_repo_style() -> None:
    """Apply the repo's matplotlib defaults. Call once near the top of a notebook."""
    plt.rcParams.update(REPO_STYLE)


def new_axes(ax: plt.Axes | None = None, **kwargs: Any) -> plt.Axes:
    """Return ``ax`` if given, otherwise create a new figure and axes.

    The pattern every helper in this module should follow, so that a plot can be drawn standalone
    or into a subplot grid without a second code path.

    Args:
        ax: an existing axes, or ``None`` to create one.
        **kwargs: forwarded to :func:`matplotlib.pyplot.subplots`.

    Returns:
        The axes to draw on.
    """
    if ax is not None:
        return ax
    _, ax = plt.subplots(**kwargs)
    return ax
