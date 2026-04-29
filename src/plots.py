"""
plots.py
========
Shared Matplotlib style and drawing helpers for all project notebooks and
thesis figures.

Usage
-----
    from src.plots import apply_notebook_style   # or: apply_style

    apply_notebook_style()        # call once at the top of every notebook

    fig, ax = plt.subplots()
    draw_ellipse(ax, mean, cov, kind="prior")
    draw_hyperplane(ax, a, tau, xlim=(-4, 4))
    shade_slab(ax, a, tau_lo, tau_hi)
    draw_question_arrow(ax, origin, a)
    draw_mean(ax, mean, kind="prior")
    format_axes(ax, xlabel=r"$\\theta_1$", ylabel=r"$\\theta_2$")
    save(fig, "my_figure")
"""

from __future__ import annotations

import os
import sys
import textwrap
import warnings
from typing import Literal, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------------------
# 1.  SEMANTIC COLOR CONSTANTS
# ---------------------------------------------------------------------------

PRIOR_BLUE      = "#6B9EC7"   # muted blue  – prior / current belief
POSTERIOR_GREEN = "#2CA02C"   # green       – posterior / updated objects
STRUCTURE_GRAY  = "#9E9E9E"   # gray        – hyperplanes / thresholds
QUESTION_ORANGE = "#E87722"   # orange      – discrimination vector a_q
ACTIVE_BLACK    = "#1A1A1A"   # near-black  – active boundaries, true state, reference geometry
SLAB_BLUE       = PRIOR_BLUE  # slab fill shares the prior-blue semantic role

# Belief-update triad — prior → undamped → damped
BELIEF_PRIOR    = PRIOR_BLUE   # "#6B9EC7"  muted blue   — prior belief
BELIEF_UNDAMPED = "#7B6BC5"    #            violet       — full Bayesian update
BELIEF_DAMPED   = "#C9556E"    #            dusty rose   — dampened update

# Derived fill colors (used for filled regions at low opacity)
_FILL: dict[str, str] = {
    "prior"     : PRIOR_BLUE,
    "posterior" : POSTERIOR_GREEN,
    "slab"      : SLAB_BLUE,
    "reference" : ACTIVE_BLACK,
}

# Default opacities
_ALPHA: dict[str, float] = {
    "prior"     : 0.15,
    "posterior" : 0.22,
    "slab"      : 0.18,
    "reference" : 0.10,
}

# ---------------------------------------------------------------------------
# 2.  DESIGN TOKENS  (generic palette for notebook scatter / histogram plots)
# ---------------------------------------------------------------------------

# Named colour tokens that map notebook-chart roles to the thesis palette.
PALETTE: dict[str, str] = {
    "blue"  : PRIOR_BLUE,        # primary data colour (user clouds, item histograms)
    "teal"  : POSTERIOR_GREEN,   # secondary data colour (marginal θ₁)
    "red"   : "#C0392B",         # warning / emphasis (target means, errors)
    "amber" : QUESTION_ORANGE,   # warm accent (marginal θ₂, discrimination vector)
    "violet": "#7C3AED",         # fifth categorical slot
    # Chrome / structural
    "ink"   : ACTIVE_BLACK,      # prominent lines, annotations, arrow text
    "mist"  : STRUCTURE_GRAY,    # tick labels, subtle annotations
    "rule"  : "#BDBDBD",         # spine and divider strokes
    "grid"  : "#EFEFEF",         # background grid lines (when enabled)
    "bg"    : "white",           # figure background
}

# Ordered colour cycle for categorical series (prop_cycle default)
CYCLE: list[str] = [
    PRIOR_BLUE,
    POSTERIOR_GREEN,
    QUESTION_ORANGE,
    ACTIVE_BLACK,
    STRUCTURE_GRAY,
]

# ---------------------------------------------------------------------------
# 3.  INTERNAL CONSTANTS
# ---------------------------------------------------------------------------

_FONT_SIZE_LABEL  = 10
_FONT_SIZE_TICK   = 9
_FONT_SIZE_LEGEND = 9
_FONT_SIZE_TITLE  = 11

_LINE_WIDTHS: dict[str, float] = {
    "default"   : 1.4,
    "hyperplane": 0.9,
    "reference" : 0.9,
    "arrow"     : 1.8,
}

_MARKER_SIZES: dict[str, int] = {
    "mean" : 7,
    "state": 9,
}

OUTPUT_DIR = "figures"


# ---------------------------------------------------------------------------
# 4.  GLOBAL RCPARAMS
# ---------------------------------------------------------------------------

def apply_notebook_style(
    *,
    use_tex: bool = False,
    base_font_size: float = 11.0,
    figure_dpi: float = 120.0,
    save_dpi: float = 300.0,
    use_retina: bool = True,
) -> None:
    """
    Apply the shared project / thesis plotting style.

    Call once at the top of every notebook or script before creating any
    figure.

    Parameters
    ----------
    use_tex:
        If True, render all text with an external LaTeX installation.
        Requires LaTeX and is noticeably slower.  The default uses
        Matplotlib's internal mathtext (Computer-Modern-like, instant).
    base_font_size:
        Base font size used for most text.  Label and tick sizes are
        derived from this value.
    figure_dpi:
        DPI for interactive notebook rendering.
    save_dpi:
        DPI used when saving figures to disk (publication quality).
    use_retina:
        If True and running in a notebook with matplotlib-inline, request
        high-density inline rendering.
    """
    mpl.rcParams.update({
        # ── Font ────────────────────────────────────────────────────────────
        "font.family"          : "serif",
        "font.serif"           : ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.size"            : base_font_size,
        "mathtext.fontset"     : "cm",
        "axes.unicode_minus"   : False,
        "text.usetex"          : use_tex,
        # ── Axes ────────────────────────────────────────────────────────────
        "axes.spines.top"      : False,
        "axes.spines.right"    : False,
        "axes.linewidth"       : 0.8,
        "axes.edgecolor"       : ACTIVE_BLACK,
        "axes.labelsize"       : base_font_size,
        "axes.labelpad"        : 5.0,
        "axes.titlesize"       : _FONT_SIZE_TITLE,
        "axes.titlepad"        : 8.0,
        "axes.prop_cycle"      : mpl.cycler(color=CYCLE),
        # ── Ticks ───────────────────────────────────────────────────────────
        "xtick.labelsize"      : _FONT_SIZE_TICK,
        "ytick.labelsize"      : _FONT_SIZE_TICK,
        "xtick.direction"      : "out",
        "ytick.direction"      : "out",
        "xtick.color"          : ACTIVE_BLACK,
        "ytick.color"          : ACTIVE_BLACK,
        "xtick.labelcolor"     : ACTIVE_BLACK,
        "ytick.labelcolor"     : ACTIVE_BLACK,
        "xtick.major.size"     : 3.5,
        "ytick.major.size"     : 3.5,
        "xtick.major.width"    : 0.7,
        "ytick.major.width"    : 0.7,
        # ── Lines ───────────────────────────────────────────────────────────
        "lines.linewidth"      : _LINE_WIDTHS["default"],
        "lines.solid_capstyle" : "round",
        "patch.linewidth"      : 0.6,
        # ── Legend ──────────────────────────────────────────────────────────
        "legend.fontsize"      : _FONT_SIZE_LEGEND,
        "legend.frameon"       : True,
        "legend.framealpha"    : 0.92,
        "legend.edgecolor"     : "0.75",
        "legend.fancybox"      : False,
        "legend.borderpad"     : 0.5,
        "legend.labelspacing"  : 0.4,
        "legend.handlelength"  : 1.4,
        # ── Figure ──────────────────────────────────────────────────────────
        "figure.dpi"           : figure_dpi,
        "figure.facecolor"     : "white",
        # ── Saving ──────────────────────────────────────────────────────────
        "savefig.dpi"          : save_dpi,
        "savefig.bbox"         : "tight",
        "savefig.facecolor"    : "white",
        "savefig.pad_inches"   : 0.15,
    })

    if use_retina and _running_in_ipython_kernel():
        try:
            from matplotlib_inline.backend_inline import set_matplotlib_formats

            set_matplotlib_formats("retina")
        except Exception:
            pass


# Alias — use whichever name feels natural in context
apply_style = apply_notebook_style


def _running_in_ipython_kernel() -> bool:
    """Return True when running inside an IPython/Jupyter kernel."""
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return False
    shell = ipython.get_ipython()
    return bool(shell and "IPKernelApp" in shell.config)


# ---------------------------------------------------------------------------
# 5.  PER-AXES UTILITIES
# ---------------------------------------------------------------------------

def despine(
    ax: Axes,
    sides: Sequence[Literal["top", "right", "left", "bottom"]] = ("top", "right"),
) -> None:
    """Remove specific spines from *ax*."""
    for side in sides:
        ax.spines[side].set_visible(False)


def style_ax(
    ax: Axes,
    *,
    grid_axis: Literal["x", "y", "both", "none"] = "none",
    despine_sides: Sequence[str] = ("top", "right"),
) -> None:
    """
    Apply consistent cosmetic styling to a single Axes.

    Useful when a figure is created outside the global rcParams context, or
    when per-axes overrides are needed (e.g. turning the grid off on a square
    geometric scatter while leaving it on companion bar charts).

    Parameters
    ----------
    ax:
        Target axes.
    grid_axis:
        Axis on which to draw grid lines.  ``"none"`` hides the grid.
    despine_sides:
        Spine sides to remove.
    """
    despine(ax, sides=despine_sides)

    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.7)
        ax.set_axisbelow(True)

    ax.tick_params(colors=ACTIVE_BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACTIVE_BLACK)
        spine.set_linewidth(0.8)


def format_axes(
    ax: Axes,
    xlabel: str = r"$\theta_1$",
    ylabel: str = r"$\theta_2$",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    equal: bool = True,
    spine_color: str = ACTIVE_BLACK,
    legend_loc: str = "upper right",
) -> None:
    """
    Apply consistent formatting for 2-D latent-space figures.

    Adds arrowheads to the ends of the bottom and left spines so the axes
    read as directed coordinate frames.  Positions and re-applies the legend
    if one already exists.

    Parameters
    ----------
    equal:
        If True, enforce equal aspect ratio — recommended for any figure
        that shows geometric objects in latent space.
    legend_loc:
        Passed to ``ax.legend()`` when a legend is already attached.
    """
    ax.set_xlabel(xlabel, fontsize=_FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=_FONT_SIZE_LABEL)

    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.8)

    ax.tick_params(colors=spine_color, which="both")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")

    # Arrowheads pinned to the spine tips via transAxes so they survive
    # any subsequent aspect-ratio or limit adjustment.
    ax.plot(1, 0, marker=">", markersize=4.5, color=spine_color,
            clip_on=False, zorder=10, transform=ax.transAxes,
            markeredgewidth=0, linestyle="none")
    ax.plot(0, 1, marker="^", markersize=4.5, color=spine_color,
            clip_on=False, zorder=10, transform=ax.transAxes,
            markeredgewidth=0, linestyle="none")

    if ax.get_legend() is not None:
        ax.legend(loc=legend_loc, fontsize=_FONT_SIZE_LEGEND)


# ---------------------------------------------------------------------------
# 6.  DRAWING HELPERS — Gaussian credibility ellipse
# ---------------------------------------------------------------------------

def draw_ellipse(
    ax: Axes,
    mean: Sequence[float],
    cov: np.ndarray,
    kind: Literal["prior", "posterior", "reference"] = "prior",
    confidence: float = 0.90,
    n_std: Optional[float] = None,
    label: Optional[str] = None,
    zorder: int = 2,
) -> Ellipse:
    """
    Draw a Gaussian credibility ellipse.

    Parameters
    ----------
    mean:
        (2,) centre of the ellipse.
    cov:
        (2, 2) positive-definite covariance matrix.
    kind:
        ``"prior"``      → blue edge + fill
        ``"posterior"``  → green edge + fill
        ``"reference"``  → near-black edge + fill
    confidence:
        Coverage level ρ.  Ignored when *n_std* is given.
    n_std:
        If provided, scale the ellipse to this many standard deviations
        instead of using a chi-squared quantile.
    """
    mean = np.asarray(mean, dtype=float)
    cov  = np.asarray(cov,  dtype=float)

    vals, vecs = np.linalg.eigh(cov)
    order  = vals.argsort()[::-1]
    vals   = vals[order]
    vecs   = vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    if n_std is None:
        from scipy.stats import chi2

        r = np.sqrt(chi2.ppf(confidence, df=2))
    else:
        r = n_std
    width, height = 2 * r * np.sqrt(np.maximum(vals, 0))

    color = _FILL[kind]
    alpha = _ALPHA[kind]

    ell = Ellipse(
        xy=(mean[0], mean[1]),
        width=width, height=height, angle=angle,
        edgecolor=color, facecolor=color,
        linewidth=_LINE_WIDTHS["default"],
        alpha=alpha, zorder=zorder,
        label=label,
    )
    ax.add_patch(ell)

    # Solid edge on top — separate patch so opacity doesn't wash out the boundary
    edge = Ellipse(
        xy=(mean[0], mean[1]),
        width=width, height=height, angle=angle,
        edgecolor=color, facecolor="none",
        linewidth=_LINE_WIDTHS["default"],
        zorder=zorder + 0.1,
    )
    ax.add_patch(edge)
    return ell


# ---------------------------------------------------------------------------
# 7.  DRAWING HELPERS — threshold hyperplanes  (2-D)
# ---------------------------------------------------------------------------

def draw_hyperplane(
    ax: Axes,
    a: Sequence[float],
    tau: float,
    xlim: Tuple[float, float] = (-4.0, 4.0),
    active: bool = False,
    label: Optional[str] = None,
    zorder: int = 3,
) -> None:
    """
    Draw the line  ``a^T θ = tau``  in a 2-D latent-space figure.

    Parameters
    ----------
    a:
        (2,) discrimination / normal vector.
    tau:
        Threshold value.
    xlim:
        x-range used to clip the line.
    active:
        If True, draw in near-black (slab boundary); otherwise in gray.
    """
    a = np.asarray(a, dtype=float)
    x0, x1 = xlim

    if abs(a[1]) > 1e-9:
        xs = np.array([x0, x1])
        ys = (tau - a[0] * xs) / a[1]
    else:
        x_val = tau / a[0]
        y_range = ax.get_ylim() if ax.get_ylim() != (0, 1) else (-4, 4)
        xs = np.array([x_val, x_val])
        ys = np.array([y_range[0], y_range[1]])

    color = ACTIVE_BLACK if active else STRUCTURE_GRAY
    ax.plot(xs, ys, color=color, linewidth=_LINE_WIDTHS["hyperplane"],
            linestyle="-", zorder=zorder, label=label)


def draw_hyperplanes(
    ax: Axes,
    a: Sequence[float],
    taus: Sequence[float],
    xlim: Tuple[float, float] = (-4.0, 4.0),
    active_indices: Optional[Sequence[int]] = None,
) -> None:
    """
    Draw a family of parallel threshold hyperplanes for one item.

    Parameters
    ----------
    taus:
        Ordered thresholds τ₁ … τₘ.
    active_indices:
        Indices of the two boundaries that bound the observed slab;
        those are drawn in black.  All others are drawn in gray.
    """
    active_set = set(active_indices) if active_indices is not None else set()
    for i, tau in enumerate(taus):
        draw_hyperplane(ax, a, tau, xlim=xlim, active=(i in active_set))


# ---------------------------------------------------------------------------
# 8.  DRAWING HELPERS — slab fill  (2-D)
# ---------------------------------------------------------------------------

def _clip_polygon_halfspace(
    polygon: list[tuple[float, float]],
    nx: float,
    ny: float,
    b: float,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a convex polygon by ``nx*x + ny*y >= b``."""
    if not polygon:
        return []
    output: list[tuple[float, float]] = []
    n = len(polygon)
    for i in range(n):
        curr = polygon[i]
        prev = polygon[i - 1]
        c_in = nx * curr[0] + ny * curr[1] >= b
        p_in = nx * prev[0] + ny * prev[1] >= b
        if c_in:
            if not p_in:
                dx, dy = curr[0] - prev[0], curr[1] - prev[1]
                denom = nx * dx + ny * dy
                if abs(denom) > 1e-12:
                    t = (b - (nx * prev[0] + ny * prev[1])) / denom
                    output.append((prev[0] + t * dx, prev[1] + t * dy))
            output.append(curr)
        elif p_in:
            dx, dy = curr[0] - prev[0], curr[1] - prev[1]
            denom = nx * dx + ny * dy
            if abs(denom) > 1e-12:
                t = (b - (nx * prev[0] + ny * prev[1])) / denom
                output.append((prev[0] + t * dx, prev[1] + t * dy))
    return output


def shade_slab(
    ax: Axes,
    a: Sequence[float],
    tau_lo: float,
    tau_hi: float,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    label: Optional[str] = None,
    zorder: int = 1,
) -> None:
    """
    Shade the slab  ``tau_lo ≤ a^T θ ≤ tau_hi``  in a 2-D latent-space figure.

    The polygon is clipped only against the two hyperplane boundaries;
    Matplotlib's axes frame handles all remaining clipping.  This ensures the
    fill always reaches the frame edge on all sides regardless of aspect-ratio
    adjustments.

    Parameters
    ----------
    xlim, ylim:
        Kept for API compatibility; no longer used (clipping is unlimited).
    """
    a = np.asarray(a, dtype=float)
    if np.linalg.norm(a) < 1e-12:
        warnings.warn("shade_slab: zero-norm vector a, skipping.")
        return

    R = 1e4
    poly: list[tuple[float, float]] = [(-R, -R), (R, -R), (R, R), (-R, R)]
    poly = _clip_polygon_halfspace(poly,  a[0],  a[1],  tau_lo)
    poly = _clip_polygon_halfspace(poly, -a[0], -a[1], -tau_hi)

    if len(poly) < 3:
        return

    patch = mpatches.Polygon(
        poly, closed=True,
        facecolor=SLAB_BLUE, edgecolor="none",
        alpha=_ALPHA["slab"], zorder=zorder,
        label=label, clip_on=True,
    )
    ax.add_patch(patch)


# ---------------------------------------------------------------------------
# 9.  DRAWING HELPERS — discrimination-direction arrow
# ---------------------------------------------------------------------------

def draw_question_arrow(
    ax: Axes,
    origin: Sequence[float],
    a: Sequence[float],
    scale: float = 1.0,
    label: Optional[str] = None,
    zorder: int = 4,
) -> None:
    """
    Draw the discrimination vector ``a_q`` as an orange arrow from *origin*.

    Parameters
    ----------
    origin:
        (2,) start point of the arrow.
    a:
        (2,) direction.  The tip lands at ``origin + scale * a / ||a||``.
    scale:
        Arrow length in data units.
    """
    origin = np.asarray(origin, dtype=float)
    a      = np.asarray(a, dtype=float)
    a_hat  = a / (np.linalg.norm(a) + 1e-15)
    dx, dy = scale * a_hat

    ax.annotate(
        "",
        xy=(origin[0] + dx, origin[1] + dy),
        xytext=(origin[0], origin[1]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=QUESTION_ORANGE,
            lw=_LINE_WIDTHS["arrow"],
            mutation_scale=14,
        ),
        zorder=zorder,
    )

    if label is not None:
        ax.annotate(
            label,
            xy=(origin[0] + dx * 1.08, origin[1] + dy * 1.08),
            color=QUESTION_ORANGE,
            fontsize=_FONT_SIZE_LABEL,
            ha="center", va="center",
            zorder=zorder,
        )


# ---------------------------------------------------------------------------
# 10.  DRAWING HELPERS — mean / latent-state markers
# ---------------------------------------------------------------------------

def draw_mean(
    ax: Axes,
    point: Sequence[float],
    kind: Literal["prior", "posterior", "true"] = "prior",
    label: Optional[str] = None,
    zorder: int = 5,
) -> None:
    """
    Draw a mean or true-state marker.

    ``kind="prior"``      → blue filled dot
    ``kind="posterior"``  → green filled dot
    ``kind="true"``       → black star (visually distinct from belief means)
    """
    point = np.asarray(point, dtype=float)

    if kind == "true":
        ax.plot(point[0], point[1],
                marker="*", color=ACTIVE_BLACK, markersize=_MARKER_SIZES["state"],
                linestyle="none", zorder=zorder, label=label)
    else:
        color = PRIOR_BLUE if kind == "prior" else POSTERIOR_GREEN
        ax.plot(point[0], point[1],
                marker="o", color=color, markersize=_MARKER_SIZES["mean"],
                linestyle="none", zorder=zorder, label=label,
                markeredgecolor="white", markeredgewidth=0.8)


# ---------------------------------------------------------------------------
# 11.  DRAWING HELPERS — reference geometry
# ---------------------------------------------------------------------------

def draw_reference_circle(
    ax: Axes,
    center: Sequence[float] = (0.0, 0.0),
    radius: float = 1.0,
    label: Optional[str] = None,
    zorder: int = 1,
) -> None:
    """Draw a dashed near-black reference circle (e.g. unit sphere)."""
    theta = np.linspace(0, 2 * np.pi, 300)
    cx, cy = center
    ax.plot(
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        color=ACTIVE_BLACK, linewidth=_LINE_WIDTHS["reference"],
        linestyle="--", zorder=zorder, label=label,
    )


def draw_reference_line(
    ax: Axes,
    slope: float = 0.0,
    intercept: float = 0.0,
    xlim: Tuple[float, float] = (-4.0, 4.0),
    label: Optional[str] = None,
    zorder: int = 1,
) -> None:
    """Draw a dashed near-black reference line  ``y = slope * x + intercept``."""
    xs = np.array(xlim)
    ax.plot(
        xs, slope * xs + intercept,
        color=ACTIVE_BLACK, linewidth=_LINE_WIDTHS["reference"],
        linestyle="--", zorder=zorder, label=label,
    )


# ---------------------------------------------------------------------------
# 12.  LEGEND UTILITIES
# ---------------------------------------------------------------------------

def prior_patch(label: str = "Prior belief") -> mpatches.Patch:
    return mpatches.Patch(facecolor=PRIOR_BLUE, alpha=0.5, label=label)


def posterior_patch(label: str = "Posterior belief") -> mpatches.Patch:
    return mpatches.Patch(facecolor=POSTERIOR_GREEN, alpha=0.5, label=label)


def slab_patch(label: str = "Observed slab") -> mpatches.Patch:
    return mpatches.Patch(facecolor=SLAB_BLUE, alpha=0.5, label=label)


def true_state_handle(label: str = r"True $\theta^*$") -> mlines.Line2D:
    """Legend handle matching the black star marker used by ``draw_mean(..., kind='true')``."""
    return mlines.Line2D([], [],
                         marker="*", color=ACTIVE_BLACK,
                         markersize=_MARKER_SIZES["state"] + 1,
                         linestyle="none", label=label)


def prior_mean_handle(label: str = r"Prior mean $\mu_0$") -> mlines.Line2D:
    """Legend handle matching the blue dot used by ``draw_mean(..., kind='prior')``."""
    return mlines.Line2D([], [],
                         marker="o", color=PRIOR_BLUE,
                         markersize=_MARKER_SIZES["mean"],
                         markeredgecolor="white", markeredgewidth=0.8,
                         linestyle="none", label=label)


def posterior_mean_handle(label: str = r"Posterior mean $\mu_1$") -> mlines.Line2D:
    """Legend handle matching the green dot used by ``draw_mean(..., kind='posterior')``."""
    return mlines.Line2D([], [],
                         marker="o", color=POSTERIOR_GREEN,
                         markersize=_MARKER_SIZES["mean"],
                         markeredgecolor="white", markeredgewidth=0.8,
                         linestyle="none", label=label)


def _wrap_label(label: str, width: int) -> str:
    return textwrap.fill(label, width=width,
                         break_long_words=False, break_on_hyphens=False)


def custom_marker_legend(
    ax: Axes,
    entries: list[dict],
    loc: str = "upper right",
    wrap_width: int = 28,
) -> None:
    """
    Build a legend whose markers and lines exactly match the plotted symbols.

    Labels longer than *wrap_width* characters are wrapped to a second line.

    Each entry is a dict with optional keys:
    ``label`` (required), ``color``, ``marker``, ``linestyle``,
    ``markersize``, ``linewidth``.

    Example
    -------
    >>> custom_marker_legend(ax, [
    ...     {"label": "Prior",       "color": PRIOR_BLUE,    "marker": "o"},
    ...     {"label": "True state",  "color": ACTIVE_BLACK,  "marker": "*",
    ...      "markersize": 10},
    ...     {"label": "Boundary",    "color": ACTIVE_BLACK,  "marker": "none",
    ...      "linestyle": "-",       "linewidth": 0.9},
    ... ])
    """
    handles = [
        mlines.Line2D(
            [0], [0],
            color=e.get("color", ACTIVE_BLACK),
            marker=e.get("marker", "o"),
            linestyle=e.get("linestyle", "none"),
            markersize=e.get("markersize", _MARKER_SIZES["mean"]),
            linewidth=e.get("linewidth", _LINE_WIDTHS["default"]),
            label=_wrap_label(e["label"], wrap_width),
        )
        for e in entries
    ]
    ax.legend(handles=handles, loc=loc, fontsize=_FONT_SIZE_LEGEND)


def sync_legend_widths(fig: plt.Figure, extra_pad: float = 55.0) -> None:
    """
    Set all legend frames in *fig* to the same width as the widest one,
    then add *extra_pad* points of horizontal padding.

    Call immediately before saving:

    >>> sync_legend_widths(fig)
    >>> fig.savefig("my_figure.pdf")
    """
    fig.canvas.draw()
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    if not legends:
        return
    max_w = max(leg.get_frame().get_width() for leg in legends)
    for leg in legends:
        leg.get_frame().set_width(max_w + extra_pad)


# ---------------------------------------------------------------------------
# 13.  EXPORT
# ---------------------------------------------------------------------------

def save(
    fig: plt.Figure,
    name: str,
    directory: str = OUTPUT_DIR,
    formats: Sequence[str] = ("pdf", "png"),
) -> None:
    """
    Save *fig* to ``<directory>/<name>.<ext>`` for every format in *formats*.

    Creates *directory* if it does not exist.
    """
    os.makedirs(directory, exist_ok=True)
    for fmt in formats:
        path = os.path.join(directory, f"{name}.{fmt}")
        fig.savefig(path)
        print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# 14.  SELF-CHECK CHECKLIST  (printed when module is run directly)
# ---------------------------------------------------------------------------

_CHECKLIST = """\
──────────────────────────────────────────────────────────
Figure Checklist  (src/plots.py)
──────────────────────────────────────────────────────────
 □  apply_notebook_style() called at the top of this script/notebook
 □  Thesis font settings applied (mathtext.fontset = "cm")
 □  Symbols match chapter notation
 □  Light blue  (PRIOR_BLUE)      → prior / current belief or slab fill
 □  Green       (POSTERIOR_GREEN) → updated / posterior objects
 □  Orange      (QUESTION_ORANGE) → a_q-type directional objects
 □  Gray        (STRUCTURE_GRAY)  → hyperplanes and thresholds
 □  Black       (ACTIVE_BLACK)    → active boundaries, true state, reference geometry
 □  One clear main message per figure
 □  Caption: first sentence descriptive, second sentence interpretive
 □  Figure looks consistent with surrounding thesis figures
──────────────────────────────────────────────────────────
"""

if __name__ == "__main__":
    print(_CHECKLIST)

    apply_notebook_style()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("src/plots.py — style demo (not a thesis figure)",
                 fontsize=10, color=ACTIVE_BLACK)

    # ── LEFT: Belief update ───────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Belief update geometry", fontsize=_FONT_SIZE_TITLE)

    prior_cov = np.array([[2.2, 0.6], [0.6, 1.0]])
    post_cov  = np.array([[0.8, 0.15], [0.15, 0.45]])
    mu_prior  = np.array([0.0, 0.0])
    mu_post   = np.array([1.1, 0.5])

    draw_ellipse(ax, mu_prior, prior_cov, kind="prior")
    draw_ellipse(ax, mu_post,  post_cov,  kind="posterior")
    draw_mean(ax, mu_prior, kind="prior")
    draw_mean(ax, mu_post,  kind="posterior")
    draw_mean(ax, [1.4, 0.9], kind="true")
    draw_question_arrow(ax, origin=[-1.5, -1.5], a=[1.0, 0.5],
                        scale=1.2, label=r"$a_q$")
    format_axes(ax, xlim=(-3.5, 3.5), ylim=(-3.0, 3.0))
    custom_marker_legend(ax, [
        {"label": "Prior ellipsoid",        "color": PRIOR_BLUE,      "marker": "s", "markersize": 9},
        {"label": "Posterior ellipsoid",     "color": POSTERIOR_GREEN, "marker": "s", "markersize": 9},
        {"label": r"Prior mean $\mu_0$",     "color": PRIOR_BLUE,      "marker": "o"},
        {"label": r"Posterior mean $\mu_1$", "color": POSTERIOR_GREEN, "marker": "o"},
        {"label": r"True $\theta^*$",        "color": ACTIVE_BLACK,    "marker": "*",
         "markersize": _MARKER_SIZES["state"] + 1},
    ])

    # ── RIGHT: Slab / threshold ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Slab and threshold hyperplanes", fontsize=_FONT_SIZE_TITLE)

    a_q = np.array([1.0, 0.5])
    draw_hyperplanes(ax2, a_q, taus=[-2.0, 0.0, 2.0],
                     xlim=(-4, 4), active_indices=[1, 2])
    shade_slab(ax2, a_q, tau_lo=0.0, tau_hi=2.0)
    draw_question_arrow(ax2, origin=[2.5, -2.5], a=a_q, scale=1.2, label=r"$a_q$")
    format_axes(ax2, xlim=(-4, 4), ylim=(-3.5, 3.5))
    custom_marker_legend(ax2, [
        {"label": "Observed slab",      "color": SLAB_BLUE,     "marker": "s", "markersize": 9},
        {"label": "Active boundaries",  "color": ACTIVE_BLACK,  "marker": "none",
         "linestyle": "-", "linewidth": _LINE_WIDTHS["hyperplane"]},
        {"label": "Other thresholds",   "color": STRUCTURE_GRAY,"marker": "none",
         "linestyle": "-", "linewidth": _LINE_WIDTHS["hyperplane"]},
    ])

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    sync_legend_widths(fig)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/plots_demo.png", dpi=150)
    print("Demo figure saved → figures/plots_demo.png")
    plt.show()
