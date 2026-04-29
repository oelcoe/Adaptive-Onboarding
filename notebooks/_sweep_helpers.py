"""Shared helpers for sweep-analysis notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.plots import (  # noqa: E402
    ACTIVE_BLACK,
    PALETTE,
    POSTERIOR_GREEN,
    PRIOR_BLUE,
    QUESTION_ORANGE,
    STRUCTURE_GRAY,
    apply_notebook_style,
    save,
    style_ax,
)


POLICY_STYLE: dict[str, dict[str, Any]] = {
    "fixed": {
        "color": STRUCTURE_GRAY,
        "ls": "--",
        "marker": "s",
        "label": "Fixed",
    },
    "random": {
        "color": PRIOR_BLUE,
        "ls": "--",
        "marker": "^",
        "label": "Random",
    },
    "myopic_exact": {
        "color": ACTIVE_BLACK,
        "ls": "-",
        "marker": "D",
        "label": "Myopic (exact)",
    },
    "surrogate_unweighted": {
        "color": QUESTION_ORANGE,
        "ls": "-",
        "marker": "o",
        "label": "Surrogate (unweighted)",
    },
    "surrogate_weighted": {
        "color": POSTERIOR_GREEN,
        "ls": "-",
        "marker": "o",
        "label": "Surrogate (weighted)",
    },
}


_CONFIG_LABELS = {
    "dim": "dim",
    "horizon": "h",
    "n_users": "n",
    "n_items": "items",
    "sensitive_frac": "s_frac",
    "sensitivity_noise_scale": "s_noise",
    "sensitivity_assignment": "sens",
}

_SWEEP_PARAM_TO_CONFIG_KEY = {
    "dim": "dim",
    "dropout": "p_dropout",
    "horizon": "horizon",
    "items": "n_items",
    "sensitive-frac": "sensitive_frac",
    "sensitivity-noise": "sensitivity_noise_scale",
    "users": "n_users",
}


def setup() -> None:
    """Apply notebook plotting style and print the results directory."""
    apply_notebook_style()
    print(f"Results directory: {RESULTS_DIR}")


def list_sweeps(param: str | None = None) -> list[Path]:
    """Return available sweep JSON files, newest first."""
    pattern = "*.json" if param is None else f"*sweep_{param.replace('-', '_')}*.json"
    return sorted(
        [path for path in RESULTS_DIR.glob(pattern) if "sweep" in path.name],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def show_sweeps(param: str | None = None, limit: int = 20) -> list[Path]:
    """Print available sweeps and return them, newest first."""
    files = list_sweeps(param)
    if not files:
        label = "any parameter" if param is None else param
        print(f"No sweep files found for {label}.")
        return []
    for index, path in enumerate(files[:limit]):
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"{index:>2}: {path.name}  ({config_label(data)})")
    return files


def load_sweep(
    param: str,
    *,
    sweep_file: str | Path | None = None,
    config_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load one sweep JSON.

    If ``sweep_file`` is provided, that exact file is loaded. Otherwise, the
    newest sweep for ``param`` matching ``config_filter`` is selected.
    """
    if sweep_file:
        path = Path(sweep_file)
        if not path.is_absolute():
            path = RESULTS_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Sweep file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loading exact file: {path.name}")
        return data

    files = list_sweeps(param)
    if config_filter:
        files = [
            path for path in files
            if _matches_config(path, config_filter)
        ]
    if not files:
        raise FileNotFoundError(
            f"No sweep file found for param='{param}' matching "
            f"{config_filter or '{}'}."
        )
    path = files[0]
    print(f"Loading latest matching file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(
    *,
    run_file: str | Path | None = None,
    pattern: str = "*.json",
) -> dict[str, Any]:
    """Load one single-run JSON, excluding sweep summaries."""
    if run_file:
        path = Path(run_file)
        if not path.is_absolute():
            path = RESULTS_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Run file not found: {path}")
        print(f"Loading exact file: {path.name}")
        return json.loads(path.read_text(encoding="utf-8"))

    files = [
        path for path in sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
        if "sweep" not in path.name
    ]
    if not files:
        raise FileNotFoundError(f"No single-run JSON found matching '{pattern}'.")
    path = files[-1]
    print(f"Loading latest single run: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def sweep_series(data: dict[str, Any], metric: str) -> dict[str, tuple[list, list]]:
    """Extract x/y values per policy from a sweep JSON."""
    series: dict[str, tuple[list, list]] = {}
    for condition in data["conditions"]:
        x = condition["value"]
        for policy, metrics in condition["policies"].items():
            xs, ys = series.setdefault(policy, ([], []))
            xs.append(x)
            ys.append(metrics.get(metric, float("nan")))
    return series


def config_label(data: dict[str, Any]) -> str:
    """Return a compact label for figure titles."""
    config = data.get("fixed_config", data.get("config", {}))
    sweep_key = _SWEEP_PARAM_TO_CONFIG_KEY.get(data.get("sweep_param"))
    pieces = []
    for key in _CONFIG_LABELS:
        if key == "sensitivity_noise_scale" and sweep_key != key:
            value = config.get(key)
            if value is None or float(value) == 0.0:
                continue
        pieces.append(_format_config_piece(key, config, sweep_key))
    axes = config.get("sensitive_axes")
    if axes is not None:
        pieces.append(f"axes={axes}")
    dropout = config.get("p_dropout", config.get("p_dropout_sens"))
    if sweep_key == "p_dropout":
        pieces.append("p_drop=varied")
    elif dropout is not None:
        pieces.append(f"p_drop={100 * dropout:.0f}%")
    return ", ".join(pieces)


def _format_config_piece(
    key: str,
    config: dict[str, Any],
    sweep_key: str | None,
) -> str:
    label = _CONFIG_LABELS[key]
    value = "varied" if key == sweep_key else config.get(key, "?")
    return f"{label}={value}"


def plot_metric_lines(
    ax,
    data: dict[str, Any],
    metric: str,
    *,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    xlabel: str,
    ylabel: str,
    title: str,
    legend: bool = True,
) -> None:
    """Plot one metric across sweep values for all policies."""
    for policy, (xs, ys) in sweep_series(data, metric).items():
        style = POLICY_STYLE[policy]
        ax.plot(
            np.array(xs, dtype=float) * x_scale,
            np.array(ys, dtype=float) * y_scale,
            color=style["color"],
            ls=style["ls"],
            marker=style["marker"],
            markersize=5,
            linewidth=1.6,
            label=style["label"],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_ax(ax, grid_axis="y")
    if legend:
        ax.legend(fontsize=8)


def equalize_y_axes(*axes, pad_fraction: float = 0.05) -> None:
    """Give multiple axes the same y-limits for direct visual comparison."""
    limits = [ax.get_ylim() for ax in axes]
    ymin = min(low for low, _ in limits)
    ymax = max(high for _, high in limits)
    if np.isclose(ymin, ymax):
        pad = 1.0 if np.isclose(ymin, 0.0) else abs(ymin) * pad_fraction
    else:
        pad = (ymax - ymin) * pad_fraction
    for ax in axes:
        ax.set_ylim(ymin - pad, ymax + pad)


def plot_weighted_delta(
    ax,
    data: dict[str, Any],
    metric: str,
    *,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot weighted minus unweighted surrogate for a metric."""
    weighted = _policy_metric_by_value(data, "surrogate_weighted", metric)
    unweighted = _policy_metric_by_value(data, "surrogate_unweighted", metric)
    common_x = sorted(set(weighted) & set(unweighted))
    deltas = [(weighted[x] - unweighted[x]) * y_scale for x in common_x]
    ax.axhline(0.0, color=STRUCTURE_GRAY, linewidth=1.0)
    ax.plot(
        np.array(common_x, dtype=float) * x_scale,
        deltas,
        color=POSTERIOR_GREEN,
        marker="o",
        linewidth=1.8,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_ax(ax, grid_axis="y")


def plot_pareto(
    ax,
    data: dict[str, Any],
    *,
    x_metric: str = "dropout_rate",
    y_metric: str = "mean_estimation_error",
    x_scale: float = 100.0,
    y_scale: float = 1.0,
    xlabel: str = "Episode dropout rate (%)",
    ylabel: str = "Mean estimation error (lower is better)",
    title: str = "Dropout-risk vs error",
    annotate: bool = True,
) -> None:
    """
    Plot one point per policy per sweep condition.

    Both axes are costs by default, so points closer to the lower-left are
    preferable. We avoid connecting the points because adjacent sweep values
    are different experimental conditions, not a policy trajectory.
    """
    for policy in data["conditions"][0]["policies"]:
        style = POLICY_STYLE[policy]
        xs = []
        ys = []
        labels = []
        for condition in data["conditions"]:
            metrics = condition["policies"].get(policy)
            if not metrics:
                continue
            xs.append(metrics.get(x_metric, float("nan")) * x_scale)
            ys.append(metrics.get(y_metric, float("nan")) * y_scale)
            labels.append(condition["value"])
        ax.scatter(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            s=42,
            alpha=0.86,
            edgecolor="white",
            linewidth=0.5,
            label=style["label"],
        )
        if annotate:
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(
                    f"{label:g}",
                    (x, y),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color=style["color"],
                    alpha=0.75,
                )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_ax(ax, grid_axis="both")
    ax.legend(fontsize=8)


def _matches_config(path: Path, config_filter: dict[str, Any]) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("fixed_config", {})
    return all(config.get(key) == value for key, value in config_filter.items())


def _policy_metric_by_value(
    data: dict[str, Any],
    policy: str,
    metric: str,
) -> dict[Any, float]:
    return {
        condition["value"]: condition["policies"][policy].get(metric, float("nan"))
        for condition in data["conditions"]
        if policy in condition["policies"]
    }


__all__ = [
    "PALETTE",
    "POLICY_STYLE",
    "RESULTS_DIR",
    "config_label",
    "equalize_y_axes",
    "list_sweeps",
    "load_run",
    "load_sweep",
    "plot_metric_lines",
    "plot_pareto",
    "plot_weighted_delta",
    "save",
    "setup",
    "show_sweeps",
    "style_ax",
    "sweep_series",
]
