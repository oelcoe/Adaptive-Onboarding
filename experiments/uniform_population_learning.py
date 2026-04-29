"""
experiments/uniform_population_learning.py
==========================================
Compare learning curves across policies when true users are sampled from a
near-uniform latent-trait population.

This script answers two time-step questions:

1. How far is the posterior mean from the user's true latent vector?
2. How large is posterior uncertainty after each answered question?

It saves a JSON summary, a CSV curve table, a Markdown report, and optionally a
PNG figure in experiments/results/.

Usage
-----
    python -m experiments.uniform_population_learning
    python -m experiments.uniform_population_learning --dim 6 --horizon 20 --users 500
    python -m experiments.uniform_population_learning --theta-low -3 --theta-high 3
    python -m experiments.uniform_population_learning --no-myopic
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.policy_comparison import (
    DEFAULT_DIM,
    DEFAULT_HORIZON,
    DEFAULT_N_CATEGORIES,
    DEFAULT_N_ITEMS,
    DEFAULT_N_USERS,
    DEFAULT_P_DROPOUT,
    DEFAULT_SEED_BANK,
    DEFAULT_SEED_POP,
    DEFAULT_SEED_SIM,
    DEFAULT_SENSITIVE_AXES,
    DEFAULT_SENSITIVITY_ASSIGNMENT,
    DEFAULT_SENSITIVE_FRAC,
    POLICIES,
)
from src.belief import BeliefState
from src.metrics import aggregate_policy_metrics
from src.policies import make_sensitive_constant_stay_prob
from src.simulate import EpisodeResult, simulate_population
from src.synthetic import synthetic_item_bank


DEFAULT_THETA_LOW = -2.0
DEFAULT_THETA_HIGH = 2.0
DEFAULT_PRIOR_SCALE = 1.0


def _format_stem_number(value: float) -> str:
    """Format numeric config values compactly for result filenames."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class BeliefSnapshot:
    """Posterior state at a particular answered-question count."""

    answered: int
    mu: NDArray[np.float64]
    Sigma: NDArray[np.float64]


def _logdet(Sigma: NDArray[np.float64]) -> float:
    sign, logabsdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    return float(logabsdet)


def _d_error(Sigma: NDArray[np.float64]) -> float:
    return float(np.exp(_logdet(Sigma) / Sigma.shape[0]))


def _dimension_metric_columns(prefix: str, values: NDArray[np.float64]) -> dict[str, float]:
    return {
        f"{prefix}_dim_{idx + 1}": float(value)
        for idx, value in enumerate(values)
    }


def _dimension_column_sort_key(name: str) -> tuple[str, int]:
    prefix, _, suffix = name.rpartition("_dim_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return name, 0


def sample_uniform_population(
    *,
    n_users: int,
    dim: int,
    theta_low: float,
    theta_high: float,
    seed: int,
) -> NDArray[np.float64]:
    """Sample true latent vectors independently from Uniform(theta_low, theta_high)."""
    if theta_low >= theta_high:
        raise ValueError("theta_low must be smaller than theta_high.")
    rng = np.random.default_rng(seed)
    return rng.uniform(low=theta_low, high=theta_high, size=(n_users, dim))


def _answered_snapshots(
    result: EpisodeResult,
    prior: BeliefState,
) -> list[BeliefSnapshot]:
    """
    Return snapshots indexed by answered-question count.

    Snapshot 0 is the prior. A dropout step does not add a snapshot because no
    response was observed and the belief did not update.
    """
    snapshots = [BeliefSnapshot(answered=0, mu=prior.mu, Sigma=prior.Sigma)]
    answered = 0
    for step in result.steps:
        if step.dropped_out:
            continue
        answered += 1
        if step.belief_mu_after is None or step.belief_Sigma_after is None:
            raise ValueError("Answered step is missing posterior state.")
        snapshots.append(
            BeliefSnapshot(
                answered=answered,
                mu=step.belief_mu_after,
                Sigma=step.belief_Sigma_after,
            )
        )
    return snapshots


def _snapshot_at(
    snapshots: list[BeliefSnapshot],
    answered: int,
    *,
    carry_forward: bool,
) -> BeliefSnapshot | None:
    if answered < len(snapshots):
        return snapshots[answered]
    if carry_forward:
        return snapshots[-1]
    return None


def aggregate_learning_curves(
    results: list[EpisodeResult],
    theta_trues: NDArray[np.float64],
    prior: BeliefState,
    horizon: int,
    *,
    carry_forward: bool,
) -> list[dict[str, float | int]]:
    """
    Aggregate estimation and uncertainty metrics for answered counts 0..horizon.

    carry_forward=False gives the conditional curve: only episodes that reached
    a count are included. carry_forward=True keeps every user in the denominator
    and freezes their final belief after dropout.
    """
    snapshots_by_episode = [_answered_snapshots(result, prior) for result in results]
    rows: list[dict[str, float | int]] = []

    for answered in range(horizon + 1):
        l2_errors: list[float] = []
        sq_errors: list[float] = []
        d_errors: list[float] = []
        traces: list[float] = []
        logdets: list[float] = []
        marginal_variances: list[NDArray[np.float64]] = []
        marginal_reductions: list[NDArray[np.float64]] = []
        marginal_step_reductions: list[NDArray[np.float64]] = []

        for snapshots, theta in zip(snapshots_by_episode, theta_trues):
            snapshot = _snapshot_at(
                snapshots,
                answered,
                carry_forward=carry_forward,
            )
            if snapshot is None:
                continue

            diff = snapshot.mu - theta
            sq_error = float(diff @ diff)
            l2_errors.append(float(np.sqrt(sq_error)))
            sq_errors.append(sq_error)
            d_errors.append(_d_error(snapshot.Sigma))
            traces.append(float(np.trace(snapshot.Sigma)))
            logdets.append(_logdet(snapshot.Sigma))
            marginal_variance = np.diag(snapshot.Sigma).astype(float)
            marginal_variances.append(marginal_variance)
            marginal_reductions.append(np.diag(prior.Sigma).astype(float) - marginal_variance)

            previous_snapshot = (
                _snapshot_at(snapshots, answered - 1, carry_forward=carry_forward)
                if answered > 0
                else None
            )
            if previous_snapshot is None:
                marginal_step_reductions.append(np.zeros(prior.dim, dtype=float))
            else:
                marginal_step_reductions.append(
                    np.diag(previous_snapshot.Sigma).astype(float) - marginal_variance
                )

        if not l2_errors:
            continue

        mean_marginal_variance = np.mean(marginal_variances, axis=0)
        mean_marginal_reduction = np.mean(marginal_reductions, axis=0)
        mean_marginal_step_reduction = np.mean(marginal_step_reductions, axis=0)

        rows.append(
            {
                "answered": answered,
                "n": len(l2_errors),
                "mean_l2_error": float(np.mean(l2_errors)),
                "median_l2_error": float(np.median(l2_errors)),
                "rmse": float(np.sqrt(np.mean(sq_errors))),
                "mean_d_error": float(np.mean(d_errors)),
                "median_d_error": float(np.median(d_errors)),
                "mean_trace": float(np.mean(traces)),
                "mean_logdet": float(np.mean(logdets)),
                **_dimension_metric_columns(
                    "mean_variance",
                    mean_marginal_variance,
                ),
                **_dimension_metric_columns(
                    "mean_variance_reduction",
                    mean_marginal_reduction,
                ),
                **_dimension_metric_columns(
                    "mean_variance_step_reduction",
                    mean_marginal_step_reduction,
                ),
            }
        )

    return rows


def _summary_lines(
    policies: list[str],
    final_rows: dict[str, dict[str, float | int]],
    dropout_rates: dict[str, float],
) -> list[str]:
    cols = [
        ("Policy", 22, "<", "s"),
        ("Dropout %", 10, ">", ".1f"),
        ("N", 8, ">", "d"),
        ("L2 error", 11, ">", ".4f"),
        ("RMSE", 11, ">", ".4f"),
        ("D-error", 11, ">", ".4f"),
        ("Trace", 11, ">", ".4f"),
    ]

    def fmt(value: object, width: int, align: str, spec: str) -> str:
        if spec == "s":
            return f"{value:{align}{width}s}"
        if spec == "d":
            return f"{int(value):{align}{width}d}"
        return f"{float(value):{align}{width}{spec}}"

    header = "  ".join(f"{name:{align}{width}s}" for name, width, align, _ in cols)
    sep = "-" * len(header)
    lines = [header, sep]

    for policy in policies:
        row = final_rows[policy]
        values = (
            policy,
            100.0 * dropout_rates[policy],
            row["n"],
            row["mean_l2_error"],
            row["rmse"],
            row["mean_d_error"],
            row["mean_trace"],
        )
        lines.append(
            "  ".join(
                fmt(value, width, align, spec)
                for value, (_, width, align, spec) in zip(values, cols)
            )
        )
    lines.append(sep)
    return lines


def _write_curve_csv(path: Path, rows: list[dict[str, object]]) -> None:
    base_fieldnames = [
        "mode",
        "policy",
        "answered",
        "n",
        "mean_l2_error",
        "median_l2_error",
        "rmse",
        "mean_d_error",
        "median_d_error",
        "mean_trace",
        "mean_logdet",
    ]
    extra_fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in base_fieldnames
        },
        key=_dimension_column_sort_key,
    )
    fieldnames = [*base_fieldnames, *extra_fieldnames]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_save_plot(
    path: Path,
    curve_rows: list[dict[str, object]],
    policies: list[str],
    *,
    mode: str = "carried",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    metrics = [
        ("mean_l2_error", "Mean L2 estimation error"),
        ("mean_d_error", "Mean D-error"),
        ("mean_trace", "Mean posterior trace"),
    ]

    for policy in policies:
        policy_rows = [
            row
            for row in curve_rows
            if row["mode"] == mode and row["policy"] == policy
        ]
        xs = [int(row["answered"]) for row in policy_rows]
        for ax, (metric, title) in zip(axes, metrics):
            ys = [float(row[metric]) for row in policy_rows]
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3, label=policy)
            ax.set_title(title)
            ax.set_xlabel("Answered questions")
            ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("lower is better")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Uniform-population learning curves ({mode})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_uniform_learning_experiment(
    *,
    dim: int = DEFAULT_DIM,
    n_items: int = DEFAULT_N_ITEMS,
    n_categories: int = DEFAULT_N_CATEGORIES,
    sensitive_frac: float = DEFAULT_SENSITIVE_FRAC,
    sensitivity_assignment: str = DEFAULT_SENSITIVITY_ASSIGNMENT,
    sensitive_axes: list[int] | None = DEFAULT_SENSITIVE_AXES,
    p_dropout: float = DEFAULT_P_DROPOUT,
    n_users: int = DEFAULT_N_USERS,
    horizon: int = DEFAULT_HORIZON,
    theta_low: float = DEFAULT_THETA_LOW,
    theta_high: float = DEFAULT_THETA_HIGH,
    prior_scale: float = DEFAULT_PRIOR_SCALE,
    seed_bank: int = DEFAULT_SEED_BANK,
    seed_pop: int = DEFAULT_SEED_POP,
    seed_sim: int = DEFAULT_SEED_SIM,
    policies: list[str] | None = None,
    save_plot: bool = True,
) -> Path:
    """Run the uniform-population learning experiment and save result files."""
    if prior_scale <= 0:
        raise ValueError("prior_scale must be positive.")

    active_policies = policies if policies is not None else POLICIES
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    stem = (
        f"{timestamp}_uniform_learning_dim{dim}_h{horizon}_"
        f"p{int(p_dropout * 100)}_n{n_users}_ps{_format_stem_number(prior_scale)}"
    )

    item_bank = synthetic_item_bank(
        n_items=n_items,
        dim=dim,
        n_categories=n_categories,
        sensitive_fraction=sensitive_frac,
        sensitivity_assignment=sensitivity_assignment,
        sensitive_axes=sensitive_axes,
        rng_seed=seed_bank,
        vary_sensitivity_levels=False,
    )
    theta_trues = sample_uniform_population(
        n_users=n_users,
        dim=dim,
        theta_low=theta_low,
        theta_high=theta_high,
        seed=seed_pop,
    )
    prior = BeliefState(
        mu=np.zeros(dim),
        Sigma=prior_scale * np.eye(dim),
    )
    stay_prob_fn = make_sensitive_constant_stay_prob(
        p_stay_sensitive=1.0 - p_dropout,
        p_stay_normal=1.0,
    )

    print()
    print("=" * 68)
    print(" UNIFORM-POPULATION LEARNING EXPERIMENT")
    print("=" * 68)
    print(f"  Users              : {n_users}")
    print(f"  Latent dimensions  : {dim}")
    print(f"  User theta         : Uniform({theta_low:g}, {theta_high:g}) per dimension")
    print(f"  Prior              : N(0, {prior_scale:g} I_{dim})")
    print(f"  Horizon            : {horizon} answered questions max")
    print(f"  Item bank          : {n_items} items")
    print(f"  Sensitive labels   : {sensitivity_assignment}")
    if sensitive_axes is not None:
        print(f"  Sensitive axes     : {sensitive_axes}")
    print(f"  Dropout prob.      : {100 * p_dropout:.0f} % per sensitive question")
    print("=" * 68)
    print()

    curve_rows: list[dict[str, object]] = []
    final_rows: dict[str, dict[str, float | int]] = {}
    dropout_rates: dict[str, float] = {}
    policy_json: dict[str, object] = {}

    for policy_idx, policy in enumerate(active_policies):
        rng = np.random.default_rng([seed_sim, policy_idx])
        t0 = time.perf_counter()
        results = simulate_population(
            theta_trues=theta_trues,
            prior_belief=prior,
            item_bank=item_bank,
            strategy=policy,
            horizon=horizon,
            stay_prob_fn=stay_prob_fn,
            rng=rng,
        )
        elapsed = time.perf_counter() - t0
        metrics = aggregate_policy_metrics(results, policy_name=policy, item_bank=item_bank)
        dropout_rates[policy] = metrics.dropout_rate

        for mode, carry_forward in [("available", False), ("carried", True)]:
            rows = aggregate_learning_curves(
                results=results,
                theta_trues=theta_trues,
                prior=prior,
                horizon=horizon,
                carry_forward=carry_forward,
            )
            for row in rows:
                curve_rows.append({"mode": mode, "policy": policy, **row})
            if mode == "carried":
                final_rows[policy] = rows[-1]

        policy_json[policy] = {
            "dropout_rate": metrics.dropout_rate,
            "mean_n_answered": metrics.mean_n_answered,
            "mean_n_asked": metrics.mean_n_asked,
            "mean_sensitive_asked": metrics.mean_sensitive_asked,
            "sensitive_rate": metrics.sensitive_rate,
            "batch_seconds": elapsed,
        }

        final = final_rows[policy]
        print(
            f"  {policy:<22}  "
            f"dropout={100 * metrics.dropout_rate:5.1f} %  "
            f"answered={metrics.mean_n_answered:5.2f}  "
            f"l2={final['mean_l2_error']:6.4f}  "
            f"d_error={final['mean_d_error']:6.4f}  "
            f"trace={final['mean_trace']:6.4f}  "
            f"({elapsed:.1f} s)"
        )

    summary_lines = _summary_lines(active_policies, final_rows, dropout_rates)
    print()
    print(" FINAL CARRIED CURVE POINT")
    print()
    for line in summary_lines:
        print(line)
    print()

    config = {
        "dim": dim,
        "n_items": n_items,
        "n_categories": n_categories,
        "sensitive_frac": sensitive_frac,
        "sensitivity_assignment": sensitivity_assignment,
        "sensitive_axes": sensitive_axes,
        "p_dropout_sens": p_dropout,
        "n_users": n_users,
        "horizon": horizon,
        "theta_distribution": "uniform",
        "theta_low": theta_low,
        "theta_high": theta_high,
        "prior_scale": prior_scale,
        "seed_bank": seed_bank,
        "seed_population": seed_pop,
        "seed_simulate": seed_sim,
    }

    json_path = results_dir / f"{stem}.json"
    csv_path = results_dir / f"{stem}_curves.csv"
    md_path = results_dir / f"{stem}.md"
    png_path = results_dir / f"{stem}.png"

    json_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "config": config,
                "policies": policy_json,
                "curves": curve_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_curve_csv(csv_path, curve_rows)

    md_lines = [
        f"# Uniform-Population Learning -- {timestamp}",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|---|---|",
        *[f"| {key} | {value} |" for key, value in config.items()],
        "",
        "## Final carried curve point",
        "",
        "```",
        *summary_lines,
        "```",
        "",
        "## Interpretation",
        "",
        "- `available` curves average only over users who reached each answered-question count.",
        "- `carried` curves keep every user in the denominator and freeze the final belief after dropout.",
        "- L2/RMSE measure how far the posterior mean is from the true latent vector.",
        "- D-error/trace measure how much posterior uncertainty remains.",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    if save_plot:
        _maybe_save_plot(png_path, curve_rows, active_policies, mode="carried")

    repo_root = Path(__file__).resolve().parents[1]
    print("Saved uniform-learning outputs:")
    print(f"  JSON     : {json_path.relative_to(repo_root)}")
    print(f"  CSV      : {csv_path.relative_to(repo_root)}")
    print(f"  Markdown : {md_path.relative_to(repo_root)}")
    if save_plot and png_path.exists():
        print(f"  PNG      : {png_path.relative_to(repo_root)}")

    return json_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Policy learning curves under near-uniform latent users.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--categories", type=int, default=DEFAULT_N_CATEGORIES)
    parser.add_argument("--sensitive-frac", type=float, default=DEFAULT_SENSITIVE_FRAC)
    parser.add_argument(
        "--sensitivity-assignment",
        type=str,
        default=DEFAULT_SENSITIVITY_ASSIGNMENT,
        choices=["random", "axis_aligned", "high_trait_tail"],
    )
    parser.add_argument("--sensitive-axes", type=int, nargs="+", default=DEFAULT_SENSITIVE_AXES)
    parser.add_argument("--dropout", type=float, default=DEFAULT_P_DROPOUT)
    parser.add_argument("--users", type=int, default=DEFAULT_N_USERS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--theta-low", type=float, default=DEFAULT_THETA_LOW)
    parser.add_argument("--theta-high", type=float, default=DEFAULT_THETA_HIGH)
    parser.add_argument("--prior-scale", type=float, default=DEFAULT_PRIOR_SCALE)
    parser.add_argument("--seed-bank", type=int, default=DEFAULT_SEED_BANK)
    parser.add_argument("--seed-pop", type=int, default=DEFAULT_SEED_POP)
    parser.add_argument("--seed-sim", type=int, default=DEFAULT_SEED_SIM)
    parser.add_argument(
        "--no-myopic",
        action="store_true",
        help="Exclude myopic_exact to make the run faster.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the PNG learning-curve plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    active_policies = [
        policy for policy in POLICIES if not (args.no_myopic and policy == "myopic_exact")
    ]
    run_uniform_learning_experiment(
        dim=args.dim,
        n_items=args.items,
        n_categories=args.categories,
        sensitive_frac=args.sensitive_frac,
        sensitivity_assignment=args.sensitivity_assignment,
        sensitive_axes=args.sensitive_axes,
        p_dropout=args.dropout,
        n_users=args.users,
        horizon=args.horizon,
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        prior_scale=args.prior_scale,
        seed_bank=args.seed_bank,
        seed_pop=args.seed_pop,
        seed_sim=args.seed_sim,
        policies=active_policies,
        save_plot=not args.no_plot,
    )
