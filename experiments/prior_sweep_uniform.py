"""
experiments/prior_sweep_uniform.py
==================================
Sweep the Gaussian prior covariance scale while true users are sampled from a
near-uniform latent-trait population.

Each condition runs experiments.uniform_population_learning with

    prior Sigma_0 = prior_scale * I

and stores a combined prior-sweep summary for plotting.

Usage
-----
    python -m experiments.prior_sweep_uniform --values 0.25 0.5 1 2 4 9 --no-myopic
    python -m experiments.prior_sweep_uniform --dim 6 --horizon 20 --users 500 --no-myopic
    python -m experiments.prior_sweep_uniform --theta-low -3 --theta-high 3 --values 1 4 9
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

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
from experiments.uniform_population_learning import (
    DEFAULT_THETA_HIGH,
    DEFAULT_THETA_LOW,
    run_uniform_learning_experiment,
)


DEFAULT_PRIOR_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 9.0]


def _summary_lines(conditions: list[dict], policies: list[str]) -> list[str]:
    cols = [
        ("prior", 8, ">", ".3g"),
        ("Policy", 22, "<", "s"),
        ("Dropout %", 10, ">", ".1f"),
        ("Answered", 10, ">", ".2f"),
        ("L2 error", 11, ">", ".4f"),
        ("RMSE", 11, ">", ".4f"),
        ("D-error", 11, ">", ".4f"),
        ("Trace", 11, ">", ".4f"),
    ]

    def fmt(value: object, width: int, align: str, spec: str) -> str:
        if spec == "s":
            return f"{value:{align}{width}s}"
        return f"{float(value):{align}{width}{spec}}"

    header = "  ".join(f"{name:{align}{width}s}" for name, width, align, _ in cols)
    sep = "-" * len(header)
    lines = [header, sep]

    for condition in conditions:
        first = True
        value = condition["prior_scale"]
        for policy in policies:
            metrics = condition["policies"].get(policy)
            if metrics is None:
                continue
            row = (
                value if first else "",
                policy,
                100.0 * metrics["dropout_rate"],
                metrics["mean_n_answered"],
                metrics["mean_l2_error"],
                metrics["rmse"],
                metrics["mean_d_error"],
                metrics["mean_trace"],
            )
            cells = []
            for index, (cell, (_, width, align, spec)) in enumerate(zip(row, cols)):
                if index == 0 and not first:
                    cells.append(" " * width)
                else:
                    cells.append(fmt(cell, width, align, spec))
            lines.append("  ".join(cells))
            first = False
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    lines.append(sep)
    return lines


def _final_carried_by_policy(run_data: dict) -> dict[str, dict[str, float]]:
    curves = run_data["curves"]
    policies = run_data["policies"]
    final: dict[str, dict[str, float]] = {}
    for policy, policy_metrics in policies.items():
        rows = [
            row
            for row in curves
            if row["mode"] == "carried" and row["policy"] == policy
        ]
        if not rows:
            continue
        row = max(rows, key=lambda item: item["answered"])
        final[policy] = {
            "dropout_rate": float(policy_metrics["dropout_rate"]),
            "mean_n_answered": float(policy_metrics["mean_n_answered"]),
            "mean_n_asked": float(policy_metrics["mean_n_asked"]),
            "sensitive_rate": float(policy_metrics["sensitive_rate"]),
            "answered": int(row["answered"]),
            "n": int(row["n"]),
            "mean_l2_error": float(row["mean_l2_error"]),
            "median_l2_error": float(row["median_l2_error"]),
            "rmse": float(row["rmse"]),
            "mean_d_error": float(row["mean_d_error"]),
            "median_d_error": float(row["median_d_error"]),
            "mean_trace": float(row["mean_trace"]),
            "mean_logdet": float(row["mean_logdet"]),
        }
    return final


def _maybe_save_plot(path: Path, combined: dict, policies: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    metrics = [
        ("mean_l2_error", "Final mean L2 error"),
        ("mean_d_error", "Final mean D-error"),
        ("mean_trace", "Final mean trace"),
    ]
    x = np.array(combined["prior_values"], dtype=float)

    for policy in policies:
        for ax, (metric, title) in zip(axes, metrics):
            y = [
                condition["policies"][policy][metric]
                for condition in combined["conditions"]
                if policy in condition["policies"]
            ]
            ax.plot(x[: len(y)], y, marker="o", linewidth=1.6, markersize=4, label=policy)
            ax.set_xscale("log")
            ax.set_xlabel("Prior scale")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("lower is better")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Uniform-population prior sweep", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_prior_sweep(
    *,
    values: list[float],
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
    seed_bank: int = DEFAULT_SEED_BANK,
    seed_pop: int = DEFAULT_SEED_POP,
    seed_sim: int = DEFAULT_SEED_SIM,
    policies: list[str] | None = None,
    save_plot: bool = True,
) -> Path:
    if not values:
        raise ValueError("values must contain at least one prior scale.")
    if any(value <= 0 for value in values):
        raise ValueError("all prior scales must be positive.")

    active_policies = policies if policies is not None else POLICIES
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    print()
    print("=" * 68)
    print(f" PRIOR SWEEP: prior_scale over {values}")
    print("=" * 68)
    print(f"  User theta         : Uniform({theta_low:g}, {theta_high:g}) per dimension")
    print(f"  Users              : {n_users}")
    print(f"  Latent dimensions  : {dim}")
    print(f"  Horizon            : {horizon}")
    print(f"  Policies           : {', '.join(active_policies)}")
    print("=" * 68)
    print()

    conditions: list[dict] = []
    run_files: list[str] = []

    for prior_scale in values:
        print(f"{'=' * 20}  prior_scale = {prior_scale:g}  {'=' * 20}")
        run_path = run_uniform_learning_experiment(
            dim=dim,
            n_items=n_items,
            n_categories=n_categories,
            sensitive_frac=sensitive_frac,
            sensitivity_assignment=sensitivity_assignment,
            sensitive_axes=sensitive_axes,
            p_dropout=p_dropout,
            n_users=n_users,
            horizon=horizon,
            theta_low=theta_low,
            theta_high=theta_high,
            prior_scale=prior_scale,
            seed_bank=seed_bank,
            seed_pop=seed_pop,
            seed_sim=seed_sim,
            policies=active_policies,
            save_plot=False,
        )
        run_files.append(run_path.name)
        run_data = json.loads(run_path.read_text(encoding="utf-8"))
        conditions.append(
            {
                "prior_scale": prior_scale,
                "run_file": run_path.name,
                "policies": _final_carried_by_policy(run_data),
            }
        )

    fixed_config = {
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
        "seed_bank": seed_bank,
        "seed_population": seed_pop,
        "seed_simulate": seed_sim,
    }
    combined = {
        "timestamp": timestamp,
        "sweep_param": "prior_scale",
        "prior_values": values,
        "fixed_config": fixed_config,
        "run_files": run_files,
        "conditions": conditions,
    }

    summary_lines = _summary_lines(conditions, active_policies)
    print()
    print("=" * 68)
    print(" PRIOR SWEEP SUMMARY")
    print("=" * 68)
    print()
    for line in summary_lines:
        print(line)
    print()

    stem = f"{timestamp}_prior_sweep_uniform_dim{dim}_h{horizon}_p{int(p_dropout * 100)}_n{n_users}"
    json_path = results_dir / f"{stem}.json"
    md_path = results_dir / f"{stem}.md"
    png_path = results_dir / f"{stem}.png"

    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    md_lines = [
        f"# Prior Sweep Under Uniform Users -- {timestamp}",
        "",
        "## Fixed configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        *[f"| {key} | {value} |" for key, value in fixed_config.items()],
        "",
        "## Results",
        "",
        "```",
        *summary_lines,
        "```",
        "",
        "## Notes",
        "",
        "_Add your observations here._",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    if save_plot:
        _maybe_save_plot(png_path, combined, active_policies)

    repo_root = Path(__file__).resolve().parents[1]
    print("Saved prior-sweep outputs:")
    print(f"  JSON     : {json_path.relative_to(repo_root)}")
    print(f"  Markdown : {md_path.relative_to(repo_root)}")
    if save_plot and png_path.exists():
        print(f"  PNG      : {png_path.relative_to(repo_root)}")

    return json_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep prior covariance scale under near-uniform users.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--values", nargs="+", type=float, default=DEFAULT_PRIOR_VALUES)
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
    parser.add_argument("--seed-bank", type=int, default=DEFAULT_SEED_BANK)
    parser.add_argument("--seed-pop", type=int, default=DEFAULT_SEED_POP)
    parser.add_argument("--seed-sim", type=int, default=DEFAULT_SEED_SIM)
    parser.add_argument("--no-myopic", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    active_policies = [
        policy for policy in POLICIES if not (args.no_myopic and policy == "myopic_exact")
    ]
    run_prior_sweep(
        values=args.values,
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
        seed_bank=args.seed_bank,
        seed_pop=args.seed_pop,
        seed_sim=args.seed_sim,
        policies=active_policies,
        save_plot=not args.no_plot,
    )
