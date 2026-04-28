"""
experiments/sweep.py
====================
Sweep a single parameter across a range of values and compare all five
policies at each point.  Results are saved individually per condition and
as a combined sweep summary.

Usage
-----
    # Sweep dropout probability (most informative first experiment)
    python -m experiments.sweep --param dropout --values 0 0.05 0.10 0.20 0.40

    # Sweep horizon for a 6-dimensional setup
    python -m experiments.sweep --param horizon --values 6 12 18 24 36 \\
        --dim 6 --dropout 0.10

    # Sweep latent dimensions
    python -m experiments.sweep --param dim --values 2 4 6 8 --horizon 20

    # Skip myopic_exact (slow) to speed up sweeps
    python -m experiments.sweep --param dropout --values 0.05 0.10 0.20 --no-myopic

    # All fixed parameters accept the same overrides as policy_comparison.py
    python -m experiments.sweep --help
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
    DEFAULT_SENSITIVE_FRAC,
    POLICIES,
    run_experiment,
)
from src.metrics import PolicyMetrics


# ---------------------------------------------------------------------------
# Swept parameter registry
# ---------------------------------------------------------------------------

# Maps CLI name -> kwarg name accepted by run_experiment()
SWEEP_PARAMS: dict[str, str] = {
    "dropout":        "p_dropout",
    "horizon":        "horizon",
    "dim":            "dim",
    "users":          "n_users",
    "sensitive-frac": "sensitive_frac",
    "items":          "n_items",
}


# ---------------------------------------------------------------------------
# Summary table across the sweep
# ---------------------------------------------------------------------------


def _sweep_summary_lines(
    param_name: str,
    sweep_values: list[float],
    all_results: list[dict[str, PolicyMetrics]],
    all_est_errors: list[dict[str, float]],
    policies: list[str],
) -> list[str]:
    """
    One row per (sweep_value, policy) showing the key metrics.
    """
    CP = 22   # policy column
    CV = 10   # sweep-value column
    CF = 11   # float column
    CN = 9    # integer column

    cols = [
        (param_name,    CV, ">", ".3g"),
        ("Policy",      CP, "<", "s"),
        ("Dropout %",   CF, ">", ".1f"),
        ("Answered",    CF, ">", ".2f"),
        ("Sens. asked", CF, ">", ".2f"),
        ("D-error",     CF, ">", ".4f"),
        ("Est. error",  CF, ">", ".4f"),
    ]

    def _fmt(value: object, width: int, align: str, fmt: str) -> str:
        if fmt == "s":
            return f"{value:{align}{width}s}"
        return f"{float(value):{align}{width}{fmt}}"

    header_parts = [f"{label:{align}{width}s}" for label, width, align, _ in cols]
    header = "  ".join(header_parts)
    sep    = "-" * len(header)

    lines = [header, sep]
    for val, pm_dict, err_dict in zip(sweep_values, all_results, all_est_errors):
        first = True
        for policy in policies:
            if policy not in pm_dict:
                continue
            pm = pm_dict[policy]
            row_vals = (
                val if first else "",   # print sweep value only on first policy row
                pm.policy_name,
                pm.dropout_rate * 100,
                pm.mean_n_answered,
                pm.mean_sensitive_asked,
                pm.mean_final_d_error,
                err_dict.get(policy, float("nan")),
            )
            # For the sweep-value cell, blank out repeated rows
            cells = []
            for i, (v, (_, width, align, fmt)) in enumerate(zip(row_vals, cols)):
                if i == 0 and not first:
                    cells.append(" " * width)
                else:
                    cells.append(_fmt(v, width, align, fmt))
            lines.append("  ".join(cells))
            first = False
        lines.append("")   # blank line between sweep values
    if lines[-1] == "":
        lines.pop()
    lines.append(sep)
    return lines


# ---------------------------------------------------------------------------
# Main sweep function
# ---------------------------------------------------------------------------


def run_sweep(
    param: str,
    values: list[float],
    *,
    skip_myopic: bool = False,
    # fixed parameters (same defaults as policy_comparison.py)
    dim: int            = DEFAULT_DIM,
    n_items: int        = DEFAULT_N_ITEMS,
    n_categories: int   = DEFAULT_N_CATEGORIES,
    sensitive_frac: float = DEFAULT_SENSITIVE_FRAC,
    p_dropout: float    = DEFAULT_P_DROPOUT,
    n_users: int        = DEFAULT_N_USERS,
    horizon: int        = DEFAULT_HORIZON,
    seed_bank: int      = DEFAULT_SEED_BANK,
    seed_pop: int       = DEFAULT_SEED_POP,
    seed_sim: int       = DEFAULT_SEED_SIM,
) -> None:
    """
    Run policy_comparison.run_experiment() for each value in ``values``,
    varying the parameter named ``param`` while keeping all others fixed.

    Parameters
    ----------
    param:
        Parameter to sweep.  Must be one of: dropout, horizon, dim, users,
        sensitive-frac, items.
    values:
        Ordered list of values to evaluate.
    skip_myopic:
        If True, exclude myopic_exact from the run (saves ~40x compute per
        condition; useful for large sweeps).
    """
    if param not in SWEEP_PARAMS:
        raise ValueError(
            f"Unknown sweep parameter '{param}'. "
            f"Choose from: {', '.join(SWEEP_PARAMS)}"
        )

    kwarg = SWEEP_PARAMS[param]
    active_policies = [p for p in POLICIES if not (skip_myopic and p == "myopic_exact")]

    # Base kwargs shared across all conditions
    base_kwargs: dict = dict(
        dim=dim,
        n_items=n_items,
        n_categories=n_categories,
        sensitive_frac=sensitive_frac,
        p_dropout=p_dropout,
        n_users=n_users,
        horizon=horizon,
        seed_bank=seed_bank,
        seed_pop=seed_pop,
        seed_sim=seed_sim,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results:    list[dict[str, PolicyMetrics]] = []
    all_est_errors: list[dict[str, float]]         = []

    width = 60
    print()
    print("=" * width)
    print(f" SWEEP: {param} over {values}")
    print("=" * width)
    if skip_myopic:
        print("  (myopic_exact excluded)")
    print()

    results_dir = Path(__file__).resolve().parent / "results"

    for val in values:
        print(f"{'=' * 20}  {param} = {val}  {'=' * 20}")
        kwargs = {**base_kwargs, kwarg: val, "policies": active_policies}

        pm_dict = run_experiment(**kwargs)

        # Load estimation errors from the JSON that run_experiment just wrote
        json_files = sorted(results_dir.glob("*.json"))
        latest_json = json.loads(json_files[-1].read_text(encoding="utf-8"))
        est_errors = {
            policy: data["mean_estimation_error"]
            for policy, data in latest_json["policies"].items()
        }

        all_results.append(pm_dict)
        all_est_errors.append(est_errors)

    # ---- combined summary ----
    print()
    print("=" * width)
    print(f" SWEEP SUMMARY: {param}")
    print("=" * width)
    print()
    summary_lines = _sweep_summary_lines(
        param_name=param,
        sweep_values=list(values),
        all_results=all_results,
        all_est_errors=all_est_errors,
        policies=active_policies,
    )
    for line in summary_lines:
        print(line)
    print()

    # ---- save combined JSON ----
    results_dir = Path(__file__).resolve().parent / "results"
    stem = f"{timestamp}_sweep_{param.replace('-', '_')}"
    combined = {
        "timestamp": timestamp,
        "sweep_param": param,
        "sweep_values": list(values),
        "fixed_config": {k: v for k, v in base_kwargs.items() if k != kwarg},
        "conditions": [
            {
                "value": val,
                "policies": {
                    policy: {
                        "dropout_rate": pm_dict[policy].dropout_rate,
                        "mean_n_answered": pm_dict[policy].mean_n_answered,
                        "mean_sensitive_asked": pm_dict[policy].mean_sensitive_asked,
                        "mean_final_d_error": pm_dict[policy].mean_final_d_error,
                        "mean_logdet_reduction": pm_dict[policy].mean_logdet_reduction,
                        "mean_estimation_error": est_errors.get(policy, float("nan")),
                    }
                    for policy in active_policies
                    if policy in pm_dict
                },
            }
            for val, pm_dict, est_errors in zip(values, all_results, all_est_errors)
        ],
    }
    json_path = results_dir / f"{stem}.json"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Sweep summary saved to {json_path.relative_to(Path(__file__).resolve().parents[1])}")

    # ---- save combined Markdown ----
    md_lines = [
        f"# Sweep: {param}  --  {timestamp}",
        "",
        "## Fixed configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in base_kwargs.items() if k != kwarg],
        "",
        f"## Results by {param}",
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
    md_path = results_dir / f"{stem}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"          Markdown: {md_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep one parameter across a policy comparison experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--param", required=True,
        choices=list(SWEEP_PARAMS.keys()),
        help="Parameter to sweep",
    )
    p.add_argument(
        "--values", required=True, nargs="+", type=float,
        help="Space-separated list of values for the swept parameter",
    )
    p.add_argument("--no-myopic",      action="store_true",
                   help="Exclude myopic_exact (much faster sweeps)")
    # Fixed parameters (same as policy_comparison.py)
    p.add_argument("--dim",            type=int,   default=DEFAULT_DIM)
    p.add_argument("--items",          type=int,   default=DEFAULT_N_ITEMS)
    p.add_argument("--sensitive-frac", type=float, default=DEFAULT_SENSITIVE_FRAC)
    p.add_argument("--dropout",        type=float, default=DEFAULT_P_DROPOUT)
    p.add_argument("--users",          type=int,   default=DEFAULT_N_USERS)
    p.add_argument("--horizon",        type=int,   default=DEFAULT_HORIZON)
    p.add_argument("--seed-bank",      type=int,   default=DEFAULT_SEED_BANK)
    p.add_argument("--seed-pop",       type=int,   default=DEFAULT_SEED_POP)
    p.add_argument("--seed-sim",       type=int,   default=DEFAULT_SEED_SIM)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_sweep(
        param=args.param,
        values=args.values,
        skip_myopic=args.no_myopic,
        dim=args.dim,
        n_items=args.items,
        sensitive_frac=args.sensitive_frac,
        p_dropout=args.dropout,
        n_users=args.users,
        horizon=args.horizon,
        seed_bank=args.seed_bank,
        seed_pop=args.seed_pop,
        seed_sim=args.seed_sim,
    )
