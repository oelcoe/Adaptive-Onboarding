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

    # Sweep extra response noise on sensitive items
    python -m experiments.sweep --param sensitivity-noise --values 0 0.5 1 2

    # Skip myopic_exact (slow) to speed up sweeps
    python -m experiments.sweep --param dropout --values 0.05 0.10 0.20 --no-myopic

    # All fixed parameters accept the same overrides as policy_comparison.py
    python -m experiments.sweep --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    DEFAULT_SENSITIVITY_NOISE_SCALE,
    DEFAULT_SEED_BANK,
    DEFAULT_SEED_POP,
    DEFAULT_SEED_SIM,
    DEFAULT_SENSITIVE_AXES,
    DEFAULT_SENSITIVITY_ASSIGNMENT,
    DEFAULT_SENSITIVE_FRAC,
    POLICIES,
    _resolve_experiment_sensitive_axes,
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
    "sensitivity-noise": "sensitivity_noise_scale",
    "items":          "n_items",
}

INTEGER_SWEEP_PARAMS = {"horizon", "dim", "users", "items"}


def _coerce_sweep_value(param: str, value: float) -> int | float:
    """Return a CLI sweep value with the type expected by run_experiment."""
    if param not in INTEGER_SWEEP_PARAMS:
        return float(value)
    if not float(value).is_integer():
        raise ValueError(f"{param} sweep values must be integers; got {value}.")
    return int(value)


def _latest_matching_result_json(
    results_dir: Path,
    *,
    started_at: float,
    expected_config: dict,
) -> dict:
    """
    Load the result JSON written for the current condition.

    This avoids accidentally reading an older or unrelated result when several
    files already exist in experiments/results/.
    """
    candidates = [
        path
        for path in results_dir.glob("*.json")
        if path.stat().st_mtime >= started_at - 1.0
    ]
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        if all(config.get(key) == value for key, value in expected_config.items()):
            return data
    raise RuntimeError(
        "Could not find the result JSON for the current sweep condition. "
        "The individual experiment may not have saved successfully."
    )


# ---------------------------------------------------------------------------
# Summary table across the sweep
# ---------------------------------------------------------------------------


def _sweep_summary_lines(
    param_name: str,
    sweep_values: list[float],
    all_results: list[dict[str, PolicyMetrics]],
    all_est_errors: list[dict[str, float]],
    policies: list[str],
    all_axis_errors: list[dict[str, float]] | None = None,
) -> list[str]:
    """
    One row per (sweep_value, policy) showing the key metrics.

    Sens.rate = mean_sensitive_asked / mean_n_asked.  This is unconfounded by
    dropout truncation (unlike raw sensitive_asked counts, which fall as users
    drop out early and never reach further sensitive questions).
    D-err(cplt) is D-error restricted to episodes that completed the full
    horizon, isolating information quality from the dropout effect.
    """
    CP = 22   # policy column
    CV = 10   # sweep-value column
    CF = 11   # float column

    cols = [
        (param_name,      CV, ">", ".3g"),
        ("Policy",        CP, "<", "s"),
        ("Dropout %",     CF, ">", ".1f"),
        ("Answered",      CF, ">", ".2f"),
        ("Sens.rate",     CF, ">", ".3f"),
        ("D-error",       CF, ">", ".4f"),
        ("D-err(cplt)",   CF, ">", ".4f"),
        ("Est. error",    CF, ">", ".4f"),
        *(
            [("Trait err.", CF, ">", ".4f")]
            if all_axis_errors is not None
            else []
        ),
    ]

    def _fmt(value: object, width: int, align: str, fmt: str) -> str:
        if fmt == "s":
            return f"{value:{align}{width}s}"
        return f"{float(value):{align}{width}{fmt}}"

    header_parts = [f"{label:{align}{width}s}" for label, width, align, _ in cols]
    header = "  ".join(header_parts)
    sep    = "-" * len(header)

    lines = [header, sep]
    for index, (val, pm_dict, err_dict) in enumerate(
        zip(sweep_values, all_results, all_est_errors)
    ):
        axis_err_dict = all_axis_errors[index] if all_axis_errors is not None else None
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
                pm.sensitive_rate,
                pm.mean_final_d_error,
                pm.mean_final_d_error_completed,
                err_dict.get(policy, float("nan")),
                *(
                    [axis_err_dict.get(policy, float("nan"))]
                    if axis_err_dict is not None
                    else []
                ),
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
    sensitivity_assignment: str = DEFAULT_SENSITIVITY_ASSIGNMENT,
    sensitive_axes: list[int] | None = DEFAULT_SENSITIVE_AXES,
    p_dropout: float    = DEFAULT_P_DROPOUT,
    sensitivity_noise_scale: float = DEFAULT_SENSITIVITY_NOISE_SCALE,
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
        sensitive-frac, sensitivity-noise, items.
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
    typed_values = [_coerce_sweep_value(param, value) for value in values]

    # Base kwargs shared across all conditions
    base_kwargs: dict = dict(
        dim=dim,
        n_items=n_items,
        n_categories=n_categories,
        sensitive_frac=sensitive_frac,
        sensitivity_assignment=sensitivity_assignment,
        sensitive_axes=sensitive_axes,
        p_dropout=p_dropout,
        sensitivity_noise_scale=sensitivity_noise_scale,
        n_users=n_users,
        horizon=horizon,
        seed_bank=seed_bank,
        seed_pop=seed_pop,
        seed_sim=seed_sim,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results:          list[dict[str, PolicyMetrics]] = []
    all_est_errors:       list[dict[str, float]]         = []
    all_sensitive_rates:  list[dict[str, float]]         = []
    all_d_error_completed: list[dict[str, float]]        = []
    all_axis_errors: list[dict[str, float]] | None = (
        [] if sensitivity_assignment == "high_trait_tail" else None
    )

    width = 60
    print()
    print("=" * width)
    print(f" SWEEP: {param} over {typed_values}")
    print("=" * width)
    if skip_myopic:
        print("  (myopic_exact excluded)")
    print()

    results_dir = Path(__file__).resolve().parent / "results"

    for val in typed_values:
        print(f"{'=' * 20}  {param} = {val}  {'=' * 20}")
        kwargs = {**base_kwargs, kwarg: val, "policies": active_policies}

        started_at = time.time()
        pm_dict = run_experiment(**kwargs)

        # Load metrics that run_experiment currently only returns via JSON.
        expected_sensitive_axes = _resolve_experiment_sensitive_axes(
            dim=kwargs["dim"],
            sensitivity_assignment=kwargs["sensitivity_assignment"],
            sensitive_axes=kwargs["sensitive_axes"],
        )
        expected_config = {
            "dim": kwargs["dim"],
            "n_items": kwargs["n_items"],
            "n_categories": kwargs["n_categories"],
            "sensitive_frac": kwargs["sensitive_frac"],
            "sensitivity_assignment": kwargs["sensitivity_assignment"],
            "sensitive_axes": expected_sensitive_axes,
            "p_dropout_sens": kwargs["p_dropout"],
            "sensitivity_noise_scale": kwargs["sensitivity_noise_scale"],
            "n_users": kwargs["n_users"],
            "horizon": kwargs["horizon"],
            "seed_bank": kwargs["seed_bank"],
            "seed_population": kwargs["seed_pop"],
            "seed_simulate": kwargs["seed_sim"],
        }
        latest_json = _latest_matching_result_json(
            results_dir,
            started_at=started_at,
            expected_config=expected_config,
        )
        est_errors = {
            policy: data["mean_estimation_error"]
            for policy, data in latest_json["policies"].items()
        }
        axis_errors = {
            policy: data.get("mean_sensitive_trait_error", float("nan"))
            for policy, data in latest_json["policies"].items()
        }
        sensitive_rates = {
            policy: data.get("sensitive_rate", float("nan"))
            for policy, data in latest_json["policies"].items()
        }
        d_error_completed = {
            policy: data.get("mean_final_d_error_completed", float("nan"))
            for policy, data in latest_json["policies"].items()
        }

        all_results.append(pm_dict)
        all_est_errors.append(est_errors)
        all_sensitive_rates.append(sensitive_rates)
        all_d_error_completed.append(d_error_completed)
        if all_axis_errors is not None:
            all_axis_errors.append(axis_errors)

    # ---- combined summary ----
    print()
    print("=" * width)
    print(f" SWEEP SUMMARY: {param}")
    print("=" * width)
    print()
    summary_lines = _sweep_summary_lines(
        param_name=param,
        sweep_values=list(typed_values),
        all_results=all_results,
        all_est_errors=all_est_errors,
        policies=active_policies,
        all_axis_errors=all_axis_errors,
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
        "sweep_values": list(typed_values),
        "fixed_config": {k: v for k, v in base_kwargs.items() if k != kwarg},
        "conditions": [
            {
                "value": val,
                "policies": {
                    policy: {
                        "dropout_rate": pm_dict[policy].dropout_rate,
                        "mean_n_answered": pm_dict[policy].mean_n_answered,
                        "mean_n_asked": pm_dict[policy].mean_n_asked,
                        "mean_sensitive_asked": pm_dict[policy].mean_sensitive_asked,
                        "sensitive_rate": all_sensitive_rates[index].get(policy, float("nan")),
                        "mean_final_d_error": pm_dict[policy].mean_final_d_error,
                        "mean_final_d_error_completed": all_d_error_completed[index].get(policy, float("nan")),
                        "mean_logdet_reduction": pm_dict[policy].mean_logdet_reduction,
                        "mean_estimation_error": est_errors.get(policy, float("nan")),
                        "mean_sensitive_trait_error": (
                            all_axis_errors[index].get(policy, float("nan"))
                            if all_axis_errors is not None
                            else None
                        ),
                    }
                    for policy in active_policies
                    if policy in pm_dict
                },
            }
            for index, (val, pm_dict, est_errors) in enumerate(
                zip(typed_values, all_results, all_est_errors)
            )
        ],
    }
    json_path = results_dir / f"{stem}.json"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

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

    repo_root = Path(__file__).resolve().parents[1]
    print("Saved sweep summary:")
    print(f"  JSON     : {json_path.relative_to(repo_root)}")
    print(f"  Markdown : {md_path.relative_to(repo_root)}")


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
    p.add_argument("--sensitivity-assignment", type=str,
                   default=DEFAULT_SENSITIVITY_ASSIGNMENT,
                   choices=["random", "axis_aligned", "high_trait_tail"])
    p.add_argument("--sensitive-axes", type=int, nargs="+", default=DEFAULT_SENSITIVE_AXES)
    p.add_argument("--dropout",        type=float, default=DEFAULT_P_DROPOUT)
    p.add_argument("--sensitivity-noise", "--sensitive-noise", type=float,
                   dest="sensitivity_noise",
                   default=DEFAULT_SENSITIVITY_NOISE_SCALE)
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
        sensitivity_assignment=args.sensitivity_assignment,
        sensitive_axes=args.sensitive_axes,
        p_dropout=args.dropout,
        sensitivity_noise_scale=args.sensitivity_noise,
        n_users=args.users,
        horizon=args.horizon,
        seed_bank=args.seed_bank,
        seed_pop=args.seed_pop,
        seed_sim=args.seed_sim,
    )
