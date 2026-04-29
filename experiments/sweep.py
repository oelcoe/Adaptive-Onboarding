"""
experiments/sweep.py
====================
Sweep a single parameter across a range of values and compare all five
policies at each point.  Results are saved individually per condition and
as a combined sweep summary.

We repeatedly call policy_comparison.run_experiment() for each value in ``values``, varying the parameter named ``param`` while keeping all others fixed.

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

# Import the policy comparison module, we do not build a seperate experiment for this sweep.
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

INTEGER_SWEEP_PARAMS = {"horizon", "dim", "users", "items"} # only allow integer values for the horizon, dim, users, and items.


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
        if path.stat().st_mtime >= started_at - 1.0 # only load the result JSON if it was started at least 1 second ago.
    ]
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True): # sort the candidates by the last modified time in descending order.
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("config", {}) # get the configuration from the result JSON.
        if all(config.get(key) == value for key, value in expected_config.items()):
            return data # return the result JSON if the configuration matches the expected configuration.
    raise RuntimeError( # raise an error if the result JSON is not found.   
        "Could not find the result JSON for the current sweep condition. "
        "The individual experiment may not have saved successfully."
    )


# ---------------------------------------------------------------------------
# Summary table across the sweep
# ---------------------------------------------------------------------------

# Build the summary table as a list of plain-text lines. We have one row per (sweep_value, policy) showing the key metrics.
# Sens.rate = mean_sensitive_asked / mean_n_asked.  This is unconfounded by dropout truncation (unlike raw sensitive_asked counts, which fall as users drop out early and never reach further sensitive questions).
# D-err(cplt) is D-error restricted to episodes that completed the full horizon, isolating information quality from the dropout effect.
# Est. error is the mean estimation error across all policies.
# Trait err. is the mean sensitive trait error across all policies. This is only relevant for the high_trait_tail sensitivity assignment.       
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

# Run the sweep for a single parameter across a range of values and compare all five policies at each point. Results are saved individually per condition and as a combined sweep summary.
def run_sweep(
    param: str, # the parameter to sweep.	
    values: list[float], # the values to sweep over.
    *,
    skip_myopic: bool = False, # if True, exclude myopic_exact from the run (saves ~40x compute per condition; useful for large sweeps).
    # fixed parameters (same defaults as policy_comparison.py)
    dim: int            = DEFAULT_DIM, # the number of latent dimensions.
    n_items: int        = DEFAULT_N_ITEMS, # the number of items in the item bank.
    n_categories: int   = DEFAULT_N_CATEGORIES, # the number of categories for the items.
    sensitive_frac: float = DEFAULT_SENSITIVE_FRAC,
    sensitivity_assignment: str = DEFAULT_SENSITIVITY_ASSIGNMENT, # the sensitivity assignment method.
    sensitive_axes: list[int] | None = DEFAULT_SENSITIVE_AXES, # the sensitive axes to use for the sensitivity assignment.
    p_dropout: float    = DEFAULT_P_DROPOUT,
    sensitivity_noise_scale: float = DEFAULT_SENSITIVITY_NOISE_SCALE, # the sensitivity noise scale.
    n_users: int        = DEFAULT_N_USERS, # the number of users in the synthetic user population.
    horizon: int        = DEFAULT_HORIZON,
    seed_bank: int      = DEFAULT_SEED_BANK, # the seed for the synthetic item bank.
    seed_pop: int       = DEFAULT_SEED_POP, # the seed for the synthetic user population.   
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
    if param not in SWEEP_PARAMS: # raise an error if the parameter is not one of the allowed parameters.
        raise ValueError(
            f"Unknown sweep parameter '{param}'. "
            f"Choose from: {', '.join(SWEEP_PARAMS)}"
        )

    kwarg = SWEEP_PARAMS[param]
    active_policies = [p for p in POLICIES if not (skip_myopic and p == "myopic_exact")] # list of policies to compare, excluding myopic_exact if skip_myopic is True.  
    typed_values = [_coerce_sweep_value(param, value) for value in values] # convert the values to the type expected by run_experiment.

    # Base kwargs shared across all conditions
    base_kwargs: dict = dict( # base keyword arguments shared across all conditions.
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # get the current timestamp.
    all_results:          list[dict[str, PolicyMetrics]] = [] # list to store the results for each condition.
    all_est_errors:       list[dict[str, float]]         = [] # list to store the estimation errors for each condition.
    all_sensitive_rates:  list[dict[str, float]]         = [] # list to store the sensitive rates for each condition.
    all_d_error_completed: list[dict[str, float]]        = [] # list to store the completed D-error for each condition.
    all_axis_errors: list[dict[str, float]] | None = (
        [] if sensitivity_assignment == "high_trait_tail" else None # list to store the sensitive trait errors for each condition. This is only relevant for the high_trait_tail sensitivity assignment.
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

    for val in typed_values: # iterate over the values to sweep over, one val is for example p_drop = 10%
        print(f"{'=' * 20}  {param} = {val}  {'=' * 20}")
        kwargs = {**base_kwargs, kwarg: val, "policies": active_policies} # create a dictionary of keyword arguments for the run_experiment function.	

        started_at = time.time() # get the current time.
        pm_dict = run_experiment(**kwargs) # run the experiment for the current value.

        # Load metrics that run_experiment currently only returns via JSON.
        expected_sensitive_axes = _resolve_experiment_sensitive_axes( # resolve the sensitive axes for the experiment.
            dim=kwargs["dim"],
            sensitivity_assignment=kwargs["sensitivity_assignment"],
            sensitive_axes=kwargs["sensitive_axes"],
        ) # expected_sensitive_axes is a list of the sensitive axes to use for the sensitivity assignment.
        expected_config = { # create a dictionary of the expected configuration.
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
        latest_json = _latest_matching_result_json( # load the latest result JSON for the current condition.    
            results_dir,
            started_at=started_at,
            expected_config=expected_config,
        )
        est_errors = { # create a dictionary of the estimation errors for each policy.
            policy: data["mean_estimation_error"]
            for policy, data in latest_json["policies"].items()
        }
        axis_errors = { # create a dictionary of the sensitive trait errors for each policy. This is only relevant for the high_trait_tail sensitivity assignment.
            policy: data.get("mean_sensitive_trait_error", float("nan"))
            for policy, data in latest_json["policies"].items()
        }
        sensitive_rates = { # create a dictionary of the sensitive rates for each policy.
            policy: data.get("sensitive_rate", float("nan"))
            for policy, data in latest_json["policies"].items()
        }
        d_error_completed = { # create a dictionary of the completed D-error for each policy.
            policy: data.get("mean_final_d_error_completed", float("nan"))
            for policy, data in latest_json["policies"].items()
        }

        all_results.append(pm_dict) # store the results for the current condition.
        all_est_errors.append(est_errors) # store the estimation errors for the current condition.
        all_sensitive_rates.append(sensitive_rates) # store the sensitive rates for the current condition.
        all_d_error_completed.append(d_error_completed) # store the completed D-error for the current condition.
        if all_axis_errors is not None:
            all_axis_errors.append(axis_errors) # store the sensitive trait errors for the current condition. This is only relevant for the high_trait_tail sensitivity assignment.

    # ---- combined summary ----
    print() # print a blank line.
    print("=" * width)
    print(f" SWEEP SUMMARY: {param}")
    print("=" * width)
    print() # print a blank line.
    summary_lines = _sweep_summary_lines( # build the summary table as a list of plain-text lines.  
        param_name=param,
        sweep_values=list(typed_values),
        all_results=all_results,
        all_est_errors=all_est_errors,
        policies=active_policies,
        all_axis_errors=all_axis_errors,
    )
    for line in summary_lines: # print the summary table.
        print(line)
    print() # print a blank line.   

    # ---- save combined JSON ----
    results_dir = Path(__file__).resolve().parent / "results" # get the results directory.
    stem = f"{timestamp}_sweep_{param.replace('-', '_')}" # create a stem for the combined results.
    combined = { # create a dictionary of the combined results.
        "timestamp": timestamp,
        "sweep_param": param,
        "sweep_values": list(typed_values),
        "fixed_config": {k: v for k, v in base_kwargs.items() if k != kwarg},
        "conditions": [
            {
                "value": val, # store the value for the current condition.
                "policies": {
                    policy: {
                        "dropout_rate": pm_dict[policy].dropout_rate, # store the dropout rate for the current condition.
                        "mean_n_answered": pm_dict[policy].mean_n_answered, # store the number of questions answered for the current condition.
                        "mean_n_asked": pm_dict[policy].mean_n_asked, # store the number of questions asked for the current condition.
                        "mean_sensitive_asked": pm_dict[policy].mean_sensitive_asked, # store the number of sensitive questions asked for the current condition.
                        "sensitive_rate": all_sensitive_rates[index].get(policy, float("nan")), # store the sensitive rate for the current condition.
                        "mean_final_d_error": pm_dict[policy].mean_final_d_error, # store the final D-error for the current condition.
                        "mean_final_d_error_completed": all_d_error_completed[index].get(policy, float("nan")), # store the completed D-error for the current condition.
                        "mean_logdet_reduction": pm_dict[policy].mean_logdet_reduction, # store the mean logdet reduction for the current condition.
                        "mean_estimation_error": est_errors.get(policy, float("nan")), # store the estimation error for the current condition.  
                        "mean_sensitive_trait_error": (
                            all_axis_errors[index].get(policy, float("nan")) # store the sensitive trait error for the current condition. This is only relevant for the high_trait_tail sensitivity assignment.
                            if all_axis_errors is not None
                            else None
                        ),
                    } # store the policy metrics for the current condition.
                    for policy in active_policies
                    if policy in pm_dict
                }, # store the policy metrics for the current condition.
            } # store the condition for the current value.
            for index, (val, pm_dict, est_errors) in enumerate(
                zip(typed_values, all_results, all_est_errors) # iterate over the values, results, and estimation errors.
            )
        ], # store the conditions for the current value.
    } # store the combined results.
    json_path = results_dir / f"{stem}.json" # create a path to the combined results JSON file.
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8") # save the combined results to the JSON file.

    # ---- save combined Markdown ----
    md_lines = [ # build the Markdown lines for the combined results.
        f"# Sweep: {param}  --  {timestamp}",
        "",
        "## Fixed configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in base_kwargs.items() if k != kwarg], # store the fixed configuration in the Markdown file.
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
    md_path = results_dir / f"{stem}.md" # create a path to the combined results Markdown file.
    md_path.write_text("\n".join(md_lines), encoding="utf-8") # save the combined results to the Markdown file.

    repo_root = Path(__file__).resolve().parents[1]
    print("Saved sweep summary:") # print a message that the sweep summary has been saved.
    print(f"  JSON     : {json_path.relative_to(repo_root)}") # print the path to the combined results JSON file.
    print(f"  Markdown : {md_path.relative_to(repo_root)}") # print the path to the combined results Markdown file.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace: # parse the command line arguments.
    p = argparse.ArgumentParser( # create an argument parser.   
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


if __name__ == "__main__": # run the sweep when the script is called directly.
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
