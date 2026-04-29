"""
experiments/policy_comparison.py
=================================
Compare five adaptive questionnaire policies under a realistic dropout model.

Policies compared
-----------------
  fixed               -- items presented in bank order (natural ordering)
  random              -- uniform random item selection at each step
  myopic_exact        -- exact one-step information-gain lookahead
  surrogate_unweighted-- log(1 + a^T Sigma a)  (ignores dropout risk)
  surrogate_weighted  -- log(1 + p_stay * a^T Sigma a)  (dropout-aware)

Usage
-----
    # Use file defaults
    python -m experiments.policy_comparison

    # Override individual parameters
    python -m experiments.policy_comparison --dropout 0.20
    python -m experiments.policy_comparison --dim 6 --horizon 30 --users 1000
    python -m experiments.policy_comparison --dropout 0.10 --sensitive-frac 0.40
    python -m experiments.policy_comparison --sensitivity-noise 2.0

    # Full list of overridable parameters
    python -m experiments.policy_comparison --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Make sure the package is importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.belief import BeliefState
from src.metrics import PolicyMetrics, aggregate_policy_metrics, mean_estimation_error
from src.policies import make_sensitive_constant_stay_prob
from src.simulate import simulate_population
from src.synthetic import generate_user_population, synthetic_item_bank


# ---------------------------------------------------------------------------
# Default configuration  (all values overridable via CLI or run_experiment())
# ---------------------------------------------------------------------------

DEFAULT_DIM            = 2
DEFAULT_N_ITEMS        = 30
DEFAULT_N_CATEGORIES   = 4
DEFAULT_SENSITIVE_FRAC = 0.30
DEFAULT_P_DROPOUT      = 0.10
DEFAULT_SENSITIVITY_NOISE_SCALE = 0.0
DEFAULT_N_USERS        = 500
DEFAULT_HORIZON        = 12
DEFAULT_SENSITIVITY_ASSIGNMENT = "random"
DEFAULT_SENSITIVE_AXES = None
DEFAULT_SEED_BANK      = 0
DEFAULT_SEED_POP       = 1
DEFAULT_SEED_SIM       = 2

POLICIES: list[str] = [
    "fixed",
    "random",
    "myopic_exact",
    "surrogate_unweighted",
    "surrogate_weighted",
]


def _format_stem_number(value: float) -> str:
    """Format numeric config values compactly for result filenames."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


# ---------------------------------------------------------------------------
# Helpers  (all accept explicit config rather than reading module globals)
# ---------------------------------------------------------------------------


def _print_header(
    *,
    dim: int,
    n_items: int,
    n_sensitive: int,
    horizon: int,
    p_dropout: float,
    sensitivity_noise_scale: float,
    n_users: int,
    sensitivity_assignment: str,
    sensitive_axes: list[int] | None,
) -> None:
    width = 60
    print()
    print("=" * width)
    print(" SIMULATION SETUP")
    print("=" * width)
    print(f"  Users              : {n_users}")
    print(f"  Latent dimensions  : {dim}")
    print(f"  Horizon            : {horizon} questions")
    print(f"  Item bank          : {n_items} items  "
          f"({n_sensitive} sensitive, {100 * n_sensitive / n_items:.0f} %)")
    print(f"  Sensitive labels   : {sensitivity_assignment}")
    if sensitive_axes is not None:
        print(f"  Sensitive axes     : {sensitive_axes}")
    print(f"  Sensitivity level  : 1.0 (fixed, all sensitive items)")
    print(f"  Dropout prob.      : {p_dropout * 100:.0f} % per sensitive question")
    if sensitivity_noise_scale != 0.0:
        print(f"  Sensitivity noise  : {sensitivity_noise_scale:g}x level multiplier")
    print(f"  Prior              : N(0, I_{dim})")
    print(f"  Batch (s)          : total wall time for all {n_users} episodes")
    print("=" * width)
    print()


def _table_lines(
    results: dict[str, PolicyMetrics],
    elapsed: dict[str, float],
    est_errors: dict[str, float],
    sensitive_trait_errors: dict[str, float] | None = None,
) -> list[str]:
    """Return the results table as a list of plain-text lines."""
    CP = 24
    CN = 10
    CF = 11

    cols = [
        ("Policy",       CP, "<", "s"),
        ("Dropouts",     CN, ">", "d"),
        ("Dropout %",    CF, ">", ".1f"),
        ("Answered",     CF, ">", ".2f"),
        ("Asked",        CF, ">", ".2f"),
        ("Sens. asked",  CF, ">", ".2f"),
        ("D-error",      CF, ">", ".4f"),
        ("Est. error",   CF, ">", ".4f"),
        *(
            [("Trait err.", CF, ">", ".4f")]
            if sensitive_trait_errors is not None
            else []
        ),
        ("Batch (s)",    CF, ">", ".1f"),
        ("ep/s",         CF, ">", ".1f"),
    ]

    def _fmt_cell(value: object, width: int, align: str, fmt: str) -> str:
        if fmt == "s":
            return f"{value:{align}{width}s}"
        if fmt == "d":
            return f"{int(value):{align}{width}d}"
        return f"{float(value):{align}{width}{fmt}}"

    header_parts = [f"{label:{align}{width}s}" for label, width, align, _ in cols]
    header = "  ".join(header_parts)
    sep    = "-" * len(header)

    rows = [
        (
            pm.policy_name,
            pm.dropout_count,
            pm.dropout_rate * 100,
            pm.mean_n_answered,
            pm.mean_n_asked,
            pm.mean_sensitive_asked,
            pm.mean_final_d_error,
            est_errors.get(pm.policy_name, float("nan")),
            *(
                [sensitive_trait_errors.get(pm.policy_name, float("nan"))]
                if sensitive_trait_errors is not None
                else []
            ),
            elapsed.get(pm.policy_name, float("nan")),
            pm.n_episodes / elapsed.get(pm.policy_name, float("nan")),
        )
        for pm in results.values()
    ]

    lines = [header, sep]
    for values in rows:
        cells = [
            _fmt_cell(v, width, align, fmt)
            for v, (_, width, align, fmt) in zip(values, cols)
        ]
        lines.append("  ".join(cells))
    lines.append(sep)
    return lines


def _print_table(
    results: dict[str, PolicyMetrics],
    elapsed: dict[str, float],
    est_errors: dict[str, float],
    sensitive_trait_errors: dict[str, float] | None = None,
) -> None:
    for line in _table_lines(results, elapsed, est_errors, sensitive_trait_errors):
        print(line)
    print()


def _mean_sensitive_trait_error(results, theta_trues, axes: list[int]) -> float:
    """Mean absolute posterior-mean error on selected latent axes."""
    return float(
        np.mean(
            [
                np.mean(np.abs(result.final_belief.mu[axes] - theta_true[axes]))
                for result, theta_true in zip(results, theta_trues)
            ]
        )
    )


def _resolve_experiment_sensitive_axes(
    *,
    dim: int,
    sensitivity_assignment: str,
    sensitive_axes: list[int] | None,
) -> list[int] | None:
    if sensitivity_assignment != "high_trait_tail":
        return None
    if sensitive_axes is not None:
        return list(dict.fromkeys(sensitive_axes))
    return [dim - 1]


def save_results(
    *,
    dim: int,
    n_items: int,
    n_categories: int,
    sensitive_frac: float,
    sensitivity_assignment: str,
    sensitive_axes: list[int] | None,
    n_sensitive: int,
    p_dropout: float,
    sensitivity_noise_scale: float,
    n_users: int,
    horizon: int,
    seed_bank: int,
    seed_pop: int,
    seed_sim: int,
    results: dict[str, PolicyMetrics],
    elapsed: dict[str, float],
    est_errors: dict[str, float],
    sensitive_trait_errors: dict[str, float] | None = None,
) -> Path:
    """
    Write a Markdown summary and a JSON data file to experiments/results/.
    Returns the path of the Markdown file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_dim{dim}_h{horizon}_p{int(p_dropout * 100)}_n{n_users}"
    if sensitivity_noise_scale != 0.0:
        stem = f"{stem}_sn{_format_stem_number(sensitivity_noise_scale)}"

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # ---- Markdown ----
    md_lines: list[str] = [
        f"# Policy Comparison -- {timestamp}",
        "",
        "## Setup",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Users | {n_users} |",
        f"| Latent dimensions | {dim} |",
        f"| Horizon | {horizon} questions |",
        f"| Item bank | {n_items} items ({n_sensitive} sensitive, "
        f"{100 * n_sensitive / n_items:.0f} %) |",
        f"| Sensitive labels | {sensitivity_assignment} |",
        *(
            [f"| Sensitive axes | {sensitive_axes} |"]
            if sensitive_axes is not None
            else []
        ),
        "| Sensitivity level | 1.0 (fixed) |",
        f"| Dropout prob. | {p_dropout * 100:.0f} % per sensitive question |",
        *(
            [f"| Sensitivity noise scale | {sensitivity_noise_scale:g} |"]
            if sensitivity_noise_scale != 0.0
            else []
        ),
        f"| Prior | N(0, I_{dim}) |",
        "",
        "## Results",
        "",
        "```",
        *_table_lines(results, elapsed, est_errors, sensitive_trait_errors),
        "```",
        "",
        "## Notes",
        "",
        "_Add your observations here._",
        "",
    ]

    md_path = results_dir / f"{stem}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # ---- JSON ----
    json_data = {
        "timestamp": timestamp,
        "config": {
            "dim": dim,
            "n_items": n_items,
            "n_categories": n_categories,
            "sensitive_frac": sensitive_frac,
            "sensitivity_assignment": sensitivity_assignment,
            "sensitive_axes": sensitive_axes,
            "n_sensitive": n_sensitive,
            "p_dropout_sens": p_dropout,
            "sensitivity_noise_scale": sensitivity_noise_scale,
            "n_users": n_users,
            "horizon": horizon,
            "seed_bank": seed_bank,
            "seed_population": seed_pop,
            "seed_simulate": seed_sim,
        },
        "policies": {
            policy: {
                "n_episodes": pm.n_episodes,
                "dropout_count": pm.dropout_count,
                "dropout_rate": pm.dropout_rate,
                "mean_n_answered": pm.mean_n_answered,
                "mean_n_asked": pm.mean_n_asked,
                "mean_sensitive_asked": pm.mean_sensitive_asked,
                "sensitive_rate": pm.sensitive_rate,
                "mean_final_d_error": pm.mean_final_d_error,
                "mean_final_d_error_completed": pm.mean_final_d_error_completed,
                "mean_final_logdet": pm.mean_final_logdet,
                "mean_logdet_reduction": pm.mean_logdet_reduction,
                "mean_estimation_error": est_errors.get(policy, float("nan")),
                "mean_sensitive_trait_error": (
                    sensitive_trait_errors.get(policy, float("nan"))
                    if sensitive_trait_errors is not None
                    else None
                ),
                "batch_seconds": elapsed.get(policy, float("nan")),
                "episodes_per_second": pm.n_episodes / elapsed.get(policy, float("nan")),
            }
            for policy, pm in results.items()
        },
    }

    json_path = results_dir / f"{stem}.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    return md_path


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment(
    *,
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
    policies: list[str] | None = None,
) -> dict[str, PolicyMetrics]:
    """
    Run the policy comparison experiment and save results to experiments/results/.

    All parameters default to the module-level DEFAULT_* constants, so calling
    ``run_experiment()`` is identical to running the script with no CLI flags.
    Can also be called programmatically from a sweep script:

        from experiments.policy_comparison import run_experiment
        for p in [0.05, 0.10, 0.20]:
            run_experiment(p_dropout=p, dim=6)
    """
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
    n_sensitive = sum(item.is_sensitive for item in item_bank)
    resolved_sensitive_axes = _resolve_experiment_sensitive_axes(
        dim=dim,
        sensitivity_assignment=sensitivity_assignment,
        sensitive_axes=sensitive_axes,
    )
    resolved_multi_axes = (
        resolved_sensitive_axes if resolved_sensitive_axes is not None else None
    )

    _print_header(
        dim=dim,
        n_items=n_items,
        n_sensitive=n_sensitive,
        horizon=horizon,
        p_dropout=p_dropout,
        sensitivity_noise_scale=sensitivity_noise_scale,
        n_users=n_users,
        sensitivity_assignment=sensitivity_assignment,
        sensitive_axes=resolved_multi_axes,
    )

    stay_prob_fn = make_sensitive_constant_stay_prob(
        p_stay_sensitive=1.0 - p_dropout,
        p_stay_normal=1.0,
    )
    prior       = BeliefState(mu=np.zeros(dim), Sigma=np.eye(dim))
    theta_trues = generate_user_population(n_users=n_users, dim=dim, rng_seed=seed_pop)

    active_policies = policies if policies is not None else POLICIES

    # ---- run each policy ----
    policy_metrics: dict[str, PolicyMetrics] = {}
    policy_elapsed: dict[str, float]         = {}
    policy_est_err: dict[str, float]         = {}
    policy_axis_err: dict[str, float] | None = (
        {} if resolved_sensitive_axes is not None else None
    )

    for policy_idx, policy in enumerate(active_policies):
        rng = np.random.default_rng([seed_sim, policy_idx])

        t0  = time.perf_counter()

        results = simulate_population(
            theta_trues=theta_trues,
            prior_belief=prior,
            item_bank=item_bank,
            strategy=policy,
            horizon=horizon,
            stay_prob_fn=stay_prob_fn,
            rng=rng,
            sensitivity_noise_scale=sensitivity_noise_scale,
        )

        elapsed = time.perf_counter() - t0
        pm      = aggregate_policy_metrics(results, policy_name=policy, item_bank=item_bank)
        est_err = mean_estimation_error(results, theta_trues)
        axis_err = (
            _mean_sensitive_trait_error(results, theta_trues, resolved_sensitive_axes)
            if resolved_sensitive_axes is not None
            else None
        )

        policy_metrics[policy] = pm
        policy_elapsed[policy] = elapsed
        policy_est_err[policy] = est_err
        if policy_axis_err is not None and axis_err is not None:
            policy_axis_err[policy] = axis_err

        axis_msg = f"trait_err={axis_err:.4f}  " if axis_err is not None else ""
        print(
            f"  {policy:<22}  "
            f"dropout={pm.dropout_rate * 100:5.1f} %  "
            f"answered={pm.mean_n_answered:5.2f}  "
            f"d_error={pm.mean_final_d_error:6.4f}  "
            f"est_err={est_err:.4f}  "
            f"{axis_msg}"
            f"({elapsed:.1f} s)"
        )

    # ---- summary table ----
    print()
    print(" POLICY COMPARISON RESULTS")
    print()
    _print_table(policy_metrics, policy_elapsed, policy_est_err, policy_axis_err)

    # ---- save to disk ----
    md_path = save_results(
        dim=dim,
        n_items=n_items,
        n_categories=n_categories,
        sensitive_frac=sensitive_frac,
        sensitivity_assignment=sensitivity_assignment,
        sensitive_axes=resolved_multi_axes,
        n_sensitive=n_sensitive,
        p_dropout=p_dropout,
        sensitivity_noise_scale=sensitivity_noise_scale,
        n_users=n_users,
        horizon=horizon,
        seed_bank=seed_bank,
        seed_pop=seed_pop,
        seed_sim=seed_sim,
        results=policy_metrics,
        elapsed=policy_elapsed,
        est_errors=policy_est_err,
        sensitive_trait_errors=policy_axis_err,
    )
    print(f"Results saved to {md_path.relative_to(Path(__file__).resolve().parents[1])}")
    print(f"  JSON: {md_path.with_suffix('.json').name}")

    return policy_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare adaptive questionnaire policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dim",            type=int,   default=DEFAULT_DIM,
                   help="Number of latent trait dimensions")
    p.add_argument("--items",          type=int,   default=DEFAULT_N_ITEMS,
                   help="Total items in the bank")
    p.add_argument("--sensitive-frac", type=float, default=DEFAULT_SENSITIVE_FRAC,
                   help="Fraction of items that are sensitive (0–1)")
    p.add_argument("--sensitivity-assignment", type=str,
                   default=DEFAULT_SENSITIVITY_ASSIGNMENT,
                   choices=["random", "axis_aligned", "high_trait_tail"],
                   help="How items are marked sensitive")
    p.add_argument("--sensitive-axes", type=int, nargs="+", default=DEFAULT_SENSITIVE_AXES,
                   help="Latent axes used by --sensitivity-assignment high_trait_tail")
    p.add_argument("--dropout",        type=float, default=DEFAULT_P_DROPOUT,
                   help="Per-question dropout probability for sensitive items (0–1)")
    p.add_argument("--sensitivity-noise", "--sensitive-noise", type=float,
                   dest="sensitivity_noise",
                   default=DEFAULT_SENSITIVITY_NOISE_SCALE,
                   help="Extra response-noise scale for sensitive items")
    p.add_argument("--users",          type=int,   default=DEFAULT_N_USERS,
                   help="Number of synthetic users")
    p.add_argument("--horizon",        type=int,   default=DEFAULT_HORIZON,
                   help="Maximum questions per episode")
    p.add_argument("--seed-bank",      type=int,   default=DEFAULT_SEED_BANK,
                   help="RNG seed for item bank generation")
    p.add_argument("--seed-pop",       type=int,   default=DEFAULT_SEED_POP,
                   help="RNG seed for user population")
    p.add_argument("--seed-sim",       type=int,   default=DEFAULT_SEED_SIM,
                   help="RNG seed for episode simulation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
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
