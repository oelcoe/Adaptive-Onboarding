"""
experiments/policy_comparison.py
=================================
Compare five adaptive questionnaire policies under a realistic dropout model.

Policies compared
-----------------
  fixed               — items presented in bank order (natural ordering)
  random              — uniform random item selection at each step
  myopic_exact        — exact one-step information-gain lookahead
  surrogate_unweighted— log(1 + a^T Sigma a)  (ignores dropout risk)
  surrogate_weighted  — log(1 + p_stay * a^T Sigma a)  (dropout-aware)

Usage
-----
    python -m experiments.policy_comparison          # from repo root
    python experiments/policy_comparison.py          # as a plain script
"""

from __future__ import annotations

import sys
import time
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
# Experiment configuration
# ---------------------------------------------------------------------------

SEED_BANK       = 0       # item-bank RNG seed
SEED_POPULATION = 1       # user-population RNG seed
SEED_SIMULATE   = 2       # episode simulation RNG seed

DIM             = 6       # latent trait dimensions
N_ITEMS         = 30      # total items in the bank
N_CATEGORIES    = 4       # ordinal response categories per item
SENSITIVE_FRAC  = 0.30    # fraction of items that are sensitive (≈ 9/30)

P_DROPOUT_SENS  = 0.10    # per-question dropout probability for sensitive items
N_USERS         = 500     # synthetic population size
HORIZON         = 20      # maximum questions per episode

PRIOR_MU    = np.zeros(DIM)
PRIOR_SIGMA = np.eye(DIM)

POLICIES: list[str] = [
    "fixed",
    "random",
    "myopic_exact",
    "surrogate_unweighted",
    "surrogate_weighted",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_item_bank() -> list:
    return synthetic_item_bank(
        n_items=N_ITEMS,
        dim=DIM,
        n_categories=N_CATEGORIES,
        sensitive_fraction=SENSITIVE_FRAC,
        rng_seed=SEED_BANK,
        vary_sensitivity_levels=False,
    )


def _print_header(n_sensitive: int) -> None:
    width = 60
    print()
    print("=" * width)
    print(" SIMULATION SETUP")
    print("=" * width)
    print(f"  Users              : {N_USERS}")
    print(f"  Latent dimensions  : {DIM}")
    print(f"  Horizon            : {HORIZON} questions")
    print(f"  Item bank          : {N_ITEMS} items  "
          f"({n_sensitive} sensitive, {100 * n_sensitive / N_ITEMS:.0f} %)")
    print(f"  Sensitivity level  : 1.0 (fixed, all sensitive items)")
    print(f"  Dropout prob.      : {P_DROPOUT_SENS * 100:.0f} % per sensitive question")
    print(f"  Prior              : N(0, I_{DIM})")
    print(f"  Batch (s)          : total wall time for all {N_USERS} episodes")
    print("=" * width)
    print()


def _print_table(
    results: dict[str, PolicyMetrics],
    elapsed: dict[str, float],
    est_errors: dict[str, float],
) -> None:
    CP = 24   # policy name column
    CN = 10   # integer column
    CF = 11   # float column

    cols = [
        ("Policy",      CP, "<", "s"),
        ("Dropouts",    CN, ">", "d"),
        ("Dropout %",   CF, ">", ".1f"),
        ("Answered",    CF, ">", ".2f"),
        ("Asked",       CF, ">", ".2f"),
        ("Sens. asked",  CF, ">", ".2f"),
        ("D-error",      CF, ">", ".4f"),
        ("Est. error",   CF, ">", ".4f"),
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

    values_by_row: list[tuple] = [
        (
            pm.policy_name,
            pm.dropout_count,
            pm.dropout_rate * 100,
            pm.mean_n_answered,
            pm.mean_n_asked,
            pm.mean_sensitive_asked,
            pm.mean_final_d_error,
            est_errors.get(pm.policy_name, float("nan")),
            elapsed.get(pm.policy_name, float("nan")),
            pm.n_episodes / elapsed.get(pm.policy_name, float("nan")),
        )
        for pm in results.values()
    ]

    print(header)
    print(sep)
    for values in values_by_row:
        cells = [
            _fmt_cell(v, width, align, fmt)
            for v, (_, width, align, fmt) in zip(values, cols)
        ]
        print("  ".join(cells))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, PolicyMetrics]:
    item_bank   = _build_item_bank()
    n_sensitive = sum(item.is_sensitive for item in item_bank)

    _print_header(n_sensitive)

    stay_prob_fn = make_sensitive_constant_stay_prob(
        p_stay_sensitive=1.0 - P_DROPOUT_SENS,
        p_stay_normal=1.0,
    )
    prior = BeliefState(mu=PRIOR_MU, Sigma=PRIOR_SIGMA)
    theta_trues = generate_user_population(
        n_users=N_USERS,
        dim=DIM,
        rng_seed=SEED_POPULATION,
    )

    # ---- run each policy ----
    policy_metrics: dict[str, PolicyMetrics] = {}
    policy_elapsed: dict[str, float]         = {}
    policy_est_err: dict[str, float]         = {}

    for policy in POLICIES:
        rng = np.random.default_rng([SEED_SIMULATE, hash(policy) % (2**31)])
        t0  = time.perf_counter()

        results = simulate_population(
            theta_trues=theta_trues,
            prior_belief=prior,
            item_bank=item_bank,
            strategy=policy,
            horizon=HORIZON,
            stay_prob_fn=stay_prob_fn,
            rng=rng,
        )

        elapsed = time.perf_counter() - t0
        pm      = aggregate_policy_metrics(results, policy_name=policy, item_bank=item_bank)
        est_err = mean_estimation_error(results, theta_trues)

        policy_metrics[policy] = pm
        policy_elapsed[policy] = elapsed
        policy_est_err[policy] = est_err

        print(
            f"  {policy:<22}  "
            f"dropout={pm.dropout_rate * 100:5.1f} %  "
            f"answered={pm.mean_n_answered:5.2f}  "
            f"d_error={pm.mean_final_d_error:6.4f}  "
            f"est_err={est_err:.4f}  "
            f"({elapsed:.1f} s)"
        )

    # ---- summary table ----
    print()
    print(" POLICY COMPARISON RESULTS")
    print()
    _print_table(policy_metrics, policy_elapsed, policy_est_err)

    return policy_metrics


if __name__ == "__main__":
    run_experiment()
