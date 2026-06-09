"""
policy_curves.py
================
Extracts optimal decision thresholds from the solved value function and plots
them as policy curves over a grid of (r1, r2) values.

What is plotted
---------------
For each (r1, r2) combination:

  * Class 1 outsourcing threshold  x*
        The lowest queue state at which the optimal action switches from
        "queue the customer" to "outsource immediately."
        - Below x*  → accept / queue
        - At or above x* → outsource

  * Class 2 service-initiation threshold  x**
        The highest idle-server state (x < 0) at which Class 2 proactively
        starts serving (rather than staying idle).
        - Only meaningful for states x < 0 (idle servers)

Two figures are produced (one per Bellman variant):
  Graph 1  — heatmap of x* over the (r1, r2) grid
  Graph 2  — line plots of x* as r1 varies (one line per r2) and vice-versa

Usage
-----
    python policy_curves.py
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── model code (copy from your main module or import it) ──────────────────────

from simulate_curves import (
    QueueingSystemParams,
    build_state_space,
    get_state_index,
    class1_decision_value,
    class2_decision_value,
    bellman_update_apriori,
    bellman_update_aposteriori,
    solve,
)

# ── policy extraction ─────────────────────────────────────────────────────────

def extract_policy(
    value_function: np.ndarray,
    params: QueueingSystemParams,
) -> dict:
    """
    Given a converged value function, returns the optimal decision at every
    state and the key threshold indices.

    Returns
    -------
    dict with keys:
        "class1_actions"   : list of "QUEUE" | "OUTSOURCE" | "ACCEPT" per state
        "class2_actions"   : list of "INITIATE" | "IDLE" per state
        "outsource_thresh" : lowest x >= 0 where Class 1 chooses OUTSOURCE
                             (None if never outsourced within queue range)
        "initiate_thresh"  : highest x < 0 where Class 2 chooses INITIATE
                             (None if never initiated)
    """
    state_space = build_state_space(params)
    V = value_function

    class1_actions = []
    class2_actions = []

    outsource_thresh = None   # first state where outsourcing is optimal
    initiate_thresh  = None   # last (highest) idle state where initiation wins

    for state in state_space:
        idx = get_state_index(state, params)

        # ── Class 1 decision ───────────────────────────────────────────────
        if state < 0:
            # Idle server: always accept
            class1_actions.append("ACCEPT")
        else:
            next_q_val  = V[idx + 1] if idx + 1 < len(V) else V[idx]
            outsource_val = V[idx] - params.outsourcing_cost
            if next_q_val >= outsource_val:
                class1_actions.append("QUEUE")
            else:
                class1_actions.append("OUTSOURCE")
                if outsource_thresh is None:
                    outsource_thresh = state   # first time we outsource

        # ── Class 2 decision ───────────────────────────────────────────────
        if state < 0:
            idle_val     = V[idx]
            initiate_val = (V[idx + 1] if idx + 1 < len(V) else V[idx]) + params.class2_reward
            if initiate_val >= idle_val:
                class2_actions.append("INITIATE")
                initiate_thresh = state        # keep updating → last idle state
            else:
                class2_actions.append("IDLE")
        else:
            class2_actions.append("IDLE")

    return {
        "state_space"      : state_space,
        "class1_actions"   : class1_actions,
        "class2_actions"   : class2_actions,
        "outsource_thresh" : outsource_thresh,
        "initiate_thresh"  : initiate_thresh,
    }


# ── worker functions (top-level for pickling) ─────────────────────────────────

def _worker_apriori(args: tuple) -> tuple:
    base_params, r1, r2 = args
    params = replace(base_params, class1_reward=r1, class2_reward=r2)
    V, _ = solve(bellman_update_apriori, params)
    pol  = extract_policy(V, params)
    return r1, r2, pol["outsource_thresh"], pol["initiate_thresh"]


def _worker_aposteriori(args: tuple) -> tuple:
    base_params, r1, r2 = args
    params = replace(base_params, class1_reward=r1, class2_reward=r2)
    V, _ = solve(bellman_update_aposteriori, params)
    pol  = extract_policy(V, params)
    return r1, r2, pol["outsource_thresh"], pol["initiate_thresh"]


def run_sweep_parallel(
    worker_fn: Callable,
    base_params: QueueingSystemParams,
    r1_values: list[float],
    r2_values: list[float],
    n_workers: int | None = None,
) -> list[tuple]:
    job_args = [
        (base_params, r1, r2)
        for r1 in r1_values
        for r2 in r2_values
    ]
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(worker_fn, job_args)
    return results


# ── plotting helpers ──────────────────────────────────────────────────────────

def _build_grid(
    results: list[tuple],
    r1_values: list[float],
    r2_values: list[float],
    field: str,          # "outsource" or "initiate"
) -> np.ndarray:
    """
    Assembles a 2-D array shaped (len(r1_values), len(r2_values)).
    Rows = r1, columns = r2.
    Missing/None entries are replaced with NaN.
    """
    lookup = {
        (round(r1, 8), round(r2, 8)): (ot, it)
        for r1, r2, ot, it in results
    }
    grid = np.full((len(r1_values), len(r2_values)), np.nan)
    for i, r1 in enumerate(r1_values):
        for j, r2 in enumerate(r2_values):
            entry = lookup.get((round(r1, 8), round(r2, 8)))
            if entry is None:
                continue
            val = entry[0] if field == "outsource" else entry[1]
            grid[i, j] = val if val is not None else np.nan
    return grid


def plot_policy_curves(
    results: list[tuple],
    r1_values: list[float],
    r2_values: list[float],
    title_prefix: str,
    filepath_prefix: str,
) -> None:
    """
    Produces two figures:

    Figure A  —  2×2 panel
        Top-left   : heatmap of outsourcing threshold x* over (r1, r2)
        Top-right  : heatmap of initiation threshold x** over (r1, r2)
        Bottom-left : line plot of x* vs r1, one line per r2
        Bottom-right: line plot of x* vs r2, one line per r1
    """
    grid_out  = _build_grid(results, r1_values, r2_values, "outsource")
    grid_init = _build_grid(results, r1_values, r2_values, "initiate")

    # ── Figure: heatmaps ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{title_prefix} — Optimal Decision Thresholds", fontsize=13, fontweight="bold")

    r1_arr = np.array(r1_values)
    r2_arr = np.array(r2_values)

    for ax, grid, label, cmap in [
        (axes[0], grid_out,  "Class 1 outsourcing threshold  x*\n(queue → outsource crossover)",  "plasma"),
        (axes[1], grid_init, "Class 2 initiation threshold  x**\n(highest idle state with proactive service)", "viridis"),
    ]:
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[r2_arr[0], r2_arr[-1], r1_arr[0], r1_arr[-1]],
        )
        ax.set_xlabel("r₂  (Class 2 reward)", fontsize=10)
        ax.set_ylabel("r₁  (Class 1 reward)", fontsize=10)
        ax.set_title(label, fontsize=10)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("State x", fontsize=9)

        # Annotate each cell
        for i, r1 in enumerate(r1_values):
            for j, r2 in enumerate(r2_values):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(
                        r2, r1, f"{int(val)}",
                        ha="center", va="center",
                        fontsize=8, color="white",
                        fontweight="bold",
                    )

    fig.tight_layout()
    path_a = f"{filepath_prefix}_heatmap.png"
    fig.savefig(path_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path_a}")

    # ── Figure: line plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{title_prefix} — Outsourcing Threshold  x*  vs Reward Parameters",
        fontsize=13, fontweight="bold",
    )

    cmap_r2 = plt.cm.viridis
    cmap_r1 = plt.cm.plasma

    # Left: x* vs r1, one line per r2
    ax = axes[0]
    for k, r2 in enumerate(r2_values):
        color = cmap_r2(k / max(len(r2_values) - 1, 1))
        y_vals = grid_out[:, k]           # rows = r1 index
        ax.plot(r1_arr, y_vals, "o-", color=color, lw=1.8, ms=5, label=f"r₂={r2:.1f}")
    ax.set_xlabel("r₁  (Class 1 reward)", fontsize=10)
    ax.set_ylabel("Outsourcing threshold  x*", fontsize=10)
    ax.set_title("x*  as r₁ varies  (per r₂)", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Right: x* vs r2, one line per r1
    ax = axes[1]
    for k, r1 in enumerate(r1_values):
        color = cmap_r1(k / max(len(r1_values) - 1, 1))
        y_vals = grid_out[k, :]           # cols = r2 index
        ax.plot(r2_arr, y_vals, "s-", color=color, lw=1.8, ms=5, label=f"r₁={r1:.1f}")
    ax.set_xlabel("r₂  (Class 2 reward)", fontsize=10)
    ax.set_ylabel("Outsourcing threshold  x*", fontsize=10)
    ax.set_title("x*  as r₂ varies  (per r₁)", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path_b = f"{filepath_prefix}_lines.png"
    fig.savefig(path_b, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path_b}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    BASE = QueueingSystemParams(
        arrival_rate=0.6,
        num_servers=2,
        service_rate=0.5,
        class2_arrival_rate=0.3,
        class1_reward=5.0,
        class2_reward=2.0,
        outsourcing_cost=1.5,
        congestion_sensitivity=0.1,
        max_queue_length=15,
        convergence_tolerance=1e-6,
        max_iterations=2000,
    )

    R1_VALUES = [2.0, 3.5, 5.0, 6.5, 8.0]
    R2_VALUES = [0.5, 1.5, 2.5, 3.5, 4.5]

    print("Running a priori sweep …")
    res_ap = run_sweep_parallel(_worker_apriori, BASE, R1_VALUES, R2_VALUES)
    plot_policy_curves(res_ap, R1_VALUES, R2_VALUES,
                       title_prefix="A Priori",
                       filepath_prefix="policy_apriori")

    print("Running a posteriori sweep …")
    res_post = run_sweep_parallel(_worker_aposteriori, BASE, R1_VALUES, R2_VALUES)
    plot_policy_curves(res_post, R1_VALUES, R2_VALUES,
                       title_prefix="A Posteriori",
                       filepath_prefix="policy_aposteriori")

    print("\nDone.")


if __name__ == "__main__":
    main()
