"""
simulate_curves.py
==================
Generates comparison curves for the queueing system by sweeping over
(r1, r2) parameter combinations using multiprocessing — no standard for-loops
at the parallelism level.

Usage
-----
    python simulate_curves.py

Outputs
-------
    curves_apriori.png      — value-function curves varying r1 and r2, a priori model
    curves_aposteriori.png  — same grid, a posteriori model
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, replace
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# ── paste (or import) your model code here ────────────────────────────────────

@dataclass
class QueueingSystemParams:
    arrival_rate: float
    num_servers: int
    service_rate: float
    class2_arrival_rate: float
    class1_reward: float
    class2_reward: float
    outsourcing_cost: float
    congestion_sensitivity: float
    max_queue_length: int
    convergence_tolerance: float
    max_iterations: int


def build_state_space(params: QueueingSystemParams):
    return list(range(-params.num_servers, params.max_queue_length + 1))


def get_state_index(state: int, params: QueueingSystemParams):
    return state + params.num_servers


def class1_decision_value(state, value_function, params):
    index = get_state_index(state, params)
    if state < 0:
        return value_function[index] + params.class1_reward
    next_queue_value = (
        value_function[index + 1] if index + 1 < len(value_function) else value_function[index]
    )
    outsource_value = value_function[index] - params.outsourcing_cost
    return max(next_queue_value, outsource_value)


def class2_decision_value(state, value_function, params):
    index = get_state_index(state, params)
    if state < 0:
        idle_value = value_function[index]
        initiate_service_value = (
            value_function[index + 1] if index + 1 < len(value_function) else value_function[index]
        ) + params.class2_reward
        return max(idle_value, initiate_service_value)
    return value_function[index]


def fil_waiting_time_operator(state, value_function, params):
    if state <= 0:
        return value_function[get_state_index(state, params)]
    x = state
    lam = params.arrival_rate
    gamma = params.class2_arrival_rate
    p = lam / (lam + gamma)
    q = gamma / (lam + gamma)
    value = 0.0
    for h in range(x):
        prob = p * (q ** h)
        next_state = max(0, x - h)
        value += prob * value_function[get_state_index(next_state, params)]
    value += (q ** x) * value_function[get_state_index(0, params)]
    return value


def bellman_update_apriori(value_function, params):
    normalization = params.arrival_rate + params.num_servers * params.service_rate
    arrival_prob = params.arrival_rate / normalization
    service_prob = params.service_rate / normalization
    new_value_function = np.zeros_like(value_function)
    state_space = build_state_space(params)
    for i, state in enumerate(state_space):
        busy_servers = min(params.num_servers, params.num_servers + state)
        effective_service_rate = busy_servers * service_prob
        class1_value = class1_decision_value(state, value_function, params)
        class2_value = class2_decision_value(state, value_function, params)
        previous_service_value = (
            class2_decision_value(state - 1, value_function, params)
            if state - 1 >= -params.num_servers else 0.0
        )
        congestion_adjustment = (
            params.class1_reward
            * (1 - params.congestion_sensitivity * state / (params.num_servers * service_prob))
            if state > 0 else 0.0
        )
        new_value_function[i] = (
            arrival_prob * class1_value
            + effective_service_rate * (previous_service_value + congestion_adjustment)
            + (1 - arrival_prob - effective_service_rate) * class2_value
        )
    return new_value_function


def bellman_update_aposteriori(value_function, params):
    normalization = (
        params.arrival_rate
        + params.num_servers * params.service_rate
        + params.class2_arrival_rate
    )
    arrival_prob = params.arrival_rate / normalization
    service_prob = params.service_rate / normalization
    class2_prob = params.class2_arrival_rate / normalization
    new_value_function = np.zeros_like(value_function)
    state_space = build_state_space(params)
    for i, state in enumerate(state_space):
        if state <= 0:
            class1_value = (
                value_function[i] + params.class1_reward
                if state < 0
                else class1_decision_value(state, value_function, params)
            )
            class2_value = class2_decision_value(state, value_function, params)
            effective_service_rate = (params.num_servers + state) * service_prob
            previous_service_value = (
                class2_decision_value(state - 1, value_function, params)
                if state - 1 >= -params.num_servers else 0.0
            )
            new_value_function[i] = (
                arrival_prob * class1_value
                + effective_service_rate * previous_service_value
                + (1 - arrival_prob - effective_service_rate) * class2_value
            )
        else:
            class1_value = class1_decision_value(state, value_function, params)
            fil_value = fil_waiting_time_operator(state, value_function, params)
            class2_value = class2_decision_value(state, value_function, params)
            congestion_adjustment = params.class1_reward * (
                1 - params.congestion_sensitivity * state / max(class2_prob, 1e-9)
            )
            new_value_function[i] = (
                class2_prob * class1_value
                + params.num_servers * service_prob * (fil_value + congestion_adjustment)
                + (1 - class2_prob - params.num_servers * service_prob) * class2_value
            )
    return new_value_function


def solve(bellman_fn: Callable, params: QueueingSystemParams):
    V = np.zeros(len(build_state_space(params)))
    history = []
    for _ in range(params.max_iterations):
        V_next = bellman_fn(V, params)
        error = float(np.max(np.abs(V_next - V)))
        history.append(error)
        V = V_next
        if error < params.convergence_tolerance:
            break
    return V, history


# ── worker functions (must be top-level for pickling) ─────────────────────────

def _run_apriori(args: tuple) -> tuple:
    """
    Worker for a priori Bellman.
    Returns (r1, r2, value_function, convergence_history).
    """
    base_params, r1, r2 = args
    params = replace(base_params, class1_reward=r1, class2_reward=r2)
    V, hist = solve(bellman_update_apriori, params)
    return r1, r2, V, hist


def _run_aposteriori(args: tuple) -> tuple:
    """
    Worker for a posteriori Bellman.
    Returns (r1, r2, value_function, convergence_history).
    """
    base_params, r1, r2 = args
    params = replace(base_params, class1_reward=r1, class2_reward=r2)
    V, hist = solve(bellman_update_aposteriori, params)
    return r1, r2, V, hist


# ── parallel sweep ─────────────────────────────────────────────────────────────

def run_sweep_parallel(
    worker_fn: Callable,
    base_params: QueueingSystemParams,
    r1_values: list[float],
    r2_values: list[float],
    n_workers: int | None = None,
) -> list[tuple]:
    """
    Runs `worker_fn` over every (r1, r2) combination in parallel.

    Parameters
    ----------
    worker_fn   : top-level callable that accepts (base_params, r1, r2)
    base_params : baseline parameter set (r1/r2 will be overridden per job)
    r1_values   : list of class-1 reward values to sweep
    r2_values   : list of class-2 reward values to sweep
    n_workers   : number of worker processes (default = CPU count)

    Returns
    -------
    List of (r1, r2, value_function, history) tuples in completion order.
    """
    job_args = [
        (base_params, r1, r2)
        for r1 in r1_values
        for r2 in r2_values
    ]
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(worker_fn, job_args)
    return results


# ── plotting ───────────────────────────────────────────────────────────────────

def _plot_value_functions(
    results: list[tuple],
    state_space: list[int],
    r1_values: list[float],
    r2_values: list[float],
    title: str,
    filepath: str,
) -> None:
    """
    Two-panel figure:
      Left  — vary r1 while fixing r2 at its median value.
      Right — vary r2 while fixing r1 at its median value.
    """
    r2_fixed = float(np.median(r2_values))
    r1_fixed = float(np.median(r1_values))

    # index results for fast lookup
    lookup: dict[tuple, np.ndarray] = {
        (round(r1, 6), round(r2, 6)): V
        for r1, r2, V, _ in results
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    cmap_r1 = plt.cm.plasma
    cmap_r2 = plt.cm.viridis

    # ── left panel: vary r1 ────────────────────────────────────────────────
    ax = axes[0]
    for k, r1 in enumerate(r1_values):
        key = (round(r1, 6), round(r2_fixed, 6))
        if key not in lookup:
            continue
        color = cmap_r1(k / max(len(r1_values) - 1, 1))
        ax.plot(state_space, lookup[key], color=color, lw=1.6, label=f"r₁={r1:.1f}")

    ax.set_title(f"Varying r₁  (r₂ = {r2_fixed:.1f} fixed)", fontsize=11)
    ax.set_xlabel("System state  x")
    ax.set_ylabel("Value function  V(x)")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axvline(0, color="grey", lw=0.8, ls=":")

    # ── right panel: vary r2 ───────────────────────────────────────────────
    ax = axes[1]
    for k, r2 in enumerate(r2_values):
        key = (round(r1_fixed, 6), round(r2, 6))
        if key not in lookup:
            continue
        color = cmap_r2(k / max(len(r2_values) - 1, 1))
        ax.plot(state_space, lookup[key], color=color, lw=1.6, label=f"r₂={r2:.1f}")

    ax.set_title(f"Varying r₂  (r₁ = {r1_fixed:.1f} fixed)", fontsize=11)
    ax.set_xlabel("System state  x")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.axvline(0, color="grey", lw=0.8, ls=":")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {filepath}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── baseline parameters ────────────────────────────────────────────────
    BASE = QueueingSystemParams(
        arrival_rate=0.6,
        num_servers=2,
        service_rate=0.5,
        class2_arrival_rate=0.3,
        class1_reward=5.0,      # will be swept
        class2_reward=2.0,      # will be swept
        outsourcing_cost=1.5,
        congestion_sensitivity=0.1,
        max_queue_length=15,
        convergence_tolerance=1e-6,
        max_iterations=2000,
    )

    # ── sweep grids ────────────────────────────────────────────────────────
    R1_VALUES = [2.0, 3.5, 5.0, 6.5, 8.0]   # class-1 reward range
    R2_VALUES = [0.5, 1.5, 2.5, 3.5, 4.5]   # class-2 reward range

    state_space = build_state_space(BASE)

    # ── a priori sweep ─────────────────────────────────────────────────────
    print("Running a priori sweep (multiprocessing) …")
    results_ap = run_sweep_parallel(_run_apriori, BASE, R1_VALUES, R2_VALUES)
    _plot_value_functions(
        results_ap,
        state_space,
        R1_VALUES,
        R2_VALUES,
        title="Value Functions — A Priori Bellman",
        filepath="curves_apriori.png",
    )

    # ── a posteriori sweep ─────────────────────────────────────────────────
    print("Running a posteriori sweep (multiprocessing) …")
    results_post = run_sweep_parallel(_run_aposteriori, BASE, R1_VALUES, R2_VALUES)
    _plot_value_functions(
        results_post,
        state_space,
        R1_VALUES,
        R2_VALUES,
        title="Value Functions — A Posteriori Bellman",
        filepath="curves_aposteriori.png",
    )

    print("\nDone. Two comparison graphs saved.")


if __name__ == "__main__":
    main()
