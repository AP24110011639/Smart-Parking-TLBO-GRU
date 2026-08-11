import time
import numpy as np
import matplotlib.pyplot as plt

from parking_data import (
    generate_occupancy_series,
    make_windows,
    generate_parking_network,
    generate_vehicle_requests,
)
from feature_selection_lasso import lasso_select_features
from GRU_predictor import GRURegressor
from tlbo_optimizer import (
    MultiObjectiveAllocationProblem,
    TLBO,
    reallocate_vehicle,
)
from sboa_optimizer import ESBOA

FEATURE_NAMES = [
    "occupancy",
    "hour_sin",
    "hour_cos",
    "traffic",
    "temp_proxy",
]


def run_prediction_stage():
    print("\n=== STAGE 1: LASSO filtering + GRU occupancy prediction ===")

    X, y = generate_occupancy_series(n_zones=4, n_samples=400)

    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1)

    selected_idx, coefs = lasso_select_features(
        X_flat,
        y_flat,
        FEATURE_NAMES,
    )

    print(
        f"[LASSO] coefficients: "
        f"{dict(zip(FEATURE_NAMES, np.round(coefs, 4)))}"
    )

    X_sel = X[:, :, selected_idx]

    window = 6
    Xw, yw = make_windows(X_sel, y, window=window)

    n = len(Xw)
    idx = np.random.default_rng(0).permutation(n)
    split = int(0.8 * n)

    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = Xw[train_idx]
    y_train = yw[train_idx]

    X_test = Xw[test_idx]
    y_test = yw[test_idx]

    model = GRURegressor(
        n_features=X_sel.shape[-1],
        hidden_size=8,
        seed=0,
    )

    losses = model.fit(
        X_train,
        y_train,
        epochs=200,
        lr=0.05,
        verbose_every=50,
    )

    y_pred_test = model.predict(X_test)

    rmse = float(np.sqrt(np.mean((y_pred_test - y_test) ** 2)))
    mae = float(np.mean(np.abs(y_pred_test - y_test)))

    print(f"[GRU] Test RMSE={rmse:.4f}  MAE={mae:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(losses)
    axes[0].set_title("GRU training loss (MSE)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")

    order = np.argsort(y_test)

    axes[1].plot(y_test[order], label="actual", lw=1)
    axes[1].plot(
        y_pred_test[order],
        label="predicted",
        lw=1,
        alpha=0.8,
    )

    axes[1].set_title(f"GRU prediction fit (RMSE={rmse:.3f})")
    axes[1].legend()

    fig.tight_layout()
    plt.show()

    return model, X_sel, selected_idx


def predict_slot_waiting_time(model, X_sel, n_slots, seed=3):
    rng = np.random.default_rng(seed)

    window = 6
    n_zones = X_sel.shape[0]

    preds = []

    for s in range(n_slots):
        z = s % n_zones
        start = rng.integers(0, X_sel.shape[1] - window)

        sample = X_sel[z, start:start + window, :][None, :, :]
        preds.append(model.predict(sample)[0])

    predicted_occupancy = np.clip(np.array(preds), 0, 1)
    waiting_time = predicted_occupancy * 15.0

    return waiting_time 

def calculate_metrics(problem, assignment):
    """Calculate metrics for a given parking assignment."""

    waiting = problem.wait[assignment]
    distance = problem.total_dist[assignment]
    carbon = problem.carbon[assignment]
    satisfaction = problem.driver_pref[assignment]

    return {
        "waiting": np.mean(waiting),
        "distance": np.mean(distance),
        "carbon": np.mean(carbon),
        "satisfaction": np.mean(satisfaction) * 100
    }


def run_allocation_stage(waiting_time):
    print("\n=== STAGE 2: Multi-objective Smart Parking Optimization (TLBO vs ESBOA) ===")

    network = generate_parking_network(
        n_blocks=4,
        stalls_per_block=8,
    )

    n_slots = network["n_slots"]
    n_vehicles = 10

    urgency = generate_vehicle_requests(
        n_vehicles,
        n_slots,
    )

    problem = MultiObjectiveAllocationProblem(
        network,
        waiting_time,
        urgency,
        weights=(0.25, 0.20, 0.20, 0.15, 0.10, 0.10),
    )

    availability = network["availability"].copy()

    tlbo = TLBO(
        problem,
        availability,
        n_vehicles,
        pop_size=25,
        max_iter=60,
        seed=2,
    )

    start = time.perf_counter()

    tlbo_sol, tlbo_fit, tlbo_assign, tlbo_hist = tlbo.optimize()

    tlbo_runtime = time.perf_counter() - start

    sboa = ESBOA(
        problem,
        availability,
        n_vehicles,
        pop_size=25,
        max_iter=60,
        seed=2,
    )

    start = time.perf_counter()

    sboa_sol, sboa_fit, sboa_assign, sboa_hist = sboa.optimize()
    tlbo_metrics = calculate_metrics(problem, tlbo_assign)
    sboa_metrics = calculate_metrics(problem, sboa_assign)

    sboa_runtime = time.perf_counter() - start

    # ---------- Objective Statistics ----------
    # Average waiting time (minutes)
    avg_wait = np.mean(waiting_time)

    # Average travel distance
    avg_distance = np.mean(problem.total_dist)

    # Average carbon emission
    avg_carbon = np.mean(problem.carbon)

    # Average driver satisfaction
    # Higher value means better satisfaction
    avg_satisfaction = np.mean(1 - problem.norm_pref)

    print("\n========= Objective Statistics =========")
    print(f"Average Waiting Time       : {avg_wait:.2f} min")
    print(f"Average Travel Distance    : {avg_distance:.2f} m")
    print(f"Average Carbon Emission    : {avg_carbon:.3f} kg CO₂")
    print(f"Average Driver Satisfaction: {avg_satisfaction:.3f}")
    print("========================================")

    print(f"[TLBO] best fitness = {tlbo_fit:.4f}")
    print(f"[SBOA] best fitness = {sboa_fit:.4f}")
    if tlbo_fit < sboa_fit:
        print("\nWinner : Proposed Multi-Objective TLBO")
        improvement = ((sboa_fit - tlbo_fit) / sboa_fit) * 100
        print(f"Improvement over ESBOA : {improvement:.2f}%")
    else:
        print("\nWinner : ESBOA")
        improvement = ((tlbo_fit - sboa_fit) / tlbo_fit) * 100
        print(f"Improvement over TLBO : {improvement:.2f}%")
        print("\n--------- Objective Statistics ---------")
        assigned = tlbo_assign
        print(f"Optimized waiting time : {problem.wait[assigned].mean():.2f}")
        print(f"Optimized travel distance : {problem.total_dist[assigned].mean():.2f}")
        print(f"Optimized carbon emission : {problem.carbon[assigned].mean():.3f}")
        print(f"Average driver satisfaction : {problem.driver_pref[assigned].mean():.3f}")
    print(f"[TLBO] assignment (vehicle -> slot): {tlbo_assign.tolist()}")
    print(f"[ESBOA] assignment (vehicle -> slot): {sboa_assign.tolist()}")
    print("\n--------- Runtime Comparison ---------")
    print(f"TLBO Runtime  : {tlbo_runtime:.6f} sec")
    print(f"ESBOA Runtime : {sboa_runtime:.6f} sec")

    if tlbo_runtime < sboa_runtime:
        improvement = ((sboa_runtime - tlbo_runtime) / sboa_runtime) * 100
        print(f"TLBO is {improvement:.2f}% faster than ESBOA")
    elif sboa_runtime < tlbo_runtime:
        improvement = ((tlbo_runtime - sboa_runtime) / tlbo_runtime) * 100
        print(f"ESBOA is {improvement:.2f}% faster than TLBO")
    else:
        print("Both algorithms have identical runtime.")
    winner = "TLBO" if tlbo_fit < sboa_fit else "ESBOA"

    print("\n" + "="*70)
    print("                    PERFORMANCE COMPARISON")
    print("="*70)

    print(f"{'Metric':<22}{'TLBO':>14}{'ESBOA':>14}")
    print("-"*70)

    print(f"{'Fitness':<22}{tlbo_fit:>14.4f}{sboa_fit:>14.4f}")

    print(f"{'Runtime (sec)':<22}"
          f"{tlbo_runtime:>14.4f}"
          f"{sboa_runtime:>14.4f}")

    print(f"{'Waiting Time (min)':<22}"
          f"{tlbo_metrics['waiting']:>14.2f}"
          f"{sboa_metrics['waiting']:>14.2f}")

    print(f"{'Travel Distance (m)':<22}"
         f"{tlbo_metrics['distance']:>14.2f}"
         f"{sboa_metrics['distance']:>14.2f}")

    print(f"{'Carbon (kg CO₂)':<22}"
          f"{tlbo_metrics['carbon']:>14.3f}"
          f"{sboa_metrics['carbon']:>14.3f}")

    print(f"{'Driver Satisfaction':<22}"
          f"{tlbo_metrics['satisfaction']:>13.2f}%"
          f"{sboa_metrics['satisfaction']:>13.2f}%")

    print("="*70)

    winner = "TLBO" if tlbo_fit < sboa_fit else "ESBOA"

    print(f"Winner : Proposed Multi-Objective {winner}")

    improvement = abs((sboa_fit - tlbo_fit) / sboa_fit) * 100

    print(f"Fitness Improvement : {improvement:.2f}%")

    print("="*70)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(tlbo_hist, label="TLBO (proposed)")
    ax.plot(sboa_hist, label="Enhanced SBOA")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best fitness (lower=better)")
    ax.set_title("Convergence Comparison of TLBO and ESBOA")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        "Convergence_Comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    _plot_layout(
        network,
        tlbo_assign,
        availability,
        "03_tlbo_allocation_map.png",
        "TLBO allocation (before reallocation)",
    )

    return (
        problem,
        network,
        availability,
        tlbo_assign,
        urgency,
    )


def run_statistical_benchmark(
    model,
    X_sel,
    n_trials=20,
    n_blocks=4,
    stalls_per_block=8,
    n_vehicles=10,
):
    """
    STAGE 2B: Statistical robustness benchmark.

    Runs TLBO and ESBOA over n_trials independent trials, each with a
    freshly generated parking network, vehicle requests, and predicted
    waiting times (different random seeds per trial). Reports mean +/-
    standard deviation for every metric, plus a paired win count and a
    Wilcoxon signed-rank test on fitness. This replaces a single lucky
    (or unlucky) run with a statistically defensible comparison.
    """

    print(f"\n=== STAGE 2B: Statistical benchmark over {n_trials} trials ===")

    records = {
        "tlbo_fit": [], "sboa_fit": [],
        "tlbo_time": [], "sboa_time": [],
        "tlbo_wait": [], "sboa_wait": [],
        "tlbo_dist": [], "sboa_dist": [],
        "tlbo_carbon": [], "sboa_carbon": [],
        "tlbo_sat": [], "sboa_sat": [],
    }

    for trial in range(n_trials):
        network = generate_parking_network(
            n_blocks=n_blocks,
            stalls_per_block=stalls_per_block,
            seed=100 + trial,
        )
        n_slots = network["n_slots"]

        urgency = generate_vehicle_requests(
            n_vehicles, n_slots, seed=200 + trial
        )

        waiting_time = predict_slot_waiting_time(
            model, X_sel, n_slots, seed=300 + trial
        )

        problem = MultiObjectiveAllocationProblem(
            network, waiting_time, urgency,
            weights=(0.25, 0.20, 0.20, 0.15, 0.10, 0.10),
        )
        availability = network["availability"].copy()

        t0 = time.perf_counter()
        tlbo = TLBO(problem, availability, n_vehicles,
                    pop_size=25, max_iter=60, seed=400 + trial)
        _, tlbo_fit, tlbo_assign, _ = tlbo.optimize()
        tlbo_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        sboa = ESBOA(problem, availability, n_vehicles,
                     pop_size=25, max_iter=60, seed=400 + trial)
        _, sboa_fit, sboa_assign, _ = sboa.optimize()
        sboa_time = time.perf_counter() - t0

        tm = calculate_metrics(problem, tlbo_assign)
        sm = calculate_metrics(problem, sboa_assign)

        records["tlbo_fit"].append(tlbo_fit)
        records["sboa_fit"].append(sboa_fit)
        records["tlbo_time"].append(tlbo_time)
        records["sboa_time"].append(sboa_time)
        records["tlbo_wait"].append(tm["waiting"])
        records["sboa_wait"].append(sm["waiting"])
        records["tlbo_dist"].append(tm["distance"])
        records["sboa_dist"].append(sm["distance"])
        records["tlbo_carbon"].append(tm["carbon"])
        records["sboa_carbon"].append(sm["carbon"])
        records["tlbo_sat"].append(tm["satisfaction"])
        records["sboa_sat"].append(sm["satisfaction"])

    def fmt(key_prefix, decimals=4):
        t = np.array(records[f"tlbo_{key_prefix}"])
        s = np.array(records[f"sboa_{key_prefix}"])
        return t, s

    fit_t, fit_s = fmt("fit")
    time_t, time_s = fmt("time")
    wait_t, wait_s = fmt("wait")
    dist_t, dist_s = fmt("dist")
    carbon_t, carbon_s = fmt("carbon")
    sat_t, sat_s = fmt("sat")

    wins = int(np.sum(fit_t < fit_s))
    ties = int(np.sum(fit_t == fit_s))

    print("\n" + "=" * 78)
    print(f"   TABLE VI (revised) — TLBO vs ESBOA, mean +/- std over {n_trials} trials")
    print("=" * 78)
    print(f"{'Metric':<24}{'TLBO':>24}{'ESBOA':>24}")
    print("-" * 78)
    print(f"{'Fitness':<24}"
          f"{f'{fit_t.mean():.4f} +/- {fit_t.std():.4f}':>24}"
          f"{f'{fit_s.mean():.4f} +/- {fit_s.std():.4f}':>24}")
    print(f"{'Runtime (s)':<24}"
          f"{f'{time_t.mean():.4f} +/- {time_t.std():.4f}':>24}"
          f"{f'{time_s.mean():.4f} +/- {time_s.std():.4f}':>24}")
    print(f"{'Waiting Time (min)':<24}"
          f"{f'{wait_t.mean():.2f} +/- {wait_t.std():.2f}':>24}"
          f"{f'{wait_s.mean():.2f} +/- {wait_s.std():.2f}':>24}")
    print(f"{'Travel Distance (m)':<24}"
          f"{f'{dist_t.mean():.2f} +/- {dist_t.std():.2f}':>24}"
          f"{f'{dist_s.mean():.2f} +/- {dist_s.std():.2f}':>24}")
    print(f"{'Carbon Emission (kg)':<24}"
          f"{f'{carbon_t.mean():.3f} +/- {carbon_t.std():.3f}':>24}"
          f"{f'{carbon_s.mean():.3f} +/- {carbon_s.std():.3f}':>24}")
    print(f"{'Driver Satisfaction %':<24}"
          f"{f'{sat_t.mean():.2f} +/- {sat_t.std():.2f}':>24}"
          f"{f'{sat_s.mean():.2f} +/- {sat_s.std():.2f}':>24}")
    print("-" * 78)
    print(f"TLBO achieved lower (better) fitness in {wins}/{n_trials} trials "
          f"({ties} ties).")

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(fit_t, fit_s)
        print(f"Wilcoxon signed-rank test on fitness: stat={stat:.3f}, p={p:.4f}")
    except ImportError:
        print("(scipy not installed — skipping Wilcoxon significance test)")

    print("=" * 78)

    # Boxplot of fitness distributions for the paper
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [fit_t, fit_s],
        tick_labels=["TLBO (proposed)", "ESBOA"],
    )
    ax.set_ylabel("Fitness (lower = better)")
    ax.set_title(f"Fitness Distribution over {n_trials} Independent Trials")
    fig.tight_layout()
    fig.savefig("05_fitness_boxplot.png", dpi=300, bbox_inches="tight")
    plt.show()

    return records


def run_dynamic_reallocation(
    problem,
    network,
    availability,
    assignment,
    urgency,
):
    print("\n=== STAGE 3: Dynamic reallocation ===")

    rng = np.random.default_rng(5)

    vehicle_to_disrupt = int(
        rng.integers(0, len(assignment))
    )

    disrupted_slot = int(
        assignment[vehicle_to_disrupt]
    )

    print(
        f"[EVENT] slot {disrupted_slot} "
        f"(assigned to vehicle {vehicle_to_disrupt}) "
        f"just became unavailable before arrival."
    )

    plot_availability = availability.copy()
    plot_availability[disrupted_slot] = False

    search_availability = plot_availability.copy()

    for v, s in enumerate(assignment):
        if v != vehicle_to_disrupt:
            search_availability[s] = False

    new_slot, new_cost = reallocate_vehicle(
        problem,
        search_availability,
        vehicle_to_disrupt,
        urgency_value=urgency[vehicle_to_disrupt],
        seed=9,
    )

    new_assignment = assignment.copy()
    new_assignment[vehicle_to_disrupt] = new_slot

    print(
        f"[TLBO reallocation] vehicle {vehicle_to_disrupt} "
        f"reassigned to slot {new_slot} "
        f"(cost={new_cost:.4f})"
    )

    _plot_layout(
        network,
        new_assignment,
        plot_availability,
        "04_after_reallocation_map.png",
        f"After dynamic reallocation "
        f"(vehicle {vehicle_to_disrupt} -> slot {new_slot})",
        highlight=vehicle_to_disrupt,
    )

    return new_assignment


def _plot_layout(
    network,
    assignment,
    availability,
    filename,
    title,
    highlight=None,
):
    pos = network["positions"]

    fig, ax = plt.subplots(figsize=(7, 4))

    free = availability

    ax.scatter(
        pos[free, 0],
        pos[free, 1],
        c="tab:green",
        label="available",
        s=60,
    )

    ax.scatter(
        pos[~free, 0],
        pos[~free, 1],
        c="tab:red",
        label="occupied",
        s=60,
    )

    for v, s in enumerate(assignment):
        color = (
            "gold"
            if highlight is not None and v == highlight
            else "tab:blue"
        )

        ax.plot(
            [
                network["entry"][0],
                pos[s, 0],
                network["exit"][0],
            ],
            [
                network["entry"][1],
                pos[s, 1],
                network["exit"][1],
            ],
            alpha=0.35,
            lw=1,
            color=color,
        )

        ax.annotate(
            str(v),
            (pos[s, 0], pos[s, 1]),
            fontsize=7,
            ha="center",
            va="center",
        )

    ax.scatter(
        *network["entry"],
        marker="^",
        c="black",
        s=100,
        label="entry",
    )

    ax.scatter(
        *network["exit"],
        marker="v",
        c="black",
        s=100,
        label="exit",
    )

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    model, X_sel, selected_idx = run_prediction_stage()

    network_preview = generate_parking_network(
        n_blocks=4,
        stalls_per_block=8,
    )

    waiting_time = predict_slot_waiting_time(
        model,
        X_sel,
        network_preview["n_slots"],
    )

    (
        problem,
        network,
        availability,
        assignment,
        urgency,
    ) = run_allocation_stage(waiting_time)

    run_statistical_benchmark(
        model,
        X_sel,
        n_trials=20,
        n_blocks=4,
        stalls_per_block=8,
        n_vehicles=10,
    )

    run_dynamic_reallocation(
        problem,
        network,
        availability,
        assignment,
        urgency,
    )

   