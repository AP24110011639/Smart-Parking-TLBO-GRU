import numpy as np


def _normalise(v):
    v = np.asarray(v, dtype=float)
    lo, hi = v.min(), v.max()

    if hi - lo < 1e-9:
        return np.zeros_like(v)

    return (v - lo) / (hi - lo)


class MultiObjectiveAllocationProblem:
    def __init__(
        self,
        network,
        predicted_waiting_time,
        urgency,
        weights=(0.25, 0.20, 0.20, 0.15, 0.10, 0.10),
    ):
        self.net = network
        self.n_slots = network["n_slots"]
        rng = np.random.default_rng(42)
        self.driver_pref = rng.uniform(0.7, 1.0, self.n_slots)
        self.norm_pref = _normalise(self.driver_pref)
        self.total_dist = (
            network["dist_entry"] + network["dist_exit"]
        )

        self.traffic = network["traffic_factor"]
        self.wait = predicted_waiting_time
        self.urgency = urgency
        self.w = weights

        self.norm_dist = _normalise(self.total_dist)
        self.norm_traffic = _normalise(self.traffic)
        self.norm_wait = _normalise(self.wait)
        self.carbon = self.total_dist * 0.18
        self.norm_carbon = _normalise(self.carbon)

        always_available = np.ones(self.n_slots, dtype=bool)
        self.sorted_slots_per_vehicle = []

        for v in range(len(urgency)):
            costs = np.array([
                self.slot_cost(v, s, always_available)
                for s in range(self.n_slots)
            ])
            self.sorted_slots_per_vehicle.append(
                np.argsort(costs)
            )

    def slot_cost(self, vehicle_idx, slot_idx, availability):
        w1, w2, w3, w4, w5, w6 = self.w

        avail_penalty = 0.0 if availability[slot_idx] else 1.0

        if vehicle_idx is not None:
            traffic_factor = self.norm_traffic[slot_idx]
            u = self.urgency[vehicle_idx] * (1 + traffic_factor)
        else:
            u = 1.0

        carbon = self.norm_carbon[slot_idx]
        satisfaction = 1 - self.norm_pref[slot_idx]

        return (
            w1 * self.norm_dist[slot_idx]
            + w2 * self.norm_traffic[slot_idx]
            + w3 * u * self.norm_wait[slot_idx]
            + w4 * avail_penalty
            + w5 * carbon
            + w6 * satisfaction
        )

    def decode(self, genes, availability):
        n_vehicles = len(genes)

        rank_pref = np.clip(
            np.round(genes).astype(int),
            0,
            self.n_slots - 1,
        )

        taken = set()
        assignment = np.full(n_vehicles, -1, dtype=int)

        order = np.argsort(
            self.urgency[:n_vehicles]
        )[::-1]

        for v in order:
            ranked_slots = self.sorted_slots_per_vehicle[v]
            p = rank_pref[v]

            candidates = ranked_slots[
                np.argsort(
                    np.abs(np.arange(self.n_slots) - p)
                )
            ]

            for c in candidates:
                if c not in taken and availability[c]:
                    assignment[v] = c
                    taken.add(c)
                    break

            if assignment[v] == -1:
                for c in candidates:
                    if c not in taken:
                        assignment[v] = c
                        taken.add(c)
                        break

        return assignment

    def fitness(self, genes, availability):
        assignment = self.decode(genes, availability)

        total = 0.0

        for v, s in enumerate(assignment):
            total += self.slot_cost(v, s, availability)

        return total, assignment


class TLBO:
    def __init__(
        self,
        problem,
        availability,
        n_vehicles,
        pop_size=25,
        max_iter=60,
        seed=0,
        stagnation_limit=8,
    ):
        self.problem = problem
        self.availability = availability
        self.n_vehicles = n_vehicles
        self.pop_size = pop_size
        self.max_iter = max_iter

        self.rng = np.random.default_rng(seed)
        self.stagnation_limit = stagnation_limit

        self.lb = 0.0
        self.ub = problem.n_slots - 1e-6

    def _clip(self, X):
        return np.clip(X, self.lb, self.ub)

    def _eval_pop(self, POP):
        fit = np.zeros(len(POP))

        for i, genes in enumerate(POP):
            fit[i], _ = self.problem.fitness(
                genes,
                self.availability,
            )

        return fit

    def optimize(self):
        D = self.n_vehicles

        POP = self.rng.uniform(
            self.lb,
            self.ub,
            size=(self.pop_size, D),
        )

        fit = self._eval_pop(POP)

        history = []

        best_idx = np.argmin(fit)
        best_sol = POP[best_idx].copy()
        best_fit = fit[best_idx]

        stagnant = 0

        for _ in range(self.max_iter):
            teacher_idx = np.argmin(fit)
            teacher = POP[teacher_idx]
            mean_pop = POP.mean(axis=0)

            for i in range(self.pop_size):
                TF = self.rng.integers(1, 3)
                r = self.rng.uniform(0, 1, D)

                new_sol = self._clip(
                    POP[i] + r * (teacher - TF * mean_pop)
                )

                new_fit, _ = self.problem.fitness(
                    new_sol,
                    self.availability,
                )

                if new_fit < fit[i]:
                    POP[i] = new_sol
                    fit[i] = new_fit

            for i in range(self.pop_size):
                j = i

                while j == i:
                    j = self.rng.integers(
                        0,
                        self.pop_size,
                    )

                r = self.rng.uniform(0, 1, D)

                if fit[i] < fit[j]:
                    new_sol = self._clip(
                        POP[i] + r * (POP[i] - POP[j])
                    )
                else:
                    new_sol = self._clip(
                        POP[i] + r * (POP[j] - POP[i])
                    )

                new_fit, _ = self.problem.fitness(
                    new_sol,
                    self.availability,
                )

                if new_fit < fit[i]:
                    POP[i] = new_sol
                    fit[i] = new_fit

            gen_best_idx = np.argmin(fit)

            if fit[gen_best_idx] < best_fit:
                best_fit = fit[gen_best_idx]
                best_sol = POP[gen_best_idx].copy()
                stagnant = 0
            else:
                stagnant += 1

            worst_idx = np.argmax(fit)

            if worst_idx != np.argmin(fit):
                POP[worst_idx] = best_sol.copy()
                fit[worst_idx] = best_fit

            if stagnant >= self.stagnation_limit:
                perturb_idx = np.argsort(fit)[-2]

                POP[perturb_idx] = self._clip(
                    POP[perturb_idx]
                    + self.rng.normal(0, 1.0 * (1 - stagnant / self.stagnation_limit), D)
                )

                fit[perturb_idx], _ = self.problem.fitness(
                    POP[perturb_idx],
                    self.availability,
                )

                stagnant = 0

            history.append(float(best_fit))

        _, best_assignment = self.problem.fitness(
            best_sol,
            self.availability,
        )

        return (
            best_sol,
            best_fit,
            best_assignment,
            history,
        )


def reallocate_vehicle(
    problem,
    availability,
    vehicle_idx,
    urgency_value,
    pop_size=15,
    max_iter=25,
    seed=None,
):
    mini_problem = MultiObjectiveAllocationProblem(
        problem.net,
        problem.wait,
        np.array([urgency_value]),
        problem.w,
    )

    tlbo = TLBO(
        mini_problem,
        availability,
        n_vehicles=1,
        pop_size=pop_size,
        max_iter=max_iter,
        seed=seed if seed is not None else np.random.randint(1_000_000),
    )

    best_sol, best_fit, best_assignment, _ = tlbo.optimize()

    return int(best_assignment[0]), best_fit