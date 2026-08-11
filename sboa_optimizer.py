import math

import numpy as np


class ESBOA:
    """
    Enhanced Secretary Bird Optimization Algorithm (ESBOA)

    Enhancements:
    1. Logistic Chaotic Initialization
    2. Adaptive Exploration Coefficient
    3. Elite Preservation Strategy (Fu et al., 2024).

    Models the survival behaviour of secretary birds and is used here
    as the baseline metaheuristic compared against TLBO for the
    multi-objective parking-slot allocation problem.

    Exploration ("hunting a snake") — three time-based stages:
        1) searching for prey        (t <  T/3)
        2) consultation / exhausting (T/3 <= t < 2T/3)
        3) attacking prey            (t >= 2T/3)

    Exploitation ("escaping a predator") — two equally-likely tactics,
    applied every iteration after the hunting stage:
        1) camouflage in the surrounding environment
        2) fleeing by running / flying away
    """

    def __init__(
        self,
        problem,
        availability,
        n_vehicles,
        pop_size=25,
        max_iter=60,
        seed=1,
    ):
        self.problem = problem
        self.availability = availability
        self.n_vehicles = n_vehicles
        self.algorithm_name = "Enhanced Secretary Bird Optimization Algorithm (ESBOA)"
        self.pop_size = pop_size
        self.max_iter = max_iter

        self.rng = np.random.default_rng(seed)

        self.lb = 0.0
        self.ub = problem.n_slots - 1e-6

    def _clip(self, X):
        return np.clip(X, self.lb, self.ub)

    def _eval_pop(self, POP):
        return np.array(
            [
                self.problem.fitness(x, self.availability)[0]
                for x in POP
            ]
        )

    def _levy(self, D, beta=1.5):
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma = (num / den) ** (1 / beta)

        u = self.rng.normal(0, sigma, D)
        v = self.rng.normal(0, 1, D)

        return u / (np.abs(v) ** (1 / beta))
    
    def _chaotic_init(self, N, D):

        X = np.zeros((N, D))

        x = self.rng.random()

        for i in range(N):
            for j in range(D):

                x = 4 * x * (1 - x)

                X[i, j] = self.lb + x * (self.ub - self.lb)

        return X
    
    def optimize(self):
        D = self.n_vehicles
        N = self.pop_size
        T = self.max_iter

        X = self._chaotic_init(N, D)
        fit = self._eval_pop(X)

        best_idx = np.argmin(fit)
        best = X[best_idx].copy()
        best_fit = fit[best_idx]

        history = []

        for t in range(1, T + 1):
            alpha = 1 - (t / T)
            # ---------------- Exploration: hunting phase ----------------
            for i in range(N):
                if t < T / 3:
                    # Stage 1: searching for prey - long-range random moves
                    r1, r2 = self.rng.choice(N, 2, replace=False)
                    R = self.rng.uniform(0, 1, D)

                    Xnew = X[i] + alpha * (X[r1] - X[r2]) * R

                elif t < 2 * T / 3:
                    # Stage 2: consultation - drift toward the best bird
                    # relative to the flock's average position
                    mean_pos = X.mean(axis=0)
                    RB = self.rng.normal(0, 1, D)

                    Xnew = X[i] + alpha * RB * (best - mean_pos)

                else:
                    # Stage 3: attacking prey - Levy-guided strike toward
                    # the best-known location, shrinking over time
                    step = self._levy(D)
                    shrink = (1 - t / T) ** (2 * t / T)

                    Xnew = best + alpha * shrink * step * X[i]

                Xnew = self._clip(Xnew)
                f_new, _ = self.problem.fitness(Xnew, self.availability)

                if f_new < fit[i]:
                    X[i] = Xnew
                    fit[i] = f_new

            # --------------- Exploitation: escaping phase ---------------
            for i in range(N):
                Rd = self.rng.uniform(-1, 1, D)
    
                if self.rng.random() < 0.5:
                    Xnew = best + alpha * Rd * X[i]
                else:
                    k = self.rng.integers(1, 3)
                    Xnew = X[i] + alpha * Rd * (best - k * X[i])

                Xnew = self._clip(Xnew)
                f_new, _ = self.problem.fitness(Xnew, self.availability)
 
                if f_new < fit[i]:
                    X[i] = Xnew
                    fit[i] = f_new

            # ---------- Elite Preservation ----------
            worst_idx = np.argmax(fit)

            X[worst_idx] = best.copy()
            fit[worst_idx] = best_fit

            gen_best_idx = np.argmin(fit)

            if fit[gen_best_idx] < best_fit:
                best_fit = fit[gen_best_idx]
                best = X[gen_best_idx].copy()

            history.append(float(best_fit))

        _, best_assignment = self.problem.fitness(best, self.availability)

        return best, best_fit, best_assignment, history
