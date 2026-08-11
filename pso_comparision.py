import numpy as np


class PSO:
    def __init__(
        self,
        problem,
        availability,
        n_vehicles,
        pop_size=25,
        max_iter=60,
        w=0.6,
        c1=1.5,
        c2=1.5,
        seed=1,
    ):
        self.problem = problem
        self.availability = availability
        self.n_vehicles = n_vehicles
        self.pop_size = pop_size
        self.max_iter = max_iter

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.rng = np.random.default_rng(seed)

        self.lb = 0.0
        self.ub = problem.n_slots - 1e-6

    def optimize(self):
        D = self.n_vehicles

        X = self.rng.uniform(
            self.lb,
            self.ub,
            (self.pop_size, D),
        )

        V = np.zeros_like(X)

        pbest = X.copy()
        pbest_fit = np.array(
            [
                self.problem.fitness(x, self.availability)[0]
                for x in X
            ]
        )

        gbest_idx = np.argmin(pbest_fit)
        gbest = pbest[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        history = []

        for _ in range(self.max_iter):
            r1 = self.rng.uniform(0, 1, (self.pop_size, D))
            r2 = self.rng.uniform(0, 1, (self.pop_size, D))

            V = (
                self.w * V
                + self.c1 * r1 * (pbest - X)
                + self.c2 * r2 * (gbest - X)
            )

            X = np.clip(X + V, self.lb, self.ub)

            fit = np.array(
                [
                    self.problem.fitness(x, self.availability)[0]
                    for x in X
                ]
            )

            improved = fit < pbest_fit

            pbest[improved] = X[improved]
            pbest_fit[improved] = fit[improved]

            gi = np.argmin(pbest_fit)

            if pbest_fit[gi] < gbest_fit:
                gbest_fit = pbest_fit[gi]
                gbest = pbest[gi].copy()

            history.append(gbest_fit)

        _, best_assignment = self.problem.fitness(
            gbest,
            self.availability,
        )

        return gbest, gbest_fit, best_assignment, history