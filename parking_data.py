import numpy as np


def generate_occupancy_series(n_zones=4, n_samples=400, seed=42):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_zones, n_samples, 5))
    y = np.zeros((n_zones, n_samples))

    t = np.arange(n_samples)
    hour = t % 24
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    for z in range(n_zones):
        base_level = rng.uniform(0.3, 0.6)
        amplitude = rng.uniform(0.25, 0.4)

        occ = (
            base_level
            + amplitude * (0.5 * (hour_sin + 1))
            + rng.normal(0, 0.03, size=n_samples)
        )
        occ = np.clip(occ, 0.02, 0.98)

        traffic = np.clip(
            0.4 + 0.3 * (0.5 * (hour_sin + 1))
            + rng.normal(0, 0.05, n_samples),
            0.05,
            1.0,
        )

        temp_proxy = (
            0.5
            + 0.1 * np.sin(2 * np.pi * t / n_samples)
            + rng.normal(0, 0.02, n_samples)
        )

        X[z, :, 0] = occ
        X[z, :, 1] = hour_sin
        X[z, :, 2] = hour_cos
        X[z, :, 3] = traffic
        X[z, :, 4] = temp_proxy

        y[z, :-1] = occ[1:]
        y[z, -1] = occ[-1]

    return X, y


def make_windows(X, y, window=6):
    n_zones, n_samples, n_feat = X.shape
    Xw, yw = [], []

    for z in range(n_zones):
        for i in range(n_samples - window):
            Xw.append(X[z, i:i + window, :])
            yw.append(y[z, i + window - 1])

    return np.array(Xw), np.array(yw)


def generate_parking_network(n_blocks=4, stalls_per_block=8, seed=7):
    rng = np.random.default_rng(seed)
    n_slots = n_blocks * stalls_per_block

    entry = np.array([0.0, 0.0])
    exit_ = np.array([n_blocks * 10.0 + 5, 0.0])

    slot_positions = []

    for b in range(n_blocks):
        block_x = 5 + b * 10
        for s in range(stalls_per_block):
            slot_y = 3 + (s % (stalls_per_block // 2)) * 3
            slot_x = block_x + (3 if s >= stalls_per_block // 2 else 0)
            slot_positions.append([slot_x, slot_y])

    slot_positions = np.array(slot_positions, dtype=float)

    dist_entry = np.linalg.norm(slot_positions - entry, axis=1)
    dist_exit = np.linalg.norm(slot_positions - exit_, axis=1)

    traffic_factor = np.clip(
        rng.normal(0.4, 0.15, n_slots),
        0.05,
        0.95,
    )

    availability = rng.random(n_slots) > 0.35

    return {
        "n_slots": n_slots,
        "positions": slot_positions,
        "entry": entry,
        "exit": exit_,
        "dist_entry": dist_entry,
        "dist_exit": dist_exit,
        "traffic_factor": traffic_factor,
        "availability": availability.copy(),
    }


def generate_vehicle_requests(n_vehicles, n_slots, seed=11):
    rng = np.random.default_rng(seed)
    urgency = rng.uniform(0.5, 1.5, n_vehicles)
    return urgency