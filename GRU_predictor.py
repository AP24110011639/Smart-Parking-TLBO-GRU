import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class GRURegressor:
    def __init__(self, n_features, hidden_size=8, seed=0):
        rng = np.random.default_rng(seed)
        H, F = hidden_size, n_features
        scale = 1.0 / np.sqrt(F + H)

        self.H = H

        self.Wz = rng.normal(0, scale, (F, H))
        self.Uz = rng.normal(0, scale, (H, H))
        self.bz = np.zeros(H)

        self.Wr = rng.normal(0, scale, (F, H))
        self.Ur = rng.normal(0, scale, (H, H))
        self.br = np.zeros(H)

        self.Wh = rng.normal(0, scale, (F, H))
        self.Uh = rng.normal(0, scale, (H, H))
        self.bh = np.zeros(H)

        self.Wy = rng.normal(0, scale, (H, 1))
        self.by = np.zeros(1)

    def _forward(self, X):
        N, T, F = X.shape
        H = self.H

        h = np.zeros((N, H))
        cache = {
            "h": [h.copy()],
            "z": [],
            "r": [],
            "hcand": [],
            "x": [],
        }

        for t in range(T):
            xt = X[:, t, :]

            z = sigmoid(xt @ self.Wz + h @ self.Uz + self.bz)
            r = sigmoid(xt @ self.Wr + h @ self.Ur + self.br)
            hcand = np.tanh(xt @ self.Wh + (r * h) @ self.Uh + self.bh)

            h = (1 - z) * h + z * hcand

            cache["z"].append(z)
            cache["r"].append(r)
            cache["hcand"].append(hcand)
            cache["x"].append(xt)
            cache["h"].append(h.copy())

        y_pred = h @ self.Wy + self.by
        return y_pred, cache

    def predict(self, X):
        y_pred, _ = self._forward(X)
        return y_pred.ravel()

    def fit(self, X, y, epochs=200, lr=0.05, verbose_every=50):
        y = y.reshape(-1, 1)
        N, T, F = X.shape
        H = self.H

        losses = []

        for ep in range(epochs):
            y_pred, cache = self._forward(X)

            err = y_pred - y
            loss = float(np.mean(err**2))
            losses.append(loss)

            h_T = cache["h"][-1]
            dWy = h_T.T @ err / N
            dby = np.mean(err, axis=0)

            dh_next = (err @ self.Wy.T) / N

            dWz = np.zeros_like(self.Wz)
            dUz = np.zeros_like(self.Uz)
            dbz = np.zeros_like(self.bz)

            dWr = np.zeros_like(self.Wr)
            dUr = np.zeros_like(self.Ur)
            dbr = np.zeros_like(self.br)

            dWh = np.zeros_like(self.Wh)
            dUh = np.zeros_like(self.Uh)
            dbh = np.zeros_like(self.bh)

            for t in reversed(range(T)):
                h_prev = cache["h"][t]
                z = cache["z"][t]
                r = cache["r"][t]
                hcand = cache["hcand"][t]
                xt = cache["x"][t]

                dh = dh_next

                dz = dh * (hcand - h_prev)
                dhcand = dh * z
                dh_prev = dh * (1 - z)

                dz_pre = dz * z * (1 - z)
                dhcand_pre = dhcand * (1 - hcand**2)

                dWz += xt.T @ dz_pre / N
                dUz += h_prev.T @ dz_pre / N
                dbz += np.mean(dz_pre, axis=0)

                dWh += xt.T @ dhcand_pre / N
                d_rh = dhcand_pre @ self.Uh.T
                dUh += (r * h_prev).T @ dhcand_pre / N
                dbh += np.mean(dhcand_pre, axis=0)

                dr = d_rh * h_prev
                dh_prev += d_rh * r

                dr_pre = dr * r * (1 - r)

                dWr += xt.T @ dr_pre / N
                dUr += h_prev.T @ dr_pre / N
                dbr += np.mean(dr_pre, axis=0)

                dh_prev += dr_pre @ self.Ur.T + dz_pre @ self.Uz.T
                dh_next = dh_prev

            for p, g in [
                (self.Wz, dWz),
                (self.Uz, dUz),
                (self.Wr, dWr),
                (self.Ur, dUr),
                (self.Wh, dWh),
                (self.Uh, dUh),
                (self.Wy, dWy),
            ]:
                p -= lr * np.clip(g, -5, 5)

            for p, g in [
                (self.bz, dbz),
                (self.br, dbr),
                (self.bh, dbh),
                (self.by, dby),
            ]:
                p -= lr * np.clip(g, -5, 5)

            if verbose_every and (ep % verbose_every == 0 or ep == epochs - 1):
                print(f"[GRU] epoch {ep:4d}  MSE={loss:.5f}")

        return losses