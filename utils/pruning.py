import numpy as np


class WeightPool:
    def __init__(self, size, rng):
        self.values = rng.uniform(-1, 1, size)
        sums = np.zeros(1)
        for x in self.values:
            sums = np.concatenate((sums, sums + x))
        self.order = np.argsort(sums)
        self.sorted_sums = sums[self.order]

    def closest_subset(self, target):
        s = self.sorted_sums
        i = np.searchsorted(s, target).clip(1, s.size - 1)
        if abs(s[i - 1] - target) <= abs(s[i] - target):
            i -= 1
        mask = (int(self.order[i]) >> np.arange(self.values.size)) & 1
        return mask.astype(bool), float(s[i])


class TicketLinear:
    def __init__(self, coef, intercept, pool_size, seed=0):
        target = np.column_stack((coef, intercept))
        target = target / np.abs(target).max()
        rng = np.random.default_rng(seed)
        self.weights = np.empty_like(target, dtype=float)
        kept = 0
        for idx, w in np.ndenumerate(target):
            mask, value = WeightPool(pool_size, rng).closest_subset(w)
            self.weights[idx] = value
            kept += int(mask.sum())
        self.sparsity = 1 - kept / (target.size * pool_size)
        self.max_error = float(np.abs(self.weights - target).max())

    def predict(self, X):
        scores = X @ self.weights[:, :-1].T + self.weights[:, -1]
        return scores.argmax(axis=1)
