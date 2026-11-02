import numpy as np


class CoverageRuns:
    def __init__(self, weights, radii):
        # shape (trials, n_max)
        self.weights = weights
        self.radii = radii
        self.trials, self.n_max = radii.shape

    def first_hit(self, eps):
        hit = self.radii <= eps
        first = hit.argmax(axis=1) + 1.0
        first[~hit.any(axis=1)] = np.inf
        return first

    def threshold(self, eps, q=0.5):
        return float(np.quantile(self.first_hit(eps), q, method="higher"))

    def standardised_offset(self, n):
        x = self.weights[:, :n]
        return np.abs(x.sum(axis=1)) / np.sqrt((x**2).sum(axis=1))
