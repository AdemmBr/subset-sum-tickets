import numpy as np

grid = np.linspace(-1, 1, 201)


# poisson model, see notebook
def coverage_given_weights(n, eps, weights):
    x = np.atleast_2d(weights)[:, :n]
    mu = x.sum(axis=1, keepdims=True) / 2
    sigma = np.sqrt((x**2).sum(axis=1, keepdims=True)) / 2
    pdf = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    rate = 2.0**n * pdf
    gaps = rate * np.exp(-2 * eps * rate)
    return np.exp(-np.trapezoid(gaps, grid, axis=1))
