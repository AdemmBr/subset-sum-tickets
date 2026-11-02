from itertools import combinations
import numpy as np
import pytest
from utils import theory
from utils.coverage_runs import CoverageRuns
from utils.pruning import TicketLinear, WeightPool
from utils.simulation import run_coverage
from utils.subset_sums import covering_radius, lattice_radii


def brute_force_sums(weights):
    out = []
    for k in range(len(weights) + 1):
        out += [sum(c) for c in combinations(weights, k)]
    return np.sort(out)


def brute_force_radius(sums):
    grid = np.linspace(-1, 1, 20001)
    idx = np.searchsorted(sums, grid).clip(1, sums.size - 1)
    return np.minimum(np.abs(grid - sums[idx - 1]), np.abs(grid - sums[idx])).max()


def test_hand_computed_radius():
    assert covering_radius(np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert covering_radius(np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])) == pytest.approx(0.25)


def test_radius_matches_grid_search():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        sums = brute_force_sums(list(rng.uniform(-1, 1, rng.integers(1, 13))))
        assert covering_radius(sums) == pytest.approx(brute_force_radius(sums), abs=2e-4)


def test_lattice_matches_brute_force():
    scale = 2**10
    for seed in range(20):
        weights = np.random.default_rng(seed).integers(-scale, scale + 1, 10)
        radii = lattice_radii(weights, scale, floor=0.0)
        for n in range(1, 11):
            exact = covering_radius(brute_force_sums(list(weights[:n] / scale)))
            assert radii[n - 1] == (exact if exact <= 0.25 else np.inf)


def test_radius_never_increases():
    scale = 2**12
    weights = np.random.default_rng(1).integers(-scale, scale + 1, 16)
    radii = lattice_radii(weights, scale, floor=0.0)
    assert np.all(np.diff(radii[np.isfinite(radii)]) <= 0)


def test_first_hit_and_threshold():
    radii = np.array([[1.0, 0.5, 0.1], [1.0, 1.0, 1.0]])
    runs = CoverageRuns(np.ones_like(radii), radii)
    assert runs.first_hit(0.5).tolist() == [2.0, np.inf]
    assert runs.threshold(0.5, q=1.0) == np.inf


def test_runs_are_reproducible_across_workers():
    a = run_coverage(n_max=8, trials=6, seed=3, workers=1)
    b = run_coverage(n_max=8, trials=6, seed=3, workers=2)
    assert np.array_equal(a.radii, b.radii)


def test_closest_subset_is_optimal():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        pool = WeightPool(int(rng.integers(1, 11)), rng)
        target = rng.uniform(-1, 1)
        mask, value = pool.closest_subset(target)
        best = min(
            abs(sum(c) - target)
            for k in range(pool.values.size + 1)
            for c in combinations(pool.values, k)
        )
        assert pool.values[mask].sum() == pytest.approx(value)
        assert abs(value - target) == pytest.approx(best)


def test_large_pool_copies_the_predictions():
    rng = np.random.default_rng(0)
    coef, intercept = rng.normal(size=(3, 5)), rng.normal(size=3)
    X = rng.normal(size=(200, 5))
    ticket = TicketLinear(coef, intercept, pool_size=12, seed=0)
    expected = (X @ coef.T + intercept).argmax(axis=1)
    assert (ticket.predict(X) == expected).mean() > 0.95
    assert 0 < ticket.sparsity < 1


def weights(n=24):
    return np.random.default_rng(0).uniform(-1, 1, (200, n))


def test_prediction_grows_with_n_and_shrinks_with_eps():
    x = weights()
    assert theory.coverage_given_weights(14, 2.0**-8, x).mean() < theory.coverage_given_weights(20, 2.0**-8, x).mean()
    assert theory.coverage_given_weights(18, 2.0**-12, x).mean() < theory.coverage_given_weights(18, 2.0**-8, x).mean()


def test_a_sum_far_from_the_interval_is_predicted_worse():
    centred = np.tile([0.5, -0.5], 10)[None, :]
    leaning = np.full((1, 20), 0.5)
    assert theory.coverage_given_weights(20, 2.0**-10, leaning).mean() < theory.coverage_given_weights(20, 2.0**-10, centred).mean()
