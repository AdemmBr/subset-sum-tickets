from multiprocessing import Pool
import numpy as np
from .coverage_runs import CoverageRuns
from .subset_sums import lattice_radii

SCALE = 2**20


def run_trial(args):
    seed, n_max = args
    rng = np.random.default_rng(seed)
    weights = rng.integers(-SCALE, SCALE + 1, n_max)
    return weights / SCALE, lattice_radii(weights, SCALE)


def run_coverage(n_max=40, trials=20000, seed=0, workers=3):
    seeds = np.random.SeedSequence(seed).spawn(trials)
    jobs = [(s, n_max) for s in seeds]
    if workers > 1:
        with Pool(workers) as pool:
            out = pool.map(run_trial, jobs, chunksize=8)
    else:
        out = [run_trial(job) for job in jobs]
    return CoverageRuns(np.stack([o[0] for o in out]), np.stack([o[1] for o in out]))
