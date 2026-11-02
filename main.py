import argparse
import numpy as np
from utils.simulation import run_coverage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-max", type=int, default=40)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="data/coverage.npz")
    args = parser.parse_args()
    runs = run_coverage(args.n_max, args.trials, args.seed, args.workers)
    np.savez_compressed(args.out, weights=runs.weights, radii=runs.radii)
