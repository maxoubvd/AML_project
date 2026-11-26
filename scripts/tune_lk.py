"""Grid search of the LK baseline hyper-parameters on the DAVIS tuning subset.

The tuning sequences (configs/subsets/davis_tuning.txt) are kept out of the
evaluation subsets, so the reported scores are not tuned on the test data.

Usage (from the repository root):
    python -m scripts.tune_lk --data data/tapvid_davis/tapvid_davis/tapvid_davis.pkl
"""
import argparse
import itertools
from pathlib import Path

import pandas as pd

from src.datasets.tapvid import load_tapvid
from src.evaluation.metrics import evaluate_2d
from src.models.baselines.lk import LKTracker

GRID = {
    "win_size": [15, 21, 31],
    "max_level": [2, 3, 4],
    "fb_threshold": [1.0, 2.0, 4.0],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--subset", default="configs/subsets/davis_tuning.txt")
    args = parser.parse_args()

    samples = list(load_tapvid(args.data, args.subset))
    rows = []

    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        tracker = LKTracker(**params)
        scores = [evaluate_2d(s, *tracker.track(s["video"], s["queries"])) for s in samples]
        mean = pd.DataFrame(scores).mean().to_dict()
        rows.append({**params, **mean})
        print(params, {k: round(v, 4) for k, v in mean.items()})

    df = pd.DataFrame(rows).sort_values("AJ", ascending=False)
    output = Path("results") / "2d" / "tuning"
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "lk_grid.csv", index=False)
    print("\nBest by AJ:\n", df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
