"""Grid search of a 2D baseline's hyper-parameters on the DAVIS tuning subset.

The tuning sequences (configs/subsets/davis_tuning.txt) are kept out of the
evaluation subsets, so the reported scores are not tuned on the test data.
Configurations are ranked by AJ, the main TAP-Vid metric.

Usage (from the repository root):
    python -m scripts.tune --tracker lk --data data/tapvid_davis/tapvid_davis/tapvid_davis.pkl
"""
import argparse
import importlib
import itertools
from pathlib import Path

import pandas as pd

from src.datasets.tapvid import load_tapvid
from src.evaluation.metrics import evaluate_2d

# tracker name -> (module, class, grid)
SEARCHES = {
    "lk": ("src.models.baselines.lk", "LKTracker", {
        "win_size": [15, 21, 31],
        "max_level": [2, 3, 4],
        "fb_threshold": [1.0, 2.0, 4.0],
    }),
    "dino_sim": ("src.models.baselines.dino_similarity", "DinoSimilarityTracker", {
        "vis_threshold": [0.7, 0.75, 0.8, 0.85, 0.9],
        "tau": [0.01, 0.02],
    }),
    "dino_proto": ("src.models.baselines.dino_prototype", "DinoPrototypeTracker", {
        "vis_threshold": [0.75, 0.8, 0.85],
        "update_threshold": [0.85, 0.9, 0.95],
        "momentum": [0.02, 0.05],
        "neighbourhood": [0.0, 0.25],
    }),
    "lk_dino": ("src.models.baselines.lk_dino", "LKDinoTracker", {
        "sim_threshold": [0.6, 0.7, 0.8, 0.85],
        "search_radius": [3, 5, 8],
    }),
    "dino_cv": ("src.models.baselines.dino_cv", "DinoConstantVelocityTracker", {
        "vis_threshold": [0.7, 0.75, 0.8, 0.85, 0.9],
        "search_radius": [2, 3, 4],
    }),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=sorted(SEARCHES), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--subset", default="configs/subsets/davis_tuning.txt")
    args = parser.parse_args()

    module, cls_name, grid = SEARCHES[args.tracker]
    cls = getattr(importlib.import_module(module), cls_name)
    samples = list(load_tapvid(args.data, args.subset))
    rows = []

    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        tracker = cls(**params)
        scores = [evaluate_2d(s, *tracker.track(s["video"], s["queries"])) for s in samples]
        mean = pd.DataFrame(scores).mean().to_dict()
        rows.append({**params, **mean})
        print(params, {k: round(v, 4) for k, v in mean.items()})

    df = pd.DataFrame(rows).sort_values("AJ", ascending=False)
    output = Path("results") / "2d" / "tuning"
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / f"{args.tracker}_grid.csv", index=False)
    print("\nBest by AJ:\n", df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
