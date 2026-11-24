import argparse
import pandas as pd
from pathlib import Path
from src.tracking.cotracker import CoTracker3
from src.datasets.tapvid import load_tapvid
from src.datasets.tapvid3d import load_tapvid3d
from src.tracking.base import StaticTracker2D, StaticTracker3D
from src.evaluation.metrics import evaluate_2d, evaluate_3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["2d", "3d"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--tracker", choices=["static", "gt", "cotracker3"], default="static")
    args = parser.parse_args()
    if args.task == "3d" and args.tracker == "cotracker3":
        parser.error("CoTracker3 only supports 2D tracking")
       # Select dataset and tracker
    if args.task == "2d":
        samples = load_tapvid(args.data, args.subset)

        if args.tracker == "cotracker3":
            tracker = CoTracker3()
        else:
            tracker = StaticTracker2D()

        evaluate = evaluate_2d

    else:
        samples = load_tapvid3d(args.data, args.subset)
        tracker = StaticTracker3D()
        evaluate = evaluate_3d

    results = []

    # Evaluate each sequence
    for sample in samples:

        if args.tracker == "gt":
            tracks = sample["gt_tracks"]
            visible = sample["gt_visible"]

        elif args.task == "2d":
            tracks, visible = tracker.track(
                sample["video"], sample["queries"]
            )

        else:
            tracks, visible = tracker.track(
                sample["video"], sample["queries"],
                sample["intrinsics"]
            )

        scores = evaluate(sample, tracks, visible)
        results.append({"name": sample["name"], **scores})
        print(sample["name"], scores)

    if not results:
        raise ValueError("No sequences found")

    # Calculate mean scores
    df = pd.DataFrame(results)
    mean = df.mean(numeric_only=True)
    df.loc[len(df)] = {"name": "mean", **mean.to_dict()}

    # Save results
    output = Path("results") / args.task
    output.mkdir(parents=True, exist_ok=True)

    df.to_csv(output / f"{args.tracker}.csv", index=False)
    print(mean)


if __name__ == "__main__":
    main()