import argparse
import pandas as pd
from pathlib import Path
from src.datasets.tapvid import load_tapvid
from src.datasets.tapvid3d import load_tapvid3d
from src.evaluation.metrics import evaluate_2d, evaluate_3d


def _static_2d():
    from src.tracking.base import StaticTracker2D
    return StaticTracker2D()


def _static_3d():
    from src.tracking.base import StaticTracker3D
    return StaticTracker3D()


def _cotracker3():
    from src.tracking.cotracker import CoTracker3
    return CoTracker3()


def _lk():
    from src.models.baselines.lk import LKTracker
    return LKTracker()


def _dino_sim():
    from src.models.baselines.dino_similarity import DinoSimilarityTracker
    return DinoSimilarityTracker()


def _dino_proto():
    from src.models.baselines.dino_prototype import DinoPrototypeTracker
    return DinoPrototypeTracker()


def _lk_dino():
    from src.models.baselines.lk_dino import LKDinoTracker
    return LKDinoTracker()


def _dino_cv():
    from src.models.baselines.dino_cv import DinoConstantVelocityTracker
    return DinoConstantVelocityTracker()


# Tracker registry: name -> factory. Imports are lazy so that a tracker's
# dependencies (torch.hub, DINOv3, VGGT, ...) load only when it is used.
# To add a tracker, add one line here.
TRACKERS = {
    "2d": {
        "static": _static_2d,
        "cotracker3": _cotracker3,
        "lk": _lk,
        "dino_sim": _dino_sim,
        "dino_proto": _dino_proto,
        "lk_dino": _lk_dino,
        "dino_cv": _dino_cv,
    },
    "3d": {
        "static": _static_3d,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["2d", "3d"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--subset", default=None)
    all_names = sorted({"gt"} | {n for task in TRACKERS.values() for n in task})
    parser.add_argument("--tracker", choices=all_names, default="static")
    args = parser.parse_args()

    if args.tracker != "gt" and args.tracker not in TRACKERS[args.task]:
        parser.error(f"Tracker '{args.tracker}' is not available for the {args.task} task")

    # Select dataset and tracker
    if args.task == "2d":
        samples = load_tapvid(args.data, args.subset)
        evaluate = evaluate_2d
    else:
        samples = load_tapvid3d(args.data, args.subset)
        evaluate = evaluate_3d

    tracker = None if args.tracker == "gt" else TRACKERS[args.task][args.tracker]()

    results = []

    # Evaluate each sequence
    for sample in samples:

        if tracker is None:
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

    # Save results: results/<task>/<tracker>[_<subset>].csv, so runs on
    # different subsets do not overwrite each other
    output = Path("results") / args.task
    output.mkdir(parents=True, exist_ok=True)

    stem = args.tracker if args.subset is None else f"{args.tracker}_{Path(args.subset).stem}"
    df.to_csv(output / f"{stem}.csv", index=False)
    print(mean)


if __name__ == "__main__":
    main()
