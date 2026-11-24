import numpy as np

from tapnet.tapvid.evaluation_datasets import compute_tapvid_metrics
from tapnet.tapvid3d.evaluation.metrics import compute_tapvid3d_metrics


def evaluate_2d(sample, pred_tracks, pred_visible):

    _, h, w, _ = sample["video"].shape

    # Resize coordinates to 256 x 256
    scale = np.array([256 / w, 256 / h])

    gt = sample["gt_tracks"] * scale
    pred = pred_tracks * scale

    # Convert queries from (t, x, y) to (t, y, x)
    queries = sample["queries"].copy()
    queries[:, 1:] *= scale
    queries = queries[:, [0, 2, 1]]

    # Compute official metrics
    metrics = compute_tapvid_metrics(
        queries[None],
        ~sample["gt_visible"].T[None],
        gt.transpose(1, 0, 2)[None],
        ~pred_visible.T[None],
        pred.transpose(1, 0, 2)[None],
        query_mode="first"
    )

    return {
        "AJ": float(metrics["average_jaccard"].mean()),
        "APD": float(metrics["average_pts_within_thresh"].mean()),
        "OA": float(metrics["occlusion_accuracy"].mean())
    }
def evaluate_3d(sample, pred_tracks, pred_visible):

    _, h, w, _ = sample["video"].shape

    # Resize camera intrinsics
    intrinsics = sample["intrinsics"] * (256 / min(h, w))

    # Change query coordinate order
    queries = sample["queries"][:, [0, 2, 1]]

    # Compute official metrics
    metrics = compute_tapvid3d_metrics(
        gt_occluded=~sample["gt_visible"],
        gt_tracks=sample["gt_tracks"],
        pred_occluded=~pred_visible,
        pred_tracks=pred_tracks,
        intrinsics_params=intrinsics,
        scaling="median",
        query_points=queries,
        order="t n"
    )

    return {
        "3D-AJ": float(np.mean(metrics["average_jaccard"])),
        "APD_3D": float(np.mean(metrics["average_pts_within_thresh"])),
        "OA": float(np.mean(metrics["occlusion_accuracy"]))
    }