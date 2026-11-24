# Branch `feat/eval-common`: shared data loading and evaluation

**Owner:** Mohammadreza · **Base:** `main` · **Merge:** PR into `main`, reviewed by at least one teammate

## Goal

Write once the code that every later branch (2D baselines, VGGT, D4RT, 3D lifting, extension) needs, so that all rows of the Q6, Q10 and Q13 tables come from the **same loaders, conventions and metric code**. No tracker is implemented here.

## Scope

| In scope | Out of scope |
|---|---|
| TAP-Vid DAVIS and TAPVid-3D loaders | Baselines, VGGT, D4RT, CoTracker3 |
| Tracker interface (2D and 3D) | Depth / pose estimation |
| Wrappers around the **official** metrics | Choosing the Step 1 subset (only make it loadable) |
| One evaluation script that writes a CSV | Plots and qualitative figures |

## Deliverables

```
src/datasets/tapvid.py        # TAP-Vid DAVIS loader
src/datasets/tapvid3d.py      # TAPVid-3D loader (ADT, PStudio, DriveTrack)
src/tracking/base.py          # Tracker2D / Tracker3D interfaces + 2 trivial trackers
src/evaluation/metrics.py     # evaluate_2d, evaluate_3d
scripts/evaluate.py           # CLI: tracker x subset -> results/<task>/<tracker>.csv
configs/subsets/*.txt         # one sequence name per line
```

## Conventions (fix them once, document them in docstrings)

All arrays are NumPy, frame axis first.

| Name | Shape | Meaning |
|---|---|---|
| `video` | `(T, H, W, 3)` uint8 | RGB frames, original resolution |
| `queries` | `(N, 3)` float | `(t, x, y)`, pixels of the original resolution |
| `tracks_2d` | `(T, N, 2)` float | `(x, y)` pixels |
| `tracks_3d` | `(T, N, 3)` float | `(X, Y, Z)` in the **camera frame of frame t** (same as TAPVid-3D GT) |
| `visible` | `(T, N)` bool | True = visible (the opposite of tapnet's `occluded`) |
| `intrinsics` | `(4,)` | `fx, fy, cx, cy` |

Note: the DAVIS pickle stores `points` as `(N, T, 2)` normalized to [0, 1] and `occluded`, and tapnet's metric functions expect `(N, T)` order. The loaders and `metrics.py` do every conversion, so tracker code never transposes or renormalizes.

## Interfaces

```python
class Tracker2D:
    name: str
    def track(self, video, queries) -> tuple[tracks_2d, visible]: ...

class Tracker3D:
    name: str
    def track(self, video, queries, intrinsics) -> tuple[tracks_3d, visible]: ...
```

Loaders return one sample per sequence: `dict(name, video, queries, gt_tracks, gt_visible[, intrinsics])`, filtered by an optional subset file. Use the **query-first** protocol (first visible frame of each GT track) for both benchmarks.

## Metrics

- **`evaluate_2d`**: call tapnet's `compute_tapvid_metrics` (in `tapnet/tapvid/evaluation_datasets.py`; check the import path of the installed version) with `query_mode="first"`. Rescale the coordinates to **256×256** first, as the TAP-Vid protocol does, so that the thresholds 1/2/4/8/16 px match the paper. Return AJ, <δ_avg, OA.
- **`evaluate_3d`**: call the official TAPVid-3D `compute_tapvid3d_metrics` (`tapnet/tapvid3d/evaluation/metrics.py`) with `scaling="median"` (global median rescaling). Return 3D-AJ, APD_3D, OA. Pass **raw, unscaled** predictions: the function does the scaling itself. Our earlier D4RT numbers were pre-aligned and used `scaling="none"`, so they are not comparable.
- Both functions return per-sequence values and the subset mean. `scripts/evaluate.py` writes both to the CSV.

## Definition of done (needed for the PR to be merged)

1. **Ground truth as prediction**: AJ = 3D-AJ = 1.0 and OA = 1.0 on 2 sequences of each benchmark.
2. **Static tracker** (each query stays at its initial position, always visible): runs end to end and gives low but non-zero scores.
3. **Real tracker**: wrap CoTracker3 behind `Tracker2D` (it already runs in `scripts/test_cotracker_davis.py`) and check that DAVIS query-first AJ is close to the published value. This also tests the interface on a real model.
4. A short section in the README: where to put each dataset and the `scripts/evaluate.py` command.

## Why it matters

If each branch computes metrics its own way (resolution, query mode, scaling), the numbers in the final tables cannot be compared, and the oral exam will ask how they were computed. One reviewed implementation avoids both problems.
