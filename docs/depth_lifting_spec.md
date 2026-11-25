# Branch `feat/depth-lifting`: depth, camera poses and 2D-to-3D lifting (Q7, base for Q8)

**Owner:** TBD · **Base:** `main` · **Merge:** PR into `main`

## Goal

Provide, for any TAPVid-3D video, per-frame depth, intrinsics and camera poses, and one function that lifts 2D tracks to 3D in the convention `evaluate_3d` expects: per-frame camera coordinates. Every Step 3 method (VGGT adaptation and the 5 "2D tracker + depth" baselines) is built on it.

## Deliverables

```
src/models/depth/base.py            # DepthEstimator interface
src/models/depth/vggt_depth.py      # VGGT depth + cameras
src/models/depth/depth_anything.py  # Depth Anything
src/tracking/lifting.py             # lift_tracks(), camera <-> world transforms
```

## Interface

```python
class DepthEstimator:
    name: str
    def estimate(self, video) -> dict(
        depth,          # (T, H, W) float, same resolution as the video
        intrinsics,     # (T, 4) fx, fy, cx, cy
        cam_to_world,   # (T, 4, 4), frame 0 = identity
    ): ...

def lift_tracks(tracks_2d, visible, depth, intrinsics) -> tracks_3d  # (T, N, 3), camera frame of t
```

`lift_tracks` samples the depth bilinearly at each 2D position and back-projects it: `X = (u - cx) / fx * Z`, `Y = (v - cy) / fy * Z`. Where the point is **occluded**, the depth at its 2D position belongs to the occluder. Hold or interpolate the last visible depth instead, and document the rule.

## Depth sources

| Source | Depth | Intrinsics / poses | Note |
|---|---|---|---|
| **VGGT** | depth head | camera head (`pose_encoding_to_extri_intri`) | Multi-frame, so the scale is consistent across frames. Use windows for long videos |
| **Depth Anything** | depth only (V2) or depth + cameras (3) | from the model, or GT intrinsics from the dataset | Per-frame monocular depth can change scale between frames, so check temporal consistency |

**Scale:** `evaluate_3d` applies one global median scale per sequence. A depth whose scale drifts over time therefore cannot be corrected and inflates the error. Report this when it happens.

## Why poses are needed

The assignment notes that the baselines need a common reference frame. The metrics use per-frame camera coordinates, but motion models (the constant velocity of `feat/3d-tracking`) and world-frame outputs (VGGT world points) must be converted with `cam_to_world`. Provide `camera_to_world()` and `world_to_camera()` in `lifting.py`.

## Caching

Cache depth, intrinsics and poses per (estimator, sequence) in `cache/depth/<estimator>/<sequence>.npz` (git-ignored). All Step 3 baselines read the cache.

## Definition of done

1. **Exact lifting test:** project the GT 3D tracks to 2D, lift them back with the GT depth of those points, and check 3D-AJ = 1.0.
2. **Upper bound:** GT 2D tracks + each estimated depth, scored with `evaluate_3d` on `configs/subsets/tapvid3d_test.txt`. This isolates the error due to depth alone and is useful for the report.
3. Both estimators run on the TAPVid-3D test subset and are cached.
4. A paragraph for Q7: how VGGT is lifted to 3D (its own depth and poses vs Depth Anything).
