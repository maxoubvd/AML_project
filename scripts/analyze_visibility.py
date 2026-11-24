
import pickle
import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Load TAP-Vid DAVIS dataset
# --------------------------------------------------

dataset_path = Path("data/tapvid_davis/tapvid_davis/tapvid_davis.pkl")

print("Loading TAP-Vid dataset...")

with dataset_path.open("rb") as f:
    dataset = pickle.load(f)

sample = dataset["breakdance"]

gt_occluded = sample["occluded"]
gt_points = sample["points"]
video = sample["video"]

H, W = video.shape[1:3]

# --------------------------------------------------
# 2. Load CoTracker3 predictions
# --------------------------------------------------

output_dir = Path("results/davis/breakdance")

pred_tracks = np.load(output_dir / "tracks.npy")
pred_visibility = np.load(output_dir / "visibility.npy")
valid = np.load(output_dir / "valid_points.npy")

print("Dataset loaded.")
print("Ground-truth points:", gt_points.shape)
print("Predicted tracks:", pred_tracks.shape)
print("Predicted visibility:", pred_visibility.shape)

# --------------------------------------------------
# 3. Keep only points initially visible at frame 0
# --------------------------------------------------

gt_points = gt_points[valid].copy()
gt_occluded = gt_occluded[valid]

# Convert normalized ground-truth coordinates to pixels
gt_points[..., 0] *= W - 1
gt_points[..., 1] *= H - 1

gt_visible = ~gt_occluded

# --------------------------------------------------
# 4. Select point 4 and its reappearance
# --------------------------------------------------

point_id = 4

# Find first occlusion
occlusion_frames = np.where(~gt_visible[point_id])[0]

if len(occlusion_frames) == 0:
    raise ValueError("This point is never occluded.")

first_occlusion = occlusion_frames[0]

# Find first subsequent visible frame
later_visible = np.where(
    gt_visible[point_id] &
    (np.arange(gt_visible.shape[1]) > first_occlusion)
)[0]

if len(later_visible) == 0:
    raise ValueError("This point never reappears.")

recovery = later_visible[0]

# --------------------------------------------------
# 5. Compare positions and visibility
# --------------------------------------------------

gt_position = gt_points[point_id, recovery]
pred_position = pred_tracks[recovery, point_id]

distance = np.linalg.norm(gt_position - pred_position)

print("\n========== DIAGNOSTIC ==========")

print(f"Point ID: {point_id}")
print(f"First occlusion: frame {first_occlusion}")
print(f"Reappears in ground truth: frame {recovery}")

print("\nVisibility:")
print("Ground truth:", gt_visible[point_id, recovery])
print("CoTracker3:", pred_visibility[recovery, point_id])

print("\nPosition at reappearance:")
print("Ground truth:", gt_position)
print("CoTracker3:", pred_position)

print(f"\nPosition error: {distance:.2f} pixels")

# --------------------------------------------------
# 6. Inspect frames around occlusion and recovery
# --------------------------------------------------

print("\n========== FRAME-BY-FRAME ==========")

start = max(0, first_occlusion - 2)
end = min(gt_visible.shape[1], recovery + 4)

for frame in range(start, end):
    gt_vis = gt_visible[point_id, frame]
    pred_vis = pred_visibility[frame, point_id]

    gt_pos = gt_points[point_id, frame]
    pred_pos = pred_tracks[frame, point_id]

    error = np.linalg.norm(gt_pos - pred_pos)

    print(
        f"Frame {frame:02d} | "
        f"GT visible: {str(bool(gt_vis)):5s} | "
        f"Pred visible: {str(bool(pred_vis)):5s} | "
        f"Position error: {error:7.2f} px"
    )

print("\nDiagnostic complete.")
