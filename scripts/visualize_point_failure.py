
import pickle
import cv2
import imageio.v2 as imageio
import numpy as np
from pathlib import Path

# 1. Load dataset and predictions
dataset_path = Path("data/tapvid_davis/tapvid_davis/tapvid_davis.pkl")

print("Loading dataset...")
with dataset_path.open("rb") as f:
    dataset = pickle.load(f)

sample = dataset["breakdance"]

video = sample["video"]
gt_points = sample["points"]
gt_occluded = sample["occluded"]

output_dir = Path("results/davis/breakdance")

pred_tracks = np.load(output_dir / "tracks.npy")
pred_visibility = np.load(output_dir / "visibility.npy")
valid = np.load(output_dir / "valid_points.npy")

# 2. Select point 4
point_id = 4

gt_points = gt_points[valid].copy()
gt_occluded = gt_occluded[valid]

H, W = video.shape[1:3]

gt_points[..., 0] *= W - 1
gt_points[..., 1] *= H - 1

# 3. Generate annotated frames
frames = []

for frame_id in range(3, 18):

    # Copy original RGB image
    frame = video[frame_id].copy()

    gt_visible = not gt_occluded[point_id, frame_id]
    pred_visible = bool(pred_visibility[frame_id, point_id])

    gt_pos = gt_points[point_id, frame_id]
    pred_pos = pred_tracks[frame_id, point_id]

    # OpenCV drawing uses BGR colors.
    # Convert RGB -> BGR before drawing.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Ground truth: red
    if gt_visible:
        x, y = np.round(gt_pos).astype(int)
        cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)

    # CoTracker3: green
    if pred_visible:
        x, y = np.round(pred_pos).astype(int)
        cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

    # Frame information
    cv2.putText(
        frame,
        f"Frame: {frame_id}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"GT visible: {gt_visible}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255) if gt_visible else (180, 180, 180),
        2
    )

    cv2.putText(
        frame,
        f"CoTracker visible: {pred_visible}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if pred_visible else (180, 180, 180),
        2
    )

    # Convert BGR -> RGB for ImageIO
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frames.append(frame)

# 4. Save using ImageIO
output_path = output_dir / "point_4_failure.mp4"

imageio.mimwrite(
    output_path,
    frames,
    fps=8,
    codec="libx264",
    pixelformat="yuv420p"
)

print("Video saved:", output_path)
