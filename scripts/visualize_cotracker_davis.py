
import pickle
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from PIL import Image, ImageDraw

# Load benchmark data
dataset_path = Path(
    "data/tapvid_davis/tapvid_davis/tapvid_davis.pkl"
)

with dataset_path.open("rb") as f:
    dataset = pickle.load(f)

sample = dataset["breakdance"]

frames = sample["video"]
points = sample["points"]
occluded = sample["occluded"]

T, H, W, _ = frames.shape

# Load CoTracker predictions
output_dir = Path("results/davis/breakdance")

tracks = np.load(output_dir / "tracks.npy")
pred_visibility = np.load(output_dir / "visibility.npy")
valid = np.load(output_dir / "valid_points.npy")

# Keep only ground-truth points used as queries
gt_points = points[valid]
gt_occluded = occluded[valid]

# Convert normalized ground-truth coordinates to pixels
gt_points = gt_points.copy()
gt_points[..., 0] *= W - 1
gt_points[..., 1] *= H - 1

print("Ground truth:", gt_points.shape)
print("Predictions:", tracks.shape)

output_frames = []

for t in range(T):
    image = Image.fromarray(frames[t]).convert("RGB")
    draw = ImageDraw.Draw(image)

    for n in range(tracks.shape[1]):

        # Draw ground-truth trajectory in red
        for prev_t in range(1, t + 1):
            if not gt_occluded[n, prev_t - 1] and not gt_occluded[n, prev_t]:
                x1, y1 = gt_points[n, prev_t - 1]
                x2, y2 = gt_points[n, prev_t]

                draw.line(
                    (float(x1), float(y1), float(x2), float(y2)),
                    fill=(255, 60, 60),
                    width=2
                )

        # Draw predicted trajectory in green
        for prev_t in range(1, t + 1):
            if pred_visibility[prev_t - 1, n] and pred_visibility[prev_t, n]:
                x1, y1 = tracks[prev_t - 1, n]
                x2, y2 = tracks[prev_t, n]

                draw.line(
                    (float(x1), float(y1), float(x2), float(y2)),
                    fill=(50, 255, 80),
                    width=2
                )

        # Current ground-truth point
        if not gt_occluded[n, t]:
            x, y = gt_points[n, t]
            r = 4
            draw.ellipse(
                (x-r, y-r, x+r, y+r),
                fill=(255, 60, 60)
            )

        # Current prediction
        if pred_visibility[t, n]:
            x, y = tracks[t, n]
            r = 4
            draw.ellipse(
                (x-r, y-r, x+r, y+r),
                fill=(50, 255, 80)
            )

    # Legend
    draw.rectangle((10, 10, 245, 48), fill=(0, 0, 0))
    draw.text((18, 16), "Red: Ground truth | Green: Prediction", fill="white")

    output_frames.append(np.array(image))

output_path = output_dir / "comparison.mp4"

iio.imwrite(
    output_path,
    np.stack(output_frames),
    fps=10
)

print("Comparison saved:", output_path)
