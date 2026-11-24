
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dataset_path = Path(
    "data/tapvid_davis/tapvid_davis/tapvid_davis.pkl"
)

# Load dataset
with dataset_path.open("rb") as f:
    dataset = pickle.load(f)

# Select one sequence
sample = dataset["breakdance"]

video = sample["video"]
points = sample["points"]
occluded = sample["occluded"]

print("Video shape:", video.shape)
print("Points shape:", points.shape)
print("Occlusion shape:", occluded.shape)
print("Number of tracked points:", points.shape[0])

# Convert normalized coordinates to pixel coordinates
height, width = video.shape[1:3]

points_px = points.copy()
points_px[..., 0] *= width - 1   # x coordinates
points_px[..., 1] *= height - 1  # y coordinates

# Visualize first, middle and last frames
T = video.shape[0]
frame_indices = [0, T // 2, T - 1]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, t in zip(axes, frame_indices):
    ax.imshow(video[t])

    # Only display points that are not occluded
    visible = ~occluded[:, t]

    ax.scatter(
        points_px[visible, t, 0],
        points_px[visible, t, 1],
        s=35,
        c="lime",
        edgecolors="black"
    )

    ax.set_title(f"Frame {t} — {visible.sum()} visible points")
    ax.axis("off")

plt.tight_layout()

Path("results").mkdir(exist_ok=True)
plt.savefig("results/davis_breakdance_preview.png", dpi=150)
plt.show()
