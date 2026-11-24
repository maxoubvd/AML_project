
import pickle
import torch
import numpy as np
import imageio.v3 as iio
from pathlib import Path

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_PATH = Path(
    "data/tapvid_davis/tapvid_davis/tapvid_davis.pkl"
)

VIDEO_NAME = "breakdance"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)

# --------------------------------------------------
# 2. Load TAP-Vid data
# --------------------------------------------------

print("Loading TAP-Vid dataset...")

with DATASET_PATH.open("rb") as f:
    dataset = pickle.load(f)

sample = dataset[VIDEO_NAME]

frames = sample["video"]
points = sample["points"]
occluded = sample["occluded"]

T, H, W, C = frames.shape
N = points.shape[0]

print("Video:", VIDEO_NAME)
print("Frames:", T)
print("Resolution:", W, "x", H)
print("Annotated points:", N)

# --------------------------------------------------
# 3. Prepare query points
# --------------------------------------------------

# TAP-Vid coordinates are normalized to [0, 1].
# CoTracker expects pixel coordinates.

points_px = points.copy()
points_px[..., 0] *= W - 1
points_px[..., 1] *= H - 1

# We query points at frame 0.
# Only use points visible in the first frame.

valid = ~occluded[:, 0]

points_initial = points_px[valid, 0]

print("Points visible at frame 0:", len(points_initial))

if len(points_initial) == 0:
    raise ValueError("No visible query points in the first frame.")

# CoTracker query format: (time, x, y)
queries = np.concatenate(
    [
        np.zeros((len(points_initial), 1), dtype=np.float32),
        points_initial.astype(np.float32)
    ],
    axis=1
)

queries = torch.from_numpy(queries).unsqueeze(0).to(device)

# --------------------------------------------------
# 4. Prepare video
# --------------------------------------------------

# Expected format: B x T x C x H x W

video = torch.from_numpy(frames).permute(0, 3, 1, 2)
video = video.unsqueeze(0).float().to(device)

print("Video tensor:", video.shape)
print("Query tensor:", queries.shape)

# --------------------------------------------------
# 5. Load CoTracker3
# --------------------------------------------------

print("Loading CoTracker3...")

cotracker = torch.hub.load(
    "facebookresearch/co-tracker",
    "cotracker3_offline"
)

cotracker = cotracker.to(device).eval()

print("CoTracker3 loaded!")

# --------------------------------------------------
# 6. Run tracking
# --------------------------------------------------

print("Running tracking...")

with torch.inference_mode():
    pred_tracks, pred_visibility = cotracker(
        video,
        queries=queries
    )

print("Tracking successful!")

print("Predicted tracks:", pred_tracks.shape)
print("Predicted visibility:", pred_visibility.shape)

# --------------------------------------------------
# 7. Save predictions
# --------------------------------------------------

output_dir = Path("results/davis/breakdance")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "tracks.npy",
        pred_tracks[0].cpu().numpy())

np.save(output_dir / "visibility.npy",
        pred_visibility[0].cpu().numpy())

np.save(output_dir / "valid_points.npy",
        valid)

print("Predictions saved in:", output_dir)
