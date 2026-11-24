
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from vggt.models.vggt import VGGT

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)

dataset_path = Path(
    "data/tapvid_davis/tapvid_davis/tapvid_davis.pkl"
)

output_dir = Path("results/davis/breakdance")
output_dir.mkdir(parents=True, exist_ok=True)

# VGGT expects dimensions compatible with its patch size (14).
# We resize the original 480x854 frames to 518x924.
TARGET_H = 518
TARGET_W = 924

NUM_FRAMES = 18

# --------------------------------------------------
# 2. Load TAP-Vid
# --------------------------------------------------

print("\nLoading TAP-Vid dataset...")

with dataset_path.open("rb") as f:
    dataset = pickle.load(f)

sample = dataset["breakdance"]

video_np = sample["video"]
points = sample["points"]
occluded = sample["occluded"]

H, W = video_np.shape[1:3]

print("Original video:", video_np.shape)

# --------------------------------------------------
# 3. Load the same query points as CoTracker3
# --------------------------------------------------

valid = np.load(output_dir / "valid_points.npy")

# Keep the same 21 points initially visible at frame 0.
initial_points = points[valid, 0].copy()

# TAP-Vid coordinates are normalized to [0, 1].
# Convert them to original image pixel coordinates.
initial_points[:, 0] *= W - 1
initial_points[:, 1] *= H - 1

print("Number of query points:", len(initial_points))

# --------------------------------------------------
# 4. Prepare video for VGGT
# --------------------------------------------------

print("\nPreparing video...")

# Select the first 18 frames.
video_np = video_np[:NUM_FRAMES]

# Convert to tensor: [T, H, W, C] -> [T, C, H, W]
video = torch.from_numpy(video_np).permute(0, 3, 1, 2)

# Convert uint8 [0,255] to float [0,1].
video = video.float() / 255.0

# Resize every frame.
video = F.interpolate(
    video,
    size=(TARGET_H, TARGET_W),
    mode="bilinear",
    align_corners=False
)

# Add batch dimension: [1, T, C, H, W]
video = video.unsqueeze(0).to(device)

print("VGGT video tensor:", video.shape)

# --------------------------------------------------
# 5. Resize query coordinates
# --------------------------------------------------

# Scale query coordinates to match resized images.
initial_points[:, 0] *= TARGET_W / W
initial_points[:, 1] *= TARGET_H / H

query_points = torch.from_numpy(
    initial_points.astype(np.float32)
).unsqueeze(0).to(device)

print("Query tensor:", query_points.shape)

# --------------------------------------------------
# 6. Load pretrained VGGT
# --------------------------------------------------

print("\nLoading VGGT...")

model = VGGT.from_pretrained("facebook/VGGT-1B")
model = model.to(device).eval()

# We only need the tracking head for this experiment.
# Disable the other prediction heads to save memory.
model.camera_head = None
model.depth_head = None
model.point_head = None

print("VGGT loaded!")

# --------------------------------------------------
# 7. Run native VGGT tracking
# --------------------------------------------------

print("\nRunning VGGT tracking...")

with torch.inference_mode():
    predictions = model(
        video,
        query_points=query_points
    )

# Remove batch dimension.
tracks = predictions["track"][0].cpu().numpy()
visibility_scores = predictions["vis"][0].cpu().numpy()
confidence_scores = predictions["conf"][0].cpu().numpy()

print("\nTracking successful!")
print("Tracks:", tracks.shape)
print("Visibility scores:", visibility_scores.shape)
print("Confidence scores:", confidence_scores.shape)

# --------------------------------------------------
# 8. Save predictions
# --------------------------------------------------

np.save(output_dir / "vggt_tracks.npy", tracks)
np.save(output_dir / "vggt_visibility.npy", visibility_scores)
np.save(output_dir / "vggt_confidence.npy", confidence_scores)

print("\nPredictions saved in:", output_dir)

print("\n========== VGGT DIAGNOSTICS ==========")

print("Track coordinate range:")
print("X:", tracks[..., 0].min(), "to", tracks[..., 0].max())
print("Y:", tracks[..., 1].min(), "to", tracks[..., 1].max())

print("\nVisibility score range:")
print("Min:", visibility_scores.min())
print("Max:", visibility_scores.max())

print("\nConfidence score range:")
print("Min:", confidence_scores.min())
print("Max:", confidence_scores.max())

print("\nFirst query point:")
print("Input:", query_points[0, 0].cpu().numpy())
print("Predicted:", tracks[0, 0])