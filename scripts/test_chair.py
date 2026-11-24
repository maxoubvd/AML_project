
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Paths (relative to the repository root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "images" / "chair"
OUTPUT_PATH = PROJECT_ROOT / "results" / "depth_maps_chair.png"

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Load images
image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
print(f"Found {len(image_paths)} images")

images = load_and_preprocess_images(
    [str(path) for path in image_paths]
).to(device)

print("Input tensor:", images.shape)

# Load model
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

# Inference
with torch.no_grad():
    predictions = model(images)

# Inspect outputs
for key, value in predictions.items():
    if torch.is_tensor(value):
        print(f"{key}: {value.shape}")
    elif isinstance(value, list):
        print(f"{key}: list with {len(value)} elements")
    else:
        print(f"{key}: {type(value)}")


import matplotlib.pyplot as plt

# Extract predictions
depth = predictions["depth"][0].cpu().numpy()
world_points = predictions["world_points"][0].cpu().numpy()

# Display depth maps
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for i, ax in enumerate(axes):
    ax.imshow(depth[i, :, :, 0], cmap="viridis")
    ax.set_title(f"View {i + 1}")
    ax.axis("off")

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150)
plt.show()

print(f"Saved depth maps to {OUTPUT_PATH}")
