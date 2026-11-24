
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Paths
VGGT_DIR = Path(
    "/Users/alex_dbrz/Work/TELECOM/3A-TURIN/Advanced Machine Learning/PROJET/vggt"
)
IMAGE_DIR = VGGT_DIR / "test_images"

# Device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load images
image_paths = sorted(IMAGE_DIR.glob("*.jpeg"))
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
plt.savefig("depth_maps_chair.png", dpi=150)
plt.show()

print("Saved depth maps to depth_maps_chair.png")
