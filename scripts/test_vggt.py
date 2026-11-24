import torch
from vggt.models.vggt import VGGT

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

model = VGGT.from_pretrained("facebook/VGGT-1B")
model = model.to(device).eval()

print("VGGT loaded successfully!")