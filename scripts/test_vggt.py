import torch
from vggt.models.vggt import VGGT

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = VGGT.from_pretrained("facebook/VGGT-1B")
model = model.to(device).eval()

print("VGGT loaded successfully!")