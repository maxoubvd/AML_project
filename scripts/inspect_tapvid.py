
import pickle
from pathlib import Path

dataset_path = Path(
    "data/tapvid_davis/tapvid_davis/tapvid_davis.pkl"
)

print(f"Dataset size: {dataset_path.stat().st_size / 1e9:.2f} GB")

with dataset_path.open("rb") as f:
    dataset = pickle.load(f)

print("Dataset type:", type(dataset))
print("Number of videos:", len(dataset))
print("Video names:", list(dataset.keys()))

# Inspect one video
name = next(iter(dataset))
sample = dataset[name]

print("\nSample video:", name)
print("Available fields:", sample.keys())

for key, value in sample.items():
    if hasattr(value, "shape"):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"{key}: type={type(value)}")
