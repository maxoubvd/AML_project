
import pickle
import cv2
from pathlib import Path

dataset_path = Path("data/tapvid_davis/tapvid_davis/tapvid_davis.pkl")

with dataset_path.open("rb") as f:
    dataset = pickle.load(f)

video = dataset["breakdance"]["video"]

# Select frame 14
frame = video[14]

# Save the original RGB frame
output_path = Path("results/davis/breakdance/original_frame.png")

cv2.imwrite(
    str(output_path),
    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
)

print("Saved:", output_path)
