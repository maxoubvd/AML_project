
from pathlib import Path
import numpy as np
import cv2

root = Path("Open-d4rt")
data = np.load(root / "data/worldtrack_release/adt_mini/Apartment_release_clean_seq148_7.npz")
result = np.load(root / "tmp/eval_apartment_failure_64/adt_mini/apartment_failure.npz")

gt = result["gt_tracks"]
pred = result["pred_tracks"]
visible = result["visibility"]

# Camera parameters
fx, fy, cx, cy = data["fx_fy_cx_cy"]

# Match the coordinate conversion used by the evaluator
extr = data["extrinsics_w2c"][:64]
relative_w2c = extr @ np.linalg.inv(extr[0])

def project(world_point, frame):
    camera_point = (
        relative_w2c[frame, :3, :3] @ world_point
        + relative_w2c[frame, :3, 3]
    )
    x, y, z = camera_point
    if not np.isfinite(camera_point).all() or z <= 1e-8:
        return None
    return (int(round(fx * x / z + cx)),
            int(round(fy * y / z + cy)))

# Select the three points with the largest mean visible-frame EPE
errors = np.linalg.norm(pred - gt, axis=2)
mean_errors = np.array([
    errors[visible[:, i], i].mean()
    for i in range(gt.shape[1])
])
worst_points = np.array([178, 40, 36])

output = Path("results/d4rt_examples")
output.mkdir(parents=True, exist_ok=True)

for number, point in enumerate(worst_points, start=1):
    # Choose the visible frame with the largest error
    frame_errors = np.where(visible[:, point], errors[:, point], -np.inf)
    frame = int(np.argmax(frame_errors))

    image_bytes = data["images_jpeg_bytes"][frame].tobytes()
    image = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    gt_uv = project(gt[frame, point], frame)
    pred_uv = project(pred[frame, point], frame)

    if gt_uv is not None:
        cv2.circle(image, gt_uv, 7, (0, 255, 0), -1)

    if pred_uv is not None:
        cv2.circle(image, pred_uv, 7, (0, 0, 255), -1)

    cv2.putText(
        image,
        f"Point {point} | Frame {frame + 1} | 3D EPE {errors[frame, point]:.3f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    path = output / f"failure_example_{number}.png"
    cv2.imwrite(str(path), image)
    print(f"Saved: {path} | GT: {gt_uv} | Prediction: {pred_uv}")
