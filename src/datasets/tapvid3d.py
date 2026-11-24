import io
import numpy as np
from pathlib import Path
from PIL import Image


def load_tapvid3d(path, subset_file=None):
    """Load 3D sequences with frame-first camera coordinates
    and query-first pixel queries (t, x, y).
    """

    files = sorted(Path(path).rglob("*.npz"))

    if subset_file:
        names = Path(subset_file).read_text().split()
        files = [p for p in files if p.stem in names]

    for file in files:
        with np.load(file) as data:
            video = np.stack([
                np.array(Image.open(io.BytesIO(b)).convert("RGB"))
                for b in data["images_jpeg_bytes"]
            ])

            tracks = data["tracks_XYZ"]
            visible = data["visibility"].astype(bool)
            intrinsics = data["fx_fy_cx_cy"]

        valid = visible.any(axis=0)
        tracks = tracks[:, valid]
        visible = visible[:, valid]

        first = visible.argmax(axis=0)
        xyz = tracks[first, np.arange(len(first))]

        uv = (
            xyz[:, :2] / xyz[:, 2:3]
            * intrinsics[:2] + intrinsics[2:]
        )

        queries = np.column_stack((first, uv))

        yield {
            "name": file.stem,
            "video": video,
            "queries": queries,
            "gt_tracks": tracks,
            "gt_visible": visible,
            "intrinsics": intrinsics,
        }
        