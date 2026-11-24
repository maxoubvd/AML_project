import pickle
import numpy as np
from pathlib import Path


def load_tapvid(path, subset_file=None):
    """Load DAVIS sequences using the query-first protocol."""

    with open(path, "rb") as f:
        dataset = pickle.load(f)

    names = (
        Path(subset_file).read_text().split()
        if subset_file else dataset.keys()
    )

    for name in names:
        sample = dataset[name]
        video = sample["video"]
        _, h, w, _ = video.shape

        tracks = sample["points"].transpose(1, 0, 2)
        tracks = tracks * np.array([w, h])

        visible = ~sample["occluded"].T

        valid = visible.any(axis=0)
        tracks = tracks[:, valid]
        visible = visible[:, valid]

        first = visible.argmax(axis=0)
        queries = np.column_stack(
            (first, tracks[first, np.arange(len(first))])
        )

        yield {
            "name": name,
            "video": video,
            "queries": queries,
            "gt_tracks": tracks,
            "gt_visible": visible,
        }