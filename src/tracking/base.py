import numpy as np
from abc import ABC, abstractmethod


class Tracker2D(ABC):
    name: str

    @abstractmethod
    def track(self, video, queries):
        """Return (T, N, 2) tracks and (T, N) visibility."""
        pass


class Tracker3D(ABC):
    name: str

    @abstractmethod
    def track(self, video, queries, intrinsics):
        """Return (T, N, 3) camera tracks and visibility."""
        pass


class StaticTracker2D(Tracker2D):
    name = "static_2d"

    def track(self, video, queries):
        t, n = len(video), len(queries)

        tracks = np.tile(queries[:, 1:], (t, 1, 1))
        visible = np.ones((t, n), dtype=bool)

        return tracks, visible


class StaticTracker3D(Tracker3D):
    name = "static_3d"

    def track(self, video, queries, intrinsics):
        t, n = len(video), len(queries)

        xy = (queries[:, 1:] - intrinsics[2:]) / intrinsics[:2]
        xyz = np.column_stack((xy, np.ones(n)))

        tracks = np.tile(xyz, (t, 1, 1))
        visible = np.ones((t, n), dtype=bool)

        return tracks, visible