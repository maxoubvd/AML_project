"""Pyramidal Lucas-Kanade baseline (Step 2, Q4)."""
import cv2
import numpy as np

from src.tracking.base import Tracker2D


class LKTracker(Tracker2D):
    """Pyramidal Lucas-Kanade (LK) point tracker, built on OpenCV.

    How LK works:
        LK assumes brightness constancy, I_t(x) = I_{t+1}(x + d), and that the
        displacement d is the same for every pixel of a small window around the
        point. A first-order Taylor expansion turns this into a 2x2 linear
        least-squares system, G d = b, where G is the structure tensor (sums of
        image-gradient products over the window). It is solved iteratively
        (Newton-Raphson) until the update is small. G must be well conditioned,
        so LK works on textured points and fails on flat regions or edges
        (the aperture problem).

        A single window only handles motions of a few pixels. The pyramid handles
        large motions: d is first estimated on a coarse, downsampled image, then
        refined level by level up to full resolution (coarse to fine).

    Tracking:
        Each query is chained frame to frame, forward from its query frame.
        Positions before the query frame are not scored under the query-first
        protocol and are filled with the query position.

    Visibility:
        A point is visible at frame t + 1 when LK converged in both directions,
        its forward-backward error (track to t + 1, then back to t, and measure
        the distance to the start) is below ``fb_threshold`` pixels, and it lies
        inside the image. Tracking continues from the predicted position even
        when the point is marked occluded, so it can be marked visible again if
        LK locks back on it.

    Args:
        win_size: side of the square LK window, in pixels, at each pyramid level.
        max_level: number of pyramid levels above full resolution (0 = no pyramid).
        fb_threshold: maximum forward-backward error, in pixels, for a visible point.
        max_iters, eps: stopping criteria of the iterative LK solver.
    """

    name = "lk"

    def __init__(self, win_size=21, max_level=3, fb_threshold=1.5, max_iters=30, eps=0.01):
        self.fb_threshold = fb_threshold
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps),
        )

    def _flow(self, img_a, img_b, pts):
        """Track pts (N, 1, 2) from img_a to img_b. Returns new points and a success mask."""
        new, status, _ = cv2.calcOpticalFlowPyrLK(img_a, img_b, pts, None, **self.lk_params)
        ok = status.reshape(-1).astype(bool) & np.isfinite(new).all(axis=(1, 2))
        return np.where(ok[:, None, None], new, pts), ok

    def track(self, video, queries):
        """Track queries (N, 3) as (t, x, y) through video (T, H, W, 3) uint8 RGB.

        Returns tracks (T, N, 2) as (x, y) pixels and visible (T, N) bool.
        """
        t_total, h, w, _ = video.shape
        n = len(queries)
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]

        query_t = queries[:, 0].astype(int)
        tracks = np.tile(queries[None, :, 1:], (t_total, 1, 1)).astype(np.float32)
        visible = np.zeros((t_total, n), dtype=bool)

        # Queries that share a query frame are tracked together
        for t0 in np.unique(query_t):
            idx = np.flatnonzero(query_t == t0)
            pts = queries[idx, 1:].astype(np.float32).reshape(-1, 1, 2)
            visible[t0, idx] = True

            for t in range(t0, t_total - 1):
                nxt, ok_fwd = self._flow(gray[t], gray[t + 1], pts)
                back, ok_bwd = self._flow(gray[t + 1], gray[t], nxt)

                fb_error = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
                xy = nxt.reshape(-1, 2)
                inside = (xy[:, 0] >= 0) & (xy[:, 0] <= w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1)

                tracks[t + 1, idx] = xy
                visible[t + 1, idx] = ok_fwd & ok_bwd & (fb_error < self.fb_threshold) & inside
                pts = nxt

        return tracks, visible
