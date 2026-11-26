"""LK + DINOv3 baseline (Step 2, Q4)."""
import cv2
import numpy as np
import torch

from src.models.baselines.dino_features import DinoFeatures, init_outputs, query_descriptors
from src.models.baselines.lk import LKTracker
from src.tracking.base import Tracker2D


class LKDinoTracker(Tracker2D):
    """Pyramidal LK for precise frame-to-frame motion, DINOv3 to detect and fix its failures.

    At each step every active point is first moved by LK (see LKTracker). The
    LK step is accepted when LK itself succeeds (convergence, forward-backward
    error, inside the image) AND the DINOv3 feature at the new position still
    looks like the query (cosine similarity >= ``sim_threshold``). The second
    test catches LK drifting onto another surface, which LK alone cannot see.

    Otherwise the point is re-localised by a DINOv3 search in a window of
    ``search_radius`` patches around its previous position. If the best
    similarity passes ``sim_threshold`` the point jumps there and is visible;
    if not, it is marked occluded and keeps its previous position.

    LK gives sub-pixel precision and DINOv3 gives robustness to drift and
    recovery after short occlusions.

    Defaults were chosen by grid search on configs/subsets/davis_tuning.txt
    (scripts/tune.py), outside the evaluation subsets.
    """

    name = "lk_dino"

    def __init__(self, sim_threshold=0.85, search_radius=5, tau=0.01, refine_radius=1,
                 lk_kwargs=None, **feature_kwargs):
        self.sim_threshold = sim_threshold
        self.search_radius = search_radius
        self.tau = tau
        self.refine_radius = refine_radius
        self.lk = LKTracker(**(lk_kwargs or {}))
        self.features = DinoFeatures(**feature_kwargs)

    def track(self, video, queries):
        fmap = self.features.extract(video)
        desc = query_descriptors(fmap, queries)
        tracks, visible = init_outputs(queries, len(video))
        query_t = queries[:, 0].astype(int)
        _, h, w, _ = video.shape
        gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
        pos = queries[:, 1:].astype(np.float32).copy()

        for t in range(query_t.min(), len(video) - 1):
            a = np.flatnonzero(query_t <= t)
            pts = pos[a].reshape(-1, 1, 2)

            # 1. LK step with its own failure tests
            nxt, ok_fwd = self.lk._flow(gray[t], gray[t + 1], pts)
            back, ok_bwd = self.lk._flow(gray[t + 1], gray[t], nxt)
            xy_lk = nxt.reshape(-1, 2)
            fb_error = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            inside = (xy_lk[:, 0] >= 0) & (xy_lk[:, 0] <= w - 1) & (xy_lk[:, 1] >= 0) & (xy_lk[:, 1] <= h - 1)
            lk_ok = ok_fwd & ok_bwd & (fb_error < self.lk.fb_threshold) & inside

            # 2. Appearance check of the LK position with DINOv3
            sim_lk = (fmap.sample(t + 1, xy_lk) * desc[a]).sum(dim=-1).cpu().numpy()
            accept = lk_ok & (sim_lk >= self.sim_threshold)

            new = pos[a].copy()
            vis = accept.copy()
            new[accept] = xy_lk[accept]

            # 3. DINOv3 re-localisation around the previous position
            redo = np.flatnonzero(~accept)
            if len(redo):
                xy_d, score = fmap.match(fmap.similarity(t + 1, desc[a[redo]]), center_xy=pos[a[redo]],
                                         radius=self.search_radius, tau=self.tau, refine_radius=self.refine_radius)
                found = (score >= self.sim_threshold).cpu().numpy()
                new[redo[found]] = xy_d.cpu().numpy()[found]
                vis[redo] = found

            pos[a] = new
            tracks[t + 1, a] = new
            visible[t + 1, a] = vis

        return tracks, visible
