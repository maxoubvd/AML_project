"""DINOv3 + constant velocity baseline (Step 2, Q4)."""
import numpy as np
import torch

from src.models.baselines.dino_features import DinoFeatures, init_outputs, query_descriptors
from src.tracking.base import Tracker2D


class DinoConstantVelocityTracker(Tracker2D):
    """DINOv3 matching guided by a constant-velocity motion model.

    At each frame the position is predicted as x_pred = x_{t-1} + v, where v is
    the last observed displacement (v = 0 at the query frame). The query
    descriptor is then matched only inside a square window of
    ``search_radius`` patches around x_pred (local search), which prevents the
    jumps to similar but distant parts that the global similarity baseline
    makes.

    If the best similarity in the window is above ``vis_threshold``, the point
    is visible, moves to the match, and v is updated to the new displacement.
    Otherwise the point is marked occluded and coasts on the prediction with
    an unchanged v (clipped to the image), so it can be found again when it
    reappears near the predicted path. Abrupt direction changes break the
    motion model.
    """

    name = "dino_cv"

    def __init__(self, vis_threshold=0.5, search_radius=3, tau=0.05, refine_radius=1, **feature_kwargs):
        self.vis_threshold = vis_threshold
        self.search_radius = search_radius
        self.tau = tau
        self.refine_radius = refine_radius
        self.features = DinoFeatures(**feature_kwargs)

    def track(self, video, queries):
        fmap = self.features.extract(video)
        desc = query_descriptors(fmap, queries)
        tracks, visible = init_outputs(queries, len(video))
        query_t = queries[:, 0].astype(int)
        dev = desc.device
        limits = torch.tensor([video.shape[2] - 1, video.shape[1] - 1], dtype=torch.float32, device=dev)

        pos = torch.as_tensor(queries[:, 1:], dtype=torch.float32, device=dev)
        vel = torch.zeros_like(pos)

        for t in range(query_t.min() + 1, len(video)):
            a = torch.as_tensor(np.flatnonzero(query_t < t), device=dev)
            pred = torch.minimum(torch.clamp(pos[a] + vel[a], min=0), limits)
            xy, score = fmap.match(fmap.similarity(t, desc[a]), center_xy=pred, radius=self.search_radius,
                                   tau=self.tau, refine_radius=self.refine_radius)
            ok = score >= self.vis_threshold
            new = torch.where(ok[:, None], xy, pred)
            vel[a] = torch.where(ok[:, None], new - pos[a], vel[a])
            pos[a] = new

            a_np = a.cpu().numpy()
            tracks[t, a_np] = new.cpu().numpy()
            visible[t, a_np] = ok.cpu().numpy()

        return tracks, visible
