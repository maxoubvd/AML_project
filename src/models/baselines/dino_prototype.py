"""DINOv3 prototype-based matching baseline (Step 2, Q4)."""
import itertools

import numpy as np
import torch
import torch.nn.functional as F

from src.models.baselines.dino_features import DinoFeatures, init_outputs
from src.tracking.base import Tracker2D


class DinoPrototypeTracker(Tracker2D):
    """Track each point by matching an adaptive DINOv3 prototype.

    Our definition of "prototype" (the assignment does not fix it; to be
    confirmed with the TA):
      1. Initialisation: the prototype is the normalised mean of the DINOv3
         features sampled on a 3x3 neighbourhood around the query point
         (offsets of +/- ``neighbourhood`` patches), which is less sensitive to
         the exact query position than a single feature.
      2. Matching: in each frame, global search of the patch most similar to
         the prototype, refined by a soft-argmax (as in the similarity baseline).
      3. Update: when the match is confident (similarity >= update_threshold),
         the prototype moves towards the matched feature with an exponential
         moving average, p <- normalise((1 - momentum) * p + momentum * f).
         This follows slow appearance changes (rotation, lighting), at the risk
         of drifting if a wrong match is accepted.

    Visibility: the best similarity is above ``vis_threshold``.

    Defaults were chosen by grid search on configs/subsets/davis_tuning.txt
    (scripts/tune.py), outside the evaluation subsets.
    The search selected neighbourhood=0 (the prototype starts as the single
    query feature): averaging a neighbourhood did not help on the tuning set.
    """

    name = "dino_proto"

    def __init__(self, vis_threshold=0.85, update_threshold=0.95, momentum=0.02, neighbourhood=0.0,
                 tau=0.01, refine_radius=1, **feature_kwargs):
        self.vis_threshold = vis_threshold
        self.update_threshold = update_threshold
        self.momentum = momentum
        self.neighbourhood = neighbourhood
        self.tau = tau
        self.refine_radius = refine_radius
        self.features = DinoFeatures(**feature_kwargs)

    def _init_prototypes(self, fmap, queries):
        sx, sy = fmap.stride
        steps = (-self.neighbourhood, 0.0, self.neighbourhood)
        offsets = np.array([(dx * sx, dy * sy) for dx, dy in itertools.product(steps, steps)])
        query_t = queries[:, 0].astype(int)
        proto = torch.empty((len(queries), fmap.feats.shape[-1]), device=fmap.feats.device)
        for t0 in np.unique(query_t):
            idx = np.flatnonzero(query_t == t0)
            xy = queries[idx, 1:]
            samples = torch.stack([fmap.sample(t0, xy + off) for off in offsets])  # (9, n, C)
            proto[idx] = F.normalize(samples.mean(dim=0), dim=-1)
        return proto

    def track(self, video, queries):
        fmap = self.features.extract(video)
        proto = self._init_prototypes(fmap, queries)
        tracks, visible = init_outputs(queries, len(video))
        query_t = queries[:, 0].astype(int)

        for t in range(query_t.min() + 1, len(video)):
            active = np.flatnonzero(query_t < t)
            xy, score = fmap.match(fmap.similarity(t, proto[active]), tau=self.tau, refine_radius=self.refine_radius)
            tracks[t, active] = xy.cpu().numpy()
            visible[t, active] = (score >= self.vis_threshold).cpu().numpy()

            confident = (score >= self.update_threshold).cpu().numpy()
            if confident.any():
                upd = active[confident]
                matched = fmap.sample(t, xy[confident])
                proto[upd] = F.normalize((1 - self.momentum) * proto[upd] + self.momentum * matched, dim=-1)

        return tracks, visible
