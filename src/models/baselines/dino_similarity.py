"""DINOv3 feature-similarity baseline (Step 2, Q4)."""
import numpy as np

from src.models.baselines.dino_features import DinoFeatures, init_outputs, query_descriptors
from src.tracking.base import Tracker2D


class DinoSimilarityTracker(Tracker2D):
    """Track each point by nearest-neighbour matching of DINOv3 features.

    The query descriptor is the DINOv3 feature bilinearly sampled at the query
    point in its query frame. In every later frame, the cosine similarity
    between this descriptor and all patches is computed, the best patch is
    taken (global search) and refined to sub-patch precision by a soft-argmax.

    Frames are matched independently: there is no drift and a point can be
    found again after an occlusion, but nothing prevents jumps between similar
    parts (e.g. the two legs of a person), and the precision is limited by the
    16-pixel patch grid.

    Visibility: the point is visible when the best similarity is above
    ``vis_threshold``.

    Args:
        vis_threshold: minimum cosine similarity for a visible point.
        tau: soft-argmax temperature.
        refine_radius: soft-argmax neighbourhood radius, in patches.
        **feature_kwargs: passed to DinoFeatures (model_name, scale, cache_dir).

    Defaults were chosen by grid search on configs/subsets/davis_tuning.txt
    (scripts/tune.py), outside the evaluation subsets.
    """

    name = "dino_sim"

    def __init__(self, vis_threshold=0.85, tau=0.01, refine_radius=1, **feature_kwargs):
        self.vis_threshold = vis_threshold
        self.tau = tau
        self.refine_radius = refine_radius
        self.features = DinoFeatures(**feature_kwargs)

    def track(self, video, queries):
        fmap = self.features.extract(video)
        desc = query_descriptors(fmap, queries)
        tracks, visible = init_outputs(queries, len(video))
        query_t = queries[:, 0].astype(int)

        for t in range(query_t.min() + 1, len(video)):
            active = np.flatnonzero(query_t < t)
            xy, score = fmap.match(fmap.similarity(t, desc[active]), tau=self.tau, refine_radius=self.refine_radius)
            tracks[t, active] = xy.cpu().numpy()
            visible[t, active] = (score >= self.vis_threshold).cpu().numpy()

        return tracks, visible
