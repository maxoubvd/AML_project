"""Dense DINOv3 features with a disk cache, and the matching helpers shared by
the four DINOv3 baselines (Step 2, Q4).

DINOv3 is a self-supervised Vision Transformer. It cuts the image into 16x16
patches and returns one feature vector per patch that describes the local
content and stays fairly stable under appearance changes. A point is matched
across frames by comparing these vectors with cosine similarity.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FeatureMap:
    """L2-normalised patch features of one video.

    feats: (T, h, w, C) float16 tensor on the compute device.
    height, width: original video resolution, in pixels.
    Patch (i, j) covers the pixels [j * sx, (j + 1) * sx) x [i * sy, (i + 1) * sy)
    of the original video, with sx = width / w and sy = height / h.
    """

    feats: torch.Tensor
    height: int
    width: int

    @property
    def grid(self):
        return self.feats.shape[1], self.feats.shape[2]

    @property
    def stride(self):
        h, w = self.grid
        return self.width / w, self.height / h

    def sample(self, t, xy):
        """Bilinearly sample the features of frame t at pixel positions xy (N, 2).

        Returns (N, C) L2-normalised descriptors.
        """
        xy = torch.as_tensor(xy, dtype=torch.float32, device=self.feats.device)
        norm = torch.stack((2 * xy[:, 0] / self.width - 1, 2 * xy[:, 1] / self.height - 1), dim=-1)
        fmap = self.feats[t].permute(2, 0, 1)[None].float()  # (1, C, h, w)
        out = F.grid_sample(fmap, norm[None, None], mode="bilinear", padding_mode="border", align_corners=False)
        return F.normalize(out[0, :, 0].T, dim=-1)

    def similarity(self, t, desc):
        """Cosine similarity between descriptors desc (N, C) and every patch of frame t: (N, h, w)."""
        return torch.einsum("hwc,nc->nhw", self.feats[t].float(), desc)

    def match(self, sim, center_xy=None, radius=None, tau=0.05, refine_radius=1):
        """Locate each point in a similarity map.

        sim: (N, h, w) similarities. If center_xy (N, 2) and radius (in patches)
        are given, the search is restricted to a square window around each
        center (local search); otherwise the whole frame is searched (global
        search). The best patch is refined to sub-patch precision with a
        soft-argmax over the (2 * refine_radius + 1)^2 neighbourhood, with
        temperature tau.

        Returns xy (N, 2) pixel positions and score (N,), the best similarity.
        """
        n, h, w = sim.shape
        dev = sim.device
        gy, gx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")

        if center_xy is not None:
            sx, sy = self.stride
            c = torch.as_tensor(center_xy, dtype=torch.float32, device=dev)
            cx, cy = c[:, 0] / sx - 0.5, c[:, 1] / sy - 0.5
            window = ((gx[None] - cx[:, None, None]).abs() <= radius) & ((gy[None] - cy[:, None, None]).abs() <= radius)
            sim = sim.masked_fill(~window, -2.0)

        flat = sim.reshape(n, -1)
        score, best = flat.max(dim=1)
        by, bx = best // w, best % w

        # Soft-argmax around the best patch
        near = ((gx[None] - bx[:, None, None]).abs() <= refine_radius) & ((gy[None] - by[:, None, None]).abs() <= refine_radius)
        weights = torch.softmax((sim / tau).masked_fill(~near, -1e4).reshape(n, -1), dim=1).reshape(n, h, w)
        ex = (weights * gx[None]).sum(dim=(1, 2))
        ey = (weights * gy[None]).sum(dim=(1, 2))

        sx, sy = self.stride
        xy = torch.stack(((ex + 0.5) * sx, (ey + 0.5) * sy), dim=-1)
        return xy, score


def query_descriptors(fmap, queries):
    """Descriptor of each query (N, 3) = (t, x, y), sampled in its own query frame: (N, C)."""
    query_t = queries[:, 0].astype(int)
    desc = torch.empty((len(queries), fmap.feats.shape[-1]), device=fmap.feats.device)
    for t0 in np.unique(query_t):
        idx = np.flatnonzero(query_t == t0)
        desc[idx] = fmap.sample(t0, queries[idx, 1:])
    return desc


def init_outputs(queries, num_frames):
    """Tracks filled with the query positions and an all-False visibility.

    Frames before a query frame are not scored under the query-first protocol.
    """
    tracks = np.tile(queries[None, :, 1:], (num_frames, 1, 1)).astype(np.float32)
    visible = np.zeros((num_frames, len(queries)), dtype=bool)
    visible[queries[:, 0].astype(int), np.arange(len(queries))] = True
    return tracks, visible


class DinoFeatures:
    """Extract dense DINOv3 patch features of a video, cached on disk.

    Frames are resized so that both sides are multiples of the patch size (16),
    after an optional upscaling by ``scale`` (finer grid, higher cost), and
    normalised with the ImageNet statistics. The last-layer patch tokens (the
    CLS and register tokens are dropped) are L2-normalised and cached in
    ``cache_dir/<model>_s<scale>/<video hash>.npy``. The cache key is a hash of
    the video pixels, so the four DINOv3 baselines share the same files.
    """

    def __init__(self, model_name=DEFAULT_MODEL, scale=1.0, cache_dir="cache/dinov3", batch_size=8, device=None):
        self.model_name = model_name
        self.scale = scale
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir) / f"{model_name.split('/')[-1]}_s{scale:g}"
        self._model = None
        self._last = (None, None)  # in-memory cache of the last video

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        return self._model

    @torch.inference_mode()
    def _compute(self, video):
        model = self._load_model()
        patch = model.config.patch_size
        n_special = 1 + model.config.num_register_tokens
        t, h, w, _ = video.shape
        hr = max(patch, round(h * self.scale / patch) * patch)
        wr = max(patch, round(w * self.scale / patch) * patch)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

        chunks = []
        for start in range(0, t, self.batch_size):
            x = torch.from_numpy(video[start:start + self.batch_size]).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(hr, wr), mode="bilinear", align_corners=False, antialias=True)
            x = (x - mean) / std
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                tokens = model(pixel_values=x).last_hidden_state[:, n_special:]
            feats = tokens.float().reshape(len(x), hr // patch, wr // patch, -1)
            chunks.append(F.normalize(feats, dim=-1).half())
        return torch.cat(chunks)

    def extract(self, video):
        """Return the FeatureMap of video (T, H, W, 3) uint8 RGB."""
        key = hashlib.sha1(np.ascontiguousarray(video).tobytes()).hexdigest()[:16]
        if self._last[0] == key:
            return self._last[1]

        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            feats = torch.from_numpy(np.load(path)).to(self.device)
        else:
            feats = self._compute(video)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, feats.cpu().numpy())

        fmap = FeatureMap(feats=feats, height=video.shape[1], width=video.shape[2])
        self._last = (key, fmap)
        return fmap
