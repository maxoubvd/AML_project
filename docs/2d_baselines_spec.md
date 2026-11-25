# Branch `feat/2d-baselines`: mandatory 2D baselines (Q4)

**Owner:** Maxime · **Base:** `main` · **Merge:** PR into `main`

## Goal

Implement the five non-learned 2D baselines behind the shared `Tracker2D` interface (`src/tracking/base.py`), so that they are scored with `scripts/evaluate.py` exactly like VGGT, D4RT and CoTracker3. Step 3 reuses them unchanged (+ depth), so they must stay pure 2D trackers. CoTracker3 already exists (`src/tracking/cotracker.py`).

## Deliverables

```
src/models/baselines/lk.py               # LKTracker
src/models/baselines/dino_features.py    # DINOv3 feature extraction + disk cache
src/models/baselines/dino_similarity.py  # DinoSimilarityTracker
src/models/baselines/dino_prototype.py   # DinoPrototypeTracker
src/models/baselines/lk_dino.py          # LKDinoTracker
src/models/baselines/dino_cv.py          # DinoConstantVelocityTracker
scripts/evaluate.py                      # tracker registry (see below)
```

**Tracker registry (do this first):** replace the hard-coded `--tracker` choices in `scripts/evaluate.py` with a `TRACKERS = {"static": ..., "cotracker3": ..., "lk": ...}` dict, so other branches only add one line each.

## Method per baseline

All trackers follow the query-first protocol: track forward from the query frame. Predictions before the query frame are not scored.

| Baseline | Position | Visibility |
|---|---|---|
| **LK** | `cv2.calcOpticalFlowPyrLK`, frame-to-frame chaining from the query frame (grayscale, pyramid levels and window size documented) | LK status flag and forward-backward error below a threshold |
| **DINOv3 similarity** | Query descriptor = feature bilinearly sampled at the query point in the query frame. In each frame: cosine similarity with all patches, then argmax refined by a local soft-argmax for sub-patch precision | Max similarity above a threshold |
| **DINOv3 prototype** | Like the similarity baseline, but the descriptor is a prototype: mean of features in a small neighbourhood of the query, updated with confident matches (EMA). Our own definition, so check it with the TA | Similarity to the prototype above a threshold |
| **LK + DINOv3** | LK propagation. When LK fails (status or FB error) or the DINOv3 similarity at the LK position drops, re-localise with a DINOv3 search in a local window | Combined rule (documented) |
| **DINOv3 + constant velocity** | Predict `x_t = x_{t-1} + v`, run the DINOv3 search only in a window around the prediction, then update `v` | Max similarity above a threshold |

## DINOv3 details

- Model: a small DINOv3 ViT from Hugging Face `transformers`. Start with ViT-S/16, which fits in 8 GB and Colab. The weights are **gated**: each user must accept the license on the model page and log in with `huggingface-cli login`.
- Patch size 16. Resize frames to a multiple of 16 and document the resolution.
- **Cache** the features per sequence on disk (`cache/dinov3/<model>/<sequence>.npy`, git-ignored). All four DINOv3 baselines share the cache.

## Rules

- Tune thresholds and window sizes on 2–3 DAVIS sequences **outside** the Step 1 subset, never on the evaluation subset.
- Every hyper-parameter is a constructor argument with a documented default.
- Each method must be explainable at the oral exam: keep a docstring explaining how it works, including how pyramidal LK works.

## Definition of done

1. Every baseline runs with `python -m scripts.evaluate --task 2d --tracker <name>` on `configs/subsets/davis_test.txt`.
2. Results on the full DAVIS set and on the Step 1 subsets are written to `results/2d/<tracker>.csv`.
3. A sanity check: on a static-camera, static-object crop, every baseline beats `static`.
4. A short section in the README listing the hyper-parameters used.

## Out of scope

VGGT, D4RT and CoTracker3 (`feat/2d-models`), the final Q5 figure and Q6 table (`feat/2d-models`), and depth (`feat/depth-lifting`).
