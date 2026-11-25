# Work plan: branches and specs

Each branch starts from `main`, follows its spec, and merges back through a PR. Shared code (loaders, metrics, tracker interface) is already in `main`; see `eval_common_spec.md`.

| Branch | Spec | Questions | Owner | Depends on |
|---|---|---|---|---|
| `feat/eval-common` (merged) | [eval_common_spec.md](eval_common_spec.md) | — | Mohammadreza | — |
| `exp/vggt_non_rigid` | [step1_subset_spec.md](step1_subset_spec.md), part A | Q1, Q2 | Paul | — |
| `feat/step1-d4rt` | [step1_subset_spec.md](step1_subset_spec.md), part B | Q1, Q2 | Mohammadreza | — |
| `feat/2d-baselines` | [2d_baselines_spec.md](2d_baselines_spec.md) | Q4 | Maxime | — |
| `feat/2d-models` | [2d_models_spec.md](2d_models_spec.md) | Q3, Q5, Q6 | TBD | `feat/2d-baselines` (for Q5/Q6 only) |
| `feat/depth-lifting` | [depth_lifting_spec.md](depth_lifting_spec.md) | Q7 | TBD | — |
| `feat/3d-tracking` | [3d_tracking_spec.md](3d_tracking_spec.md) | Q8, Q9, Q10 | TBD | 2d-baselines, 2d-models, depth-lifting |
| `feat/extension` | to be written after the TA validates the idea | Q11–Q14 | team | Step 2 and Step 3 tables |

## Rules for every branch

- Code goes in `src/`, experiments in `notebooks/<branch topic>/`. Nobody edits the root `notebook.ipynb` in parallel.
- Every result in a table comes from `scripts/evaluate.py`, so every row uses the same metric code.
- Merge `main` into your branch regularly. Keep PRs small enough to review.
- Datasets, weights and caches stay out of git (`data/`, `cache/`, `*.npz`, `*.pt`).
