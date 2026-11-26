# Step 1: subset selection (Q1, Q2)

**Goal:** show, with official benchmark metrics, one concrete flaw of each model on a subset chosen to expose it, plus at least 3 commented qualitative examples. Everything must reuse the shared code merged in `main` (`src/datasets`, `src/evaluation`, `scripts/evaluate.py`; see `docs/eval_common_spec.md`).

| Part | Owner | Branch | Benchmark |
|---|---|---|---|
| A. VGGT: non-rigid motion | Paul | `exp/vggt_non_rigid` (existing), PR into `main` | TAP-Vid DAVIS (2D) |
| B. D4RT: independent queries / temporal coupling | Mohammadreza | `feat/step1-d4rt` from `main` | TAPVid-3D (3D) |

## What already exists

- `exp/vggt_non_rigid`: VGGT runner (`scripts/run_vggt.py`), window selection and export (`scripts/vggt_inputs.py`), error diagnostics and HTML player (`scripts/vggt_analysis.py`, two notebooks), runs on dance-twirl and paragliding-launch.
- `results/d4rt_analysis.md`: D4RT on WorldTrack (2 sequences, preliminary): sequence-length degradation, spatially concentrated errors, 3 failure examples.
- `scripts/test_*_davis.py`: VGGT vs CoTracker3 on breakdance (Alexis).

## A. VGGT on non-rigid motion

**Claim to test (stated by the VGGT authors):** VGGT handles only minor non-rigid motion and fails under substantial non-rigid deformation.

### Adjustments to `exp/vggt_non_rigid` before its PR

1. **Dataset path:** read `data/tapvid_davis/tapvid_davis/tapvid_davis.pkl`, the path `src/datasets/tapvid.py` uses. The current `tapvid_davis/` folder at the repo root is **not git-ignored**, so the 1 GB pickle could be committed by accident. Remove `tapvid_davis/README.md` and `tapvid_davis/SOURCES.md` (they belong to the dataset archive).
2. **Official metrics:** add `src/tracking/vggt.py`, a `Tracker2D` wrapper around VGGT's tracking head (same frame window for every sequence, as in `run_vggt.py`), and score it with `scripts/evaluate.py --task 2d`. Q2 asks for AJ, <δ_avg and OA; the existing EPE diagnostics stay as qualitative tools.
3. **No absolute paths in saved files:** `summary.json` stores `/home/paul/...`; store paths relative to the repo root.
4. **Commit only final runs:** keep in `results/analysis/` only the runs used in the report, not every timestamped attempt.

### Subset definition

- Two groups of DAVIS sequences, listed in `configs/subsets/davis_nonrigid.txt` and `configs/subsets/davis_rigid.txt`, e.g. non-rigid: dance-twirl, breakdance, parkour, judo; rigid: drift-chicane and other vehicle sequences.
- **Objective criterion** (so the selection is not just visual): a rigidity score from the GT tracks, the variance over time of pairwise distances between visible points of the object. A rigid motion keeps these distances constant. Report the score of every selected sequence.
- **Same protocol for both groups:** same window length and stride, and query-first.

### Evidence

| Table row | Non-rigid subset | Rigid subset |
|---|---|---|
| VGGT | AJ / <δ_avg / OA | AJ / <δ_avg / OA |
| CoTracker3 (same windows) | AJ / <δ_avg / OA | AJ / <δ_avg / OA |

CoTracker3 is the control: if it also drops on the non-rigid subset, the gap comes from sequence difficulty, not from VGGT. Then show 3 qualitative examples (GT vs prediction overlay), each with its own AJ / <δ_avg / OA and a comment linking the failure to the claim.

## B. D4RT on TAPVid-3D

**Hypothesis (our inference, not stated by the authors):** each query is decoded independently, with no temporal coupling, so tracks can jitter or drift over time.

1. **Move from WorldTrack to TAPVid-3D:** a few ADT / PStudio / DriveTrack sequences loaded with `src/datasets/tapvid3d.py`, scored with `evaluate_3d` on **raw** predictions (official median scaling). The earlier WorldTrack numbers were pre-aligned and are not comparable.
2. **Sanity check:** compare with Table 4 of the D4RT paper. A large gap means our findings apply to Open-D4RT, and the report must say so.
3. **Test the hypothesis:** frame-to-frame jitter (second difference of 3D positions) of D4RT vs the jitter of the GT tracks, split into static and moving points. Comparing with the GT is enough to test the claim, so this does not wait for other branches. Add the comparison with CoTracker3 + depth later, once `feat/depth-lifting` is merged.
4. **Decision rule:** keep a claim only if it holds on several sequences; otherwise drop it and say so.
5. 3 qualitative examples with 3D-AJ / APD_3D / OA.

## Definition of done

- Subset files in `configs/subsets/` and the rigidity score per sequence.
- Metrics tables for A and B produced by `scripts/evaluate.py` (CSV in `results/`).
- 3 qualitative examples for A and for B (at least 3 in total, as Q2 requires), each commented with numbers.
- A short write-up (`results/step1_analysis.md`) answering Q1 and Q2 that separates what the papers state from our inference and gives the strength of the evidence.

## Out of scope

The baselines (Step 2), 3D lifting of VGGT (Step 3) and the extension (Step 4).
