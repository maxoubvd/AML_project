# Paper Summary: TAP-Vid: A Benchmark for Tracking Any Point in a Video

- **Paper Title:** TAP-Vid: A Benchmark for Tracking Any Point in a Video
- **Authors:** Carl Doersch, Ankush Gupta, Larisa Markeeva, Adrià Recasens, Lucas Smaira, Yusuf Aytar, João Carreira, Andrew Zisserman, Yi Yang (DeepMind & VGG Oxford)
- **Publication:** NeurIPS 2022 (Datasets and Benchmarks Track) / arXiv:2211.03726
- **Official Resources:** [GitHub (Google DeepMind TAPNet)](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md)
- **Role in AML Project:** Primary reference benchmark for **Step 2 (2D Point Tracking)** and foundation for evaluation methodology across the entire project.

---

## 1. Executive Summary & Core Motivation

Before TAP-Vid, video motion understanding was largely divided into two extremes:
1. **Short-range, dense motion estimation:** Optical flow (e.g., RAFT, PWC-Net) operates between consecutive frames or across very short windows. When integrated over long trajectories, optical flow suffers from catastrophic error accumulation (drift) and cannot handle occlusions.
2. **Coarse, object-level tracking:** Multiple Object Tracking (MOT) and bounding box tracking follow bounding boxes or coarse object centroids, ignoring non-rigid physical deformations, surface articulation, and fine-grained geometry.

**TAP-Vid formalizes Tracking Any Point (TAP)**: Given an arbitrary physical surface point specified by query coordinates $(x, y)$ in a query frame $t$, follow that physical point throughout the entire video sequence across both visible and occluded intervals.

TAP-Vid provides the first standardized, large-scale, multi-domain benchmark for long-term physical point tracking with ground-truth position, occlusion flags, and rigorous multi-threshold evaluation metrics.

---

## 2. Problem Formulation & Task Definition

Let a video sequence be represented as $V = (I_1, I_2, \dots, I_T)$, where $I_t \in \mathbb{R}^{H \times W \times 3}$ and $T$ is the number of frames.

A query is defined by:
$$q = (x, y, t_q)$$
where $(x, y) \in [0, W-1] \times [0, H-1]$ is the 2D continuous coordinate of a surface point on the query frame $t_q \in \{1, \dots, T\}$.

The tracking model must output for every frame $\tau \in \{1, \dots, T\}$:
1. **Predicted 2D coordinates:** $\hat{p}_\tau = (\hat{x}_\tau, \hat{y}_\tau) \in \mathbb{R}^2$
2. **Predicted occlusion / visibility flag:** $\hat{v}_\tau \in \{0, 1\}$ (or occlusion logit / probability), where $v_\tau = 1$ indicates that the query point is visible and not occluded in frame $\tau$.

### Query Modes
- **First-Frame Mode:** All query points are chosen in frame $t_q = 1$. The tracker only needs to track forward in time.
- **Query-Frame Mode (Any-Frame):** Query points can be initiated at any frame $t_q$ where the point is visible. The tracker must track both **forward** ($\tau > t_q$) and **backward** ($\tau < t_q$) in time.

---

## 3. Dataset Composition & Construction

TAP-Vid consists of both synthetic datasets (perfect ground truth) and real-world datasets (human-verified/semi-automatic ground truth):

| Split / Dataset | Domain | Number of Videos | Ground Truth Source | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **TAP-Vid-DAVIS** | Real | 30 videos (val set) | Denser human annotations on segmentation masks | High resolution, complex non-rigid deformation, challenging occlusions, moving camera. |
| **TAP-Vid-Kinetics** | Real | 1,189 videos (test set) | Flow propagation + strict human verification via AMT | Highly diverse human actions, camera motion, motion blur, partial occlusions, diverse real-world scenes. |
| **TAP-Vid-RGB-Stacking**| Synthetic / Real | 50 videos | Robot simulator / real robot arm | Manipulating colored blocks, severe self-occlusions, contact dynamics. |
| **TAP-Vid-Kubric** | Synthetic | Large-scale train set | Kubric PyBullet simulator with photorealistic ray-tracing | Rigid objects falling/colliding, dynamic cameras, perfect 2D/3D and occlusion labels. Ideal for training. |

---

## 4. Evaluation Metrics (Crucial for Project & Oral Exam)

Videos are resized to a canonical evaluation resolution of $256 \times 256$ pixels. Evaluation uses three primary metrics:

### 1. Occlusion Accuracy ($OA$)
Measures the classification accuracy of the binary occlusion/visibility prediction:
$$OA = \frac{1}{T} \sum_{\tau=1}^T \mathbb{I}(\hat{v}_\tau = v_\tau)$$
where $\mathbb{I}(\cdot)$ is the indicator function.

### 2. Average Position Accuracy for Visible Points ($< \delta_{avg}$)
Evaluates purely the regression accuracy of the spatial coordinates **conditioned on the point being visible in ground truth ($v_\tau = 1$)**:
$$\text{Acc}_\delta = \frac{1}{\sum_{\tau=1}^T v_\tau} \sum_{\tau: v_\tau = 1} \mathbb{I}(\|\hat{p}_\tau - p_\tau\|_2 < \delta)$$
where $\|\cdot\|_2$ is the Euclidean distance in pixel space at $256 \times 256$ resolution.
The metric $< \delta_{avg}$ is the mean across 5 logarithmic thresholds:
$$< \delta_{avg} = \frac{1}{5} \sum_{\delta \in \{1, 2, 4, 8, 16\}} \text{Acc}_\delta$$

### 3. Average Jaccard ($AJ$)
The flagship metric combining position accuracy and occlusion prediction into a single unified score. For each threshold $\delta \in \{1, 2, 4, 8, 16\}$:
- **True Positives ($TP_\delta$):** Ground truth is visible ($v_\tau = 1$), predicted as visible ($\hat{v}_\tau = 1$), and spatial error is within threshold ($\|\hat{p}_\tau - p_\tau\|_2 < \delta$).
- **False Positives ($FP_\delta$):** Predicted visible ($\hat{v}_\tau = 1$), but either ground truth is occluded ($v_\tau = 0$) or spatial error exceeds threshold ($\|\hat{p}_\tau - p_\tau\|_2 \ge \delta$).
- **False Negatives ($FN_\delta$):** Ground truth is visible ($v_\tau = 1$), but either predicted as occluded ($\hat{v}_\tau = 0$) or spatial error exceeds threshold ($\|\hat{p}_\tau - p_\tau\|_2 \ge \delta$).

The Jaccard index at threshold $\delta$ is:
$$\text{Jaccard}_\delta = \frac{TP_\delta}{TP_\delta + FP_\delta + FN_\delta}$$
$$AJ = \frac{1}{5} \sum_{\delta \in \{1, 2, 4, 8, 16\}} \text{Jaccard}_\delta$$

> **Oral Exam Key Insight:** Why is $AJ$ strictly better than $< \delta_{avg}$ or $OA$ alone?
> A trivial model predicting all points as always visible could achieve high $< \delta_{avg}$ if it guesses well when visible, but its $AJ$ is severely penalized by false positives during occlusions. Conversely, a conservative model predicting points as occluded whenever uncertain would be penalized by false negatives. $AJ$ forces the model to balance precise localization and reliable occlusion detection.

---

## 5. Connections to the AML Project Steps & Specific Questions

### Connection to Step 1: Subset Selection (Q1, Q2)
- In Step 1, you must select a representative subset from TAP-Vid that highlights specific failure modes of VGGT and D4RT.
- **Identifiable Flaws in TAP-Vid:**
  1. **Long-term occlusions and re-emergence:** Points that disappear behind obstacles for 20+ frames and reappear. Feedforward models often lose track or latch onto background distractor features.
  2. **Non-rigid deformations & fast motion:** Fast articulated movements (e.g., human dancing in Kinetics or animal motion in DAVIS) challenge models that rely heavily on static geometry or rigid local patches.
  3. **Low-texture / homogeneous surfaces:** Tracking points on uniform walls, roads, or sky where feature matching has ambiguity.
  4. **Out-of-frame exits:** Points leaving the camera frustum and re-entering.
- **Addressing Q1 & Q2:** Selecting a subset characterized by long occlusions or fast non-rigid deformation directly tests whether VGGT's tracking head or D4RT's querying mechanism can maintain identity across frames without spatial drift.

### Connection to Step 2: 2D Point Tracking & Mandatory Baselines (Q3 to Q6)
- **Q3 (How VGGT & D4RT produce 2D tracks):**
  - **VGGT:** Has a native tracking head based on CoTracker2. Tracking features $T_i$ are extracted via DPT, query features are bilinearly sampled at $(x, y)$, cross-frame correlation maps are computed, and self-attention decodes 2D coordinates $\hat{y}$ and visibility logits.
  - **D4RT:** Naturally designed for 3D, but predicts 2D via:
    - *Approach A (Direct Auxiliary Head):* D4RT decoder has a dedicated 2D coordinate prediction head $P_{2D} \in \mathbb{R}^2$ alongside visibility.
    - *Approach B (3D Unprojection + Projection):* Query 3D point $P(t_{tgt}, t_{cam}=t_{tgt}) = (X, Y, Z)$ and project to 2D image coordinates using camera intrinsics: $u = f_x \frac{X}{Z} + c_x, v = f_y \frac{Y}{Z} + c_y$.
- **Q4 (Mandatory 2D Baselines):**
  - *Pyramidal Lucas-Kanade (LK):* Classical optical flow via gradient descent on multi-scale Gaussian pyramids. Suffers from drift and cannot handle occlusions.
  - *DINOv3 Feature-Similarity Matching:* Extract dense DINOv3 ViT features. Find point in frame $t$ by cosine similarity argmax with query feature at $t_q$.
  - *DINOv3 Prototype-Based Matching:* Maintain an exponential moving average (or running prototype) of features over time to adapt to appearance changes.
  - *LK + DINOv3:* Use optical flow for a coarse displacement prior, then refine within a local window using DINOv3 feature similarity.
  - *DINOv3 + Constant Velocity:* Predict location assuming constant 2D velocity $(\hat{p}_t = \hat{p}_{t-1} + (\hat{p}_{t-1} - \hat{p}_{t-2}))$, restrict DINOv3 feature search to a local window around this prediction.
  - *CoTracker3:* State-of-the-art transformer-based sliding-window learned point tracker trained on pseudo-labeled real videos.
- **Q5 & Q6:** Evaluate all baselines + VGGT + D4RT on the chosen subset, reporting $AJ$, $< \delta_{avg}$, and $OA$ in Table format.

### Connection to Step 4: Extension (Q11 to Q14)
- **Improving D4RT for 2D tracking:** D4RT queries 3D points independently without iterative refinement. An extension can add a lightweight 2D refinement head (e.g. cross-attention or local correlation layer fine-tuned with LoRA) to improve sub-pixel 2D localization on TAP-Vid.

---

## 6. Oral Exam Preparation: Technical Questions & Answers

**Q1: How does TAP-Vid handle points that leave the image frame?**
> **Answer:** Points outside the image boundary are labeled as occluded / non-visible ($v_\tau = 0$). Even if their physical position could theoretically be extrapolated, the metric treats them as occluded. If a model predicts them as visible, it incurs false positives.

**Q2: What is the exact difference between $< \delta_{avg}$ and Average Jaccard ($AJ$)?**
> **Answer:** $< \delta_{avg}$ only evaluates frames where ground truth visibility is 1 ($v_\tau = 1$), ignoring any frame where the point is occluded. A tracker that never predicts occlusion can still achieve a perfect $< \delta_{avg}$ if its visible predictions are accurate. $AJ$ penalizes false positive predictions during occlusions and false negative predictions during visible frames, combining localization and occlusion detection into an intersection-over-union metric.

**Q3: Why do classical optical flow methods like Lucas-Kanade perform poorly on TAP-Vid?**
> **Answer:** LK relies on brightness constancy and local gradient descent between adjacent frames. When chained across dozens or hundreds of frames:
> 1. Inherent sub-pixel errors accumulate monotonically (drift).
> 2. When a point is temporarily occluded, LK latches onto the occluder's surface and moves with it, never recovering the original target point when it re-emerges.
