# Paper Summary: VGGT: Visual Geometry Grounded Transformer

- **Paper Title:** VGGT: Visual Geometry Grounded Transformer
- **Authors:** Jianyuan Wang, Minghao Chen, Nikita Karaev, Christian Rupprecht, David Novotny, Andrea Vedaldi (VGG Oxford & Meta AI)
- **Publication:** CVPR 2025 / arXiv:2503.11651
- **Official Resources:** [GitHub (facebookresearch/vggt)](https://github.com/facebookresearch/vggt)
- **Role in AML Project:** Core model under study. Designed natively for **2D Point Tracking** (via built-in CoTracker2 head) and **Feedforward 3D Reconstruction** (cameras, depth maps, point maps for static scenes). Must be adapted for **3D Point Tracking** in Step 3.

---

## 1. Executive Summary & Core Motivation

Classical 3D reconstruction and multi-view geometry rely on iterative optimization pipelines (e.g., COLMAP, Bundle Adjustment), which are computationally slow (taking minutes or hours), sensitive to initialization, and prone to failure when visual overlap is small or textures are repetitive.

Recent learning-based feedforward methods like DUSt3R and MASt3R predict 3D point maps, but are constrained to processing **image pairs** ($N=2$), requiring quadratic pairwise evaluations and costly global alignment for video sequences.

**VGGT is a unified, feedforward foundation transformer for visual geometry.** It ingests an arbitrary sequence of $N$ images (from 1 to hundreds of frames) in a single pass and simultaneously predicts:
1. **Camera parameters** (intrinsics and extrinsics for all frames).
2. **Dense metric depth maps** and **3D point maps**.
3. **Dense tracking features** that drive a built-in long-range 2D point tracking module.

All of this is computed in under a second feed-forward without iterative bundle adjustment.

---

## 2. Problem Formulation & Representations

Given an unordered or ordered sequence of $N$ RGB images $(I_1, I_2, \dots, I_N)$, with $I_i \in \mathbb{R}^{3 \times H \times W}$:
$$f\big((I_i)_{i=1}^N\big) = (g_i, D_i, P_i, T_i)_{i=1}^N$$

Where for each frame $i$:
- **Camera Parameters ($g_i \in \mathbb{R}^9$):**
  $$g_i = [q_i, t_i, f_i]$$
  - Rotation quaternion: $q_i \in \mathbb{R}^4$
  - Translation vector: $t_i \in \mathbb{R}^3$
  - Field of view (intrinsics): $f_i \in \mathbb{R}^2$ (assuming principal point at image center).
- **Dense Depth Map ($D_i \in \mathbb{R}^{H \times W}$):** Metric depth along the camera ray.
- **Dense Point Map ($P_i \in \mathbb{R}^{3 \times H \times W}$):** 3D coordinates of every pixel, expressed in the **coordinate frame of the first camera** $g_1$.
- **Tracking Features ($T_i \in \mathbb{R}^{C \times H \times W}$):** Dense feature grid used by the tracking head.

### The Canonical Coordinate Frame
VGGT designates the first image $I_1$ as the world anchor frame:
- Extrinsics for camera 1 are fixed to identity: $q_1 = [0, 0, 0, 1]$, $t_1 = [0, 0, 0]$.
- All camera poses $g_i$ and point maps $P_i$ are predicted directly in camera 1's coordinate system.
- To achieve this, the network uses special learned camera and register tokens for frame 1 distinct from frames $2 \dots N$.

---

## 3. Architecture & Key Innovations

```
Input Images (I_1 ... I_N)
      │
      ▼
 DINO Patchification ──► Tokens t_I + Camera Tokens t_g + Register Tokens t_R
                              │
                              ▼
            ┌────────────────────────────────────┐
            │  Alternating-Attention Transformer │
            │  (24 layers: Frame-wise + Global)  │
            └────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    Camera Head           DPT Heads         Tracking Head
  (Self-Attn + MLP)     (Conv / Upsample)     (CoTracker2)
          │                   │                   │
   Cameras g_i = [q,t,f]   Depth D_i & Point P_i   2D Tracks &
                           + Uncertainties Σ_i     Visibilities
```

### 3.1. Alternating-Attention (AA) Transformer Backbone
Standard global self-attention across all tokens in all $N$ frames would have quadratic cost $\mathcal{O}((N \cdot K)^2)$, which is prohibitive for long sequences. Standard frame-wise attention cannot communicate 3D geometry across views.

VGGT introduces **Alternating-Attention (AA)** (24 layers):
- **Frame-wise Self-Attention:** Tokens attend only within their respective image $I_i$. Normalizes activations, preserves local spatial context.
- **Global Self-Attention:** Tokens attend across all frames jointly. Establishes visual correspondences and enforces 3D consistency.
- No cross-attention is used; pure self-attention ensures permutation equivariance for frames $2 \dots N$.

### 3.2. Prediction Heads
1. **Camera Head:** Takes refined camera tokens $\hat{t}_i^g$, processes them through 4 self-attention layers and a linear layer to regress $g_i = [q_i, t_i, f_i]$.
2. **Dense DPT Head:** Dense Prediction Transformer (DPT) upsamples image tokens $\hat{t}_i^I$ to dense feature maps $F_i \in \mathbb{R}^{C' \times H \times W}$.
   - $3 \times 3$ conv layers predict depth $D_i$, point maps $P_i$, and tracking features $T_i$.
   - Simultaneously predicts aleatoric uncertainty maps $\Sigma_i^D, \Sigma_i^P \in \mathbb{R}_+^{H \times W}$.
3. **Tracking Head (CoTracker2 Architecture):**
   - Given query point $y_j$ at query frame $I_q$, it bilinearly samples the query feature from $T_q$.
   - Computes correlation maps against all target feature maps $T_i$.
   - Passes correlation features through self-attention layers to predict **2D correspondences $\hat{y}_{j,i} \in \mathbb{R}^2$** and visibility flags.

### 3.3. Training Objective
Trained end-to-end on massive 3D datasets (MegaDepth, ScanNet, Co3Dv2, DL3DV, Replica, ARIA Synthetic, Kubric, etc.):
$$\mathcal{L} = \mathcal{L}_{camera} + \mathcal{L}_{depth} + \mathcal{L}_{pmap} + 0.05 \cdot \mathcal{L}_{track}$$
- $\mathcal{L}_{camera}$: Huber loss on camera parameters.
- $\mathcal{L}_{depth}, \mathcal{L}_{pmap}$: Uncertainty-weighted L1 + gradient loss:
  $$\|\Sigma_i \odot (\hat{D}_i - D_i)\| + \|\Sigma_i \odot (\nabla \hat{D}_i - \nabla D_i)\| - \alpha \log \Sigma_i$$
- $\mathcal{L}_{track}$: L1 tracking loss on 2D coordinates + binary cross-entropy on visibility.

---

## 4. Fundamental Limitations & Flaws (Crucial for Step 1)

While VGGT is remarkably fast and powerful for static multi-view reconstruction, it possesses structural assumptions that create major vulnerabilities in dynamic video tracking:

### 1. The Rigid Static Scene Assumption
VGGT maps all frames to a **single rigid coordinate frame** centered at camera 1. Its training data consists overwhelmingly of static 3D scenes captured from multiple camera angles.
- If an object in the video moves independently (e.g. a car driving, a person walking), VGGT's camera head and point map head assume the motion is caused by camera movement or static scene geometry.
- The predicted point maps $P_i$ for dynamic objects become warped or collapsed.

### 2. Disconnected 2D Tracking and 3D Geometry
In VGGT, **the tracking head outputs only 2D coordinates $\hat{y} \in \mathbb{R}^2$**.
- It does **not** natively output a 3D trajectory through time.
- To obtain a 3D track, one must unproject the 2D point using estimated depth and camera poses. Any error in 2D tracking, depth estimation, or camera pose compounds multiplicatively.

### 3. Non-Rigid Deformations
As stated in the paper's limitations:
> *"Moreover, although our model handles scenes with minor non-rigid motions, it fails in scenarios involving substantial non-rigid deformation."*

---

## 5. Connections to AML Project Steps & Questions

### Connection to Step 1: Subset Selection (Q1, Q2)
- **Q1 (VGGT's Flaw):** VGGT fails when scene dynamics violate rigidity. When an object undergoes independent or non-rigid motion, VGGT's static 3D inductive bias causes severe geometric distortion.
- **Q2 (Qualitative Evidence):** Select clips from TAPVid-3D (**DriveTrack** or **Panoptic Studio**) where objects move independently. Show that while 2D tracking might follow the object, unprojected 3D points fly off into space or stay pinned to the static background frustum.

### Connection to Step 2: 2D Point Tracking (Q3, Q5, Q6)
- **Q3 (How VGGT produces 2D tracks):**
  Pass video frames through VGGT backbone to get tracking features $T_i$. Bilinearly sample the query point feature in query frame $t_q$, correlate with target frames, and run the built-in CoTracker2 tracking head to get 2D trajectories and visibility logits.
- **Performance:** Because CoTracker2 is built into VGGT, VGGT exhibits competitive 2D tracking performance ($AJ \approx 55\text{--}65\%$) on standard 2D benchmarks like TAP-Vid.

### Connection to Step 3: 3D Point Tracking Adaptation (Q7, Q9, Q10)
- **Q7 (How to adapt VGGT to 3D):**
  VGGT does not output 3D trajectories directly. You must adapt it using one of two strategies:
  1. **Self-Contained Adaptation (VGGT Depth + Pose):**
     - Run VGGT tracking head to get 2D track $\hat{p}_t = (\hat{u}_t, \hat{v}_t)$.
     - Sample VGGT's predicted depth map $\hat{D}_t$ at $(\hat{u}_t, \hat{v}_t)$ to get depth $\hat{Z}_t$.
     - Unproject to 3D camera coordinates using VGGT's predicted focal length $f_t$:
       $$\hat{X}_t = \frac{(\hat{u}_t - c_x) \hat{Z}_t}{f_x}, \quad \hat{Y}_t = \frac{(\hat{v}_t - c_y) \hat{Z}_t}{f_y}$$
     - Transform to the common world/anchor frame using predicted extrinsics $(q_t, t_t)$.
  2. **External Depth Adaptation (VGGT 2D + Depth Anything 3):**
     - Use VGGT purely for 2D tracks $\hat{p}_t$.
     - Estimate dense depth using Depth Anything 3.
     - Unproject using camera poses (ground truth or estimated).
- **Q10:** Report $3D\text{-}AJ$, $APD_{3D}$, and $OA$ on your selected subset.

### Connection to Step 4: Extension (Q11 to Q14)
- **Extension Direction 1 (Improving VGGT for 3D Point Tracking):**
  - **Problem (Q11):** VGGT's predicted depth and point maps assume static scenes and suffer from scale inconsistency on dynamic objects.
  - **Implementation (Q12):** Train a lightweight temporal 3D adapter. Keep VGGT frozen. Extract tracking tokens and local features; train a small MLP/transformer with LoRA to predict a dynamic depth residual $\Delta Z_t$ or directly regress 3D coordinates from the tracking features.
  - **Colab Feasibility:** Caching VGGT features $T_i$ avoids running the heavy 24-layer transformer during training, allowing rapid, lightweight fine-tuning on Colab in AMP.

---

## 6. Oral Exam Preparation: Technical Questions & Answers

**Q1: What is the role of Alternating-Attention in VGGT, and why not use standard full self-attention?**
> **Answer:** Full self-attention across all tokens of all $N$ frames scales quadratically with $(N \cdot K)^2$, causing GPU out-of-memory errors for $N > 10$. Frame-wise attention alone cannot model cross-frame geometry. Alternating-Attention interleaves frame-wise attention (which processes spatial structure and normalizes features within each image) with global attention (which cross-references corresponding tokens across different frames), achieving cross-view geometric reasoning with linear complexity in $N$.

**Q2: How does VGGT define its coordinate frame, and why is frame 1 special?**
> **Answer:** VGGT represents all 3D predictions in the coordinate system of the first camera ($g_1 = (I, 0)$). It injects specialized learnable camera and register tokens for frame 1, distinct from frames $2 \dots N$. This breaks permutation equivariance specifically for the first frame, anchoring the 3D reference frame without requiring arbitrary gauge fixing during test time.

**Q3: Why can't VGGT's Point Map ($P_i$) be used directly as a 3D track for a dynamic object?**
> **Answer:** VGGT's point map $P_i(u, v)$ maps pixel $(u, v)$ in frame $i$ to a 3D coordinate in camera 1's space assuming the physical scene is stationary between exposures. If an object moves dynamically, the physical point changes its 3D position relative to the world. VGGT does not model time-dependent deformation fields in $P_i$; it attempts to project the dynamic point as if it were part of a rigid static background.
