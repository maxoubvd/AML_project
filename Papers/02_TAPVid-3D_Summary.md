# Paper Summary: TAPVid-3D: A Benchmark for Tracking Any Point in 3D

- **Paper Title:** TAPVid-3D: A Benchmark for Tracking Any Point in 3D
- **Authors:** Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, João Carreira, Andrew Zisserman, Gabriel Brostow, Carl Doersch (Google DeepMind, UCL, Oxford)
- **Publication:** NeurIPS 2024 / arXiv:2407.05921
- **Official Resources:** [Project Page & Dataset](https://tapvid3d.github.io/)
- **Role in AML Project:** Primary reference benchmark for **Step 3 (3D Point Tracking)**, providing the dataset splits, 3D coordinate definitions, depth scale alignment schemes, and 3D evaluation metrics ($APD_{3D}$, $3D\text{-}AJ$, $OA$).

---

## 1. Executive Summary & Core Motivation

While 2D point tracking (TAP-Vid) tracks pixel coordinates $(u, v)$ over time, physical interaction in robotics, AR/VR, autonomous driving, and dynamic scene reconstruction requires understanding full **3D trajectories $(X, Y, Z)$** of arbitrary surface points in the physical world.

Prior to TAPVid-3D:
- 3D reconstruction benchmarks (e.g., ScanNet, MegaDepth) were strictly **static**, evaluating stationary point clouds or camera poses.
- Dynamic point tracking benchmarks lacked metric 3D ground truth, relied on small synthetic sequences, or only measured 2D re-projections.

**TAPVid-3D introduces the first large-scale real-world benchmark for long-range 3D point tracking (TAP-3D)**, spanning 4,000+ real-world video clips across diverse environments, complex dynamic objects, multi-agent interactions, and camera motions.

---

## 2. Problem Formulation: Tracking Any Point in 3D (TAP-3D)

Given a monocular video sequence $V = (I_1, I_2, \dots, I_T)$ and a query surface point specified at query frame $t_q$ by its 2D coordinates $(u, v)$ (or its initial 3D position):
$$q = (u, v, t_q)$$

The goal is to predict for every frame $t \in \{1, \dots, T\}$:
1. **The 3D point position:** $\hat{P}_t = (\hat{X}_t, \hat{Y}_t, \hat{Z}_t) \in \mathbb{R}^3$, expressed in camera coordinates of frame $t$ (or converted to world coordinates).
2. **The binary visibility/occlusion flag:** $\hat{v}_t \in \{0, 1\}$, where $v_t = 1$ if the point is visible in frame $t$.

---

## 3. Benchmark Dataset Splits & Characteristics

TAPVid-3D is constructed from three distinct real-world data sources with precise 3D annotations:

| Split | Source Dataset | Video Count | Frames / Clip | Resolution / FPS | Scenario Characteristics & Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Aria Digital Twin (ADT)** | Project Aria (Meta) | 1,956 clips (50 minival) | 300 frames | $512 \times 512$ @ 30 FPS | Indoor rooms; wearable sensor motions; humans interacting with furniture, doors, objects; multi-room transitions. |
| **DriveTrack** | Waymo Open Dataset | 2,457 clips (50 minival) | 25–300 frames | $1920 \times 1280$ @ 10 FPS | Autonomous driving; outdoor streets; high-speed vehicle motions, moving pedestrians, dynamic background/foreground disparity, long distance ranges. |
| **Panoptic Studio** | CMU Panoptic | 156 clips (50 minival) | 150 frames | $640 \times 360$ @ 30 FPS | Indoor multi-camera dome; extreme non-rigid human articulation (dancing, gesturing, social interaction), dense occlusions, deformable clothing. |

---

## 4. Evaluation Metrics & The Scale Ambiguity Problem

Evaluating 3D point tracks from monocular video is fundamentally complicated by **monocular scale ambiguity**: a monocular model cannot determine the absolute metric scale of an unseen scene without prior knowledge.

### 4.1. Scale Alignment Schemes
Before computing errors, predictions $\hat{P}_t^i$ are rescaled to match ground truth $P_t^i$:
1. **Global Median Rescaling (Standard Benchmark Default):**
   A single global scale scalar $s$ is computed as the median ratio over all visible points and frames:
   $$s = \text{median}_{i, t, v_t^i = 1} \left( \frac{\|P_t^i\|_2}{\|\hat{P}_t^i\|_2} \right), \quad \hat{P}_t^i \leftarrow s \cdot \hat{P}_t^i$$
2. **Per-Trajectory Rescaling:**
   Rescales each point track $i$ independently using the scale ratio at its query timestamp $t_q$:
   $$s^i = \frac{\|P_{t_q}^i\|_2}{\|\hat{P}_{t_q}^i\|_2}, \quad \hat{P}_t^i \leftarrow s^i \cdot \hat{P}_t^i$$
   (Used when a model is good at estimating individual relative motions but drifts across space/time).
3. **Local Neighborhood Rescaling:** Aligns tracks within local spatial clusters to handle independent multi-object motions.

### 4.2. Depth-Adaptive 3D Distance Threshold $\delta_{3D}$
In 2D tracking, thresholds are fixed pixel radii $\delta_{2D} \in \{1, 2, 4, 8, 16\}$. In 3D, a fixed Euclidean distance (e.g. 10 cm) is unfair because points further away naturally have higher spatial uncertainty under perspective projection.

TAPVid-3D defines the 3D threshold by unprojecting the 2D pixel threshold using ground-truth depth $Z(P_t)$ and camera focal length $f$:
$$\delta_{3D}(P_t) = Z(P_t) \cdot \frac{\delta_{2D}}{f}, \quad \text{for } \delta_{2D} \in \{1, 2, 4, 8, 16\}$$
- Closer points require finer 3D metric accuracy.
- Farther points have proportionally relaxed 3D distance tolerance.
- **Key Mathematical Equivalence:** If depth estimation is perfectly accurate, $APD_{3D}$ becomes numerically identical to 2D position accuracy $< \delta_{avg}$.

### 4.3. The 3D Evaluation Metrics
1. **Average Position Distance ($APD_{3D}$ or $< \delta_{avg}^{3D}$):**
   Evaluates visible points ($v_t^i = 1$):
   $$APD_{3D} = \frac{1}{\sum_{i,t} v_t^i} \sum_{i,t, v_t^i = 1} \mathbb{I}\left( \|\hat{P}_t^i - P_t^i\|_2 < \delta_{3D}(P_t^i) \right)$$
2. **Occlusion Accuracy ($OA$):**
   Classification accuracy of binary visibility:
   $$OA = \frac{1}{\sum_{i,t} 1} \sum_{i,t} \mathbb{I}(\hat{v}_t^i = v_t^i)$$
3. **3D Average Jaccard ($3D\text{-}AJ$):**
   $$\alpha_t^i = \mathbb{I}\left( \|\hat{P}_t^i - P_t^i\|_2 < \delta_{3D}(P_t^i) \right)$$
   $$3D\text{-}AJ = \frac{1}{5} \sum_{\delta_{2D}} \frac{\sum_{i,t} v_t^i \hat{v}_t^i \alpha_t^i}{\sum_{i,t} v_t^i + \sum_{i,t} (1 - v_t^i)\hat{v}_t^i + \sum_{i,t} v_t^i \hat{v}_t^i (1 - \alpha_t^i)}$$

---

## 5. Key Empirical Findings & Failure Modes (Figure 2 Analysis)

The paper investigates standard composite baselines: **2D Tracker + Monocular Depth Estimator** (e.g., BootsTAPIR / CoTracker + ZoeDepth / COLMAP).

Crucial observation: While 2D tracking performance reaches $AJ \approx 50\text{--}65\%$, 3D tracking performance plunges to $3D\text{-}AJ \approx 15\text{--}25\%$. The paper highlights four fundamental failure modes:
1. **Depth Noise:** Monocular depth estimates have high frame-to-frame variance. A small 2-pixel jitter in 2D combined with depth fluctuations creates wildly jagged, physically impossible 3D paths.
2. **Temporal Scale Drift:** Depth estimators process frames independently or over short windows. As the camera moves or zooms, the predicted depth scale drifts over time, causing tracks to artificially expand or shrink along the optical axis.
3. **Spatial Scale Inconsistency:** Monocular depth networks predict depth based on object semantic priors. As a result, a moving foreground car or human is often scaled completely differently from the background road/building. A single global scale factor cannot reconcile both.
4. **Static Scene Priors (COLMAP / SfM failure):** Structure-from-Motion (COLMAP) strictly assumes a static scene. When applied to dynamic objects, SfM fails to triangulate points or treats dynamic motion as erroneous outliers.

---

## 6. Connections to the AML Project Steps & Specific Questions

### Connection to Step 1: Subset Selection (Q1, Q2)
- **Q1 (Identifying Flaws):**
  - **VGGT's Flaw:** VGGT predicts geometry in the reference frame of camera 1 assuming a static scene. In dynamic splits like **DriveTrack** (moving vehicles) or **Panoptic Studio** (deforming humans), VGGT cannot distinguish camera ego-motion from object motion, leading to collapsed or distorted 3D trajectories.
  - **D4RT's Flaw:** D4RT is designed for dynamic scenes, but in sequences with severe occlusions or rapid out-of-frame transitions (like DriveTrack), independent queries without temporal recurrent memory can produce noisy 3D tracks when visual evidence is lost.
- **Q2 (Qualitative & Quantitative Evidence):** Select a subset focusing on either:
  - *Dynamic multi-body motion* (e.g. DriveTrack with moving vehicles).
  - *Non-rigid human body articulation* (Panoptic Studio).
  Show that VGGT 3D-AJ drops sharply on moving objects compared to static background, whereas D4RT handles dynamic motion better but may exhibit depth variance.

### Connection to Step 3: 3D Point Tracking & Baselines (Q7 to Q10)
- **Q7 (Adapting VGGT to 3D Tracking):**
  VGGT outputs 2D tracks $y_{j,i}$ and depth maps $D_i$ + camera extrinsics/intrinsics $g_i$.
  - *Option 1 (Internal):* Sample VGGT's predicted depth $D_i$ at predicted 2D locations $\hat{y}_i$, then unproject into 3D using VGGT's predicted camera intrinsics $f_i$ and transform to world coordinates using extrinsics $(q_i, t_i)$.
  - *Option 2 (External Depth):* Run Depth Anything 2/3 on the frames, sample depth at VGGT's 2D predicted positions, and unproject using camera poses.
- **Q8 (Implementing 3D Baselines):**
  All baselines take a 2D tracker, sample estimated depth maps (e.g. Depth Anything 3), and unproject to 3D camera coordinates:
  - *LK + Depth*
  - *DINOv3 + Depth*
  - *DINOv3 + Depth + Constant Velocity (3D)*
  - *LK + DINOv3 + Depth*
  - *CoTracker3 + Depth*
  > **Note on Reference Frame:** Since evaluation requires a consistent world/reference frame, camera poses (ground truth or estimated) must be used to transform camera-frame 3D coordinates into the common reference frame.
- **Q9 & Q10:** Provide qualitative visualizations of 3D point trajectories and fill the benchmark table with $3D\text{-}AJ$, $APD_{3D}$, and $OA$.

### Connection to Step 4: Extension (Q11 to Q14)
- **Improving VGGT for 3D Tracking:** Train a lightweight temporal depth refinement module or dynamic motion residual adapter (using LoRA on VGGT's DPT/tracking tokens) to correct depth scale drift on dynamic objects.

---

## 7. Oral Exam Preparation: Technical Questions & Answers

**Q1: Why does TAPVid-3D unproject pixel thresholds ($\delta_{3D} = Z \cdot \delta_{2D} / f$) instead of using a constant metric threshold (like 10 cm)?**
> **Answer:** Under perspective projection, spatial resolution is inversely proportional to depth. A 10 cm error at 1 meter depth represents a massive perceptual and angular error (causing a large pixel shift), whereas a 10 cm error at 50 meters depth is sub-pixel and visually indistinguishable. The depth-adaptive threshold preserves scale equivariance with 2D tracking and makes $APD_{3D}$ equal to 2D $< \delta_{avg}$ when depth is exact.

**Q2: What is the purpose of Global Median Rescaling, and when does it fail?**
> **Answer:** Monocular depth estimators predict relative depth up to an unknown metric scale factor $s$. Global median rescaling finds the optimal scale factor by taking the median ratio between ground truth and predicted point norms.
> It fails when the model suffers from **scale drift over time** (scale changes from frame 1 to frame 100) or **spatial inconsistency** (the network scales foreground moving objects differently from background structures).

**Q3: Why can't we simply run COLMAP (SfM) to get 3D tracks on TAPVid-3D?**
> **Answer:** COLMAP relies on epipolar geometry under the strict assumption that the entire scene is rigid and stationary ($x' F x = 0$). In dynamic scenes (vehicles, walking humans), dynamic points violate epipolar constraints and are either rejected as outliers, triangulated into nonsensical 3D coordinates, or corrupt the bundle adjustment camera pose optimization.
