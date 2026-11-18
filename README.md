# AML Project: 2D and 3D Point Tracking using VGGT and D4RT

> **Politecnico di Torino — Advanced Machine Learning (2026–2027)**  
> **Instructor:** Alessandro Marinai  
> **Student Workspace:** `/home/maxime/Desktop/PoliTo/S1/Advanced Machine Learning/Project/AML_project`

---

## 📌 Project Overview
The objective of this project is to study, benchmark, and adapt two foundation vision models for point tracking in video sequences:
- **VGGT (Visual Geometry Grounded Transformer)**: Designed for multi-view geometry and 2D point tracking on static scenes.
- **D4RT (Dynamic 4D Reconstruction & Tracking)**: Designed for feedforward dynamic 3D scene reconstruction and 3D point tracking.

Evaluation is performed on two complementary benchmarks:
- **TAP-Vid** (2D Point Tracking benchmark)
- **TAPVid-3D** (3D Point Tracking benchmark)

---

## 📚 Paper Summaries & Analysis Documents

Each reference paper has a dedicated markdown document detailing its technical formulation, architecture, evaluation metrics, critical flaws, connections to the project questions (Q1–Q14), and oral exam prep:

1. **[01_TAP-Vid_Summary.md](file:///home/maxime/Desktop/PoliTo/S1/Advanced%20Machine%20Learning/Project/AML_project/Papers/01_TAP-Vid_Summary.md)**:
   *Tracking Any Point in a Video* (Doersch et al., NeurIPS 2022).
   Covers 2D point tracking formulation, datasets (Kinetics, DAVIS, Kubric), and metrics ($AJ$, $< \delta_{avg}$, $OA$).

2. **[02_TAPVid-3D_Summary.md](file:///home/maxime/Desktop/PoliTo/S1/Advanced%20Machine%20Learning/Project/AML_project/Papers/02_TAPVid-3D_Summary.md)**:
   *Tracking Any Point in 3D* (Koppula et al., NeurIPS 2024).
   Covers 3D point tracking formulation, datasets (Aria Digital Twin, DriveTrack, Panoptic Studio), depth-adaptive thresholds ($\delta_{3D} = Z \cdot \delta_{2D} / f$), global median rescaling, and 3D metrics ($3D\text{-}AJ$, $APD_{3D}$, $OA$).

3. **[03_VGGT_Summary.md](file:///home/maxime/Desktop/PoliTo/S1/Advanced%20Machine%20Learning/Project/AML_project/Papers/03_VGGT_Summary.md)**:
   *Visual Geometry Grounded Transformer* (Wang et al., CVPR 2025).
   Covers Alternating-Attention transformer backbone, camera anchor coordinate frame ($g_1 = (I, 0)$), DPT heads, built-in CoTracker2 tracking head, and the static-scene assumption limitation.

4. **[04_D4RT_Summary.md](file:///home/maxime/Desktop/PoliTo/S1/Advanced%20Machine%20Learning/Project/AML_project/Papers/04_D4RT_Summary.md)**:
   *Efficiently Reconstructing Dynamic Scenes One D4RT at a Time* (Zhang et al., CVPR 2026).
   Covers universal pointwise querying $q = (u, v, t_{src}, t_{tgt}, t_{cam})$, unified 4D decoding (tracks, depth, point cloud, camera extrinsics/intrinsics), native dynamic scene tracking, and 2D adaptation.

5. **[PROJECT_SYNTHESIS_AND_QUESTIONS_MAP.md](file:///home/maxime/Desktop/PoliTo/S1/Advanced%20Machine%20Learning/Project/AML_project/Papers/PROJECT_SYNTHESIS_AND_QUESTIONS_MAP.md)**:
   Cross-paper comparative analysis, architectural comparison matrix, and a direct mapping to **all project questions (Q1 to Q14)** across Steps 1 to 4.

---

## 🛠️ Project Steps Breakdown

- **Step 1: Subset Selection from the 2 Benchmarks**
  - Identify concrete flaws in VGGT (static scene bias on dynamic/non-rigid objects) and D4RT (sub-pixel 2D jitter / patch lookup sensitivity under occlusion).
  - Justify subset selection and document with 3 qualitative examples and metrics (Q1, Q2).
- **Step 2: 2D Point Tracking**
  - Produce 2D tracks with VGGT (native CoTracker2 head) and D4RT (auxiliary 2D head or 3D perspective projection) (Q3).
  - Implement 6 mandatory 2D baselines: Lucas–Kanade, DINOv3 similarity, DINOv3 prototype, LK+DINOv3, DINOv3+Constant Velocity, CoTracker3 (Q4).
  - Visual qualitative predictions (Q5) and benchmark results table ($AJ$, $< \delta_{avg}$, $OA$) (Q6).
- **Step 3: 3D Point Tracking**
  - Adapt VGGT to 3D tracking (unprojecting 2D tracks via VGGT depth/poses or external Depth Anything 3) (Q7).
  - Native 3D tracking with D4RT.
  - Implement 5 mandatory 3D baselines with depth and camera poses (Q8).
  - Visual qualitative 3D trajectories (Q9) and benchmark results table ($3D\text{-}AJ$, $APD_{3D}$, $OA$) under Global Median Rescaling (Q10).
- **Step 4: Extension (Training-based on Colab)**
  - Choose one direction: Improving VGGT for 3D tracking, improving D4RT for 2D tracking, or a novel motivated idea (Q11).
  - Lightweight training strategy: LoRA, AMP (FP16), feature caching (Q12).
  - Benchmark comparison table (Q13) and discussion of results (Q14).

---

## 🔗 Official Code Repositories
- **TAP-Vid:** https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md
- **TAP-Vid 3D:** https://tapvid3d.github.io/
- **VGGT:** https://github.com/facebookresearch/vggt
- **OpenD4RT:** https://github.com/Lijiaxin0111/Open-d4rt
