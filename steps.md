## Step 1 : Subset selection from the 2 benchmarks

- Read VGGT and D4RT papers and try to think of possible limitation of that approach

- Look at VGGT and D4RT model predictions on some samples of the benchmark

- Try to identify what are the flaws of the two models and where their predictions are mistaken, try to find a relevant problem.

- Find a way of selecting the subset that can represent that flaw that arised from the qualitatives you observed.

---

- Q1. What is the flaw in the predictions you are trying to address with the specific subset
selection? Based on the papers you read, do you find a reason why this particular thing is not
working?
- Q2: I want you to show at least 3 qualitatives of the identified flaw commenting the results
for those specific images and quantifying the flaws with the benchmark metrics

## Step 2 : 2D Point Tracking

### Mandatory baselines

To compare VGGT and D4RT with alternative tracking approaches, implement and evaluate also some baselines. Evaluate the following mandatory baselines on the selected TAP-Vid subset:

- Pyramidal Lucas–Kanade (LK)
- DINOv3 feature-similarity matching
- DINOv3 prototype-based matching
- LK + DINOv3
- DINOv3 + Constant Velocity
- CoTracker3

---

- Q3. How did you use VGGT and D4RT to produce 2D point tracking results?
- Q4: How did you implement each of the mandatory baselines?
- Q5: Show the predictions of each of the baselines, VGGT and D4RT on the same sample chosen from your subset.
- Q6: In a table you should report the average Jaccard (AJ), average position accuracy of visible points, and binary occlusion accuracy (AO) (see Tab. 3 in the benchmark paper [3]) for all the baselines, VGGT and D4RT.

## Step 3 : 3D Point Tracking

D4RT is natively designed for dynamic 3D tracking while VGGT must be adapted to the 3D tracking setting. Since the original 2D tracking formulation alone does not provide a complete depth trajectory, an additional mechanism must be introduced to recover or estimate the depth dimension (e.g. exploiting VGGT's own output space or using an external depth estimator like Depth Anything).

### Mandatory baselines

Compare the proposed methods with the following 3D tracking baselines:

- Optical flow (LK) + depth
- DINOv3 + depth
- DINOv3 + depth + constant velocity
- LK + DINOv3 + depth
- CoTracker3 + depth

> **Note:** Be careful when implementing these baselines because they assume a common reference frame for 3D tracking and therefore you will need camera poses.

---

- Q7: How did you adapt VGGT to produce 3D point tracking results?
- Q8: How did you implement each of the mandatory baselines?
- Q9: Show the predictions of each of the baselines, VGGT and D4RT on the same sample chosen from your subset.
- Q10: Show in a table report the 3D-AJ, the APD and OA metrics results (defined in table 3 of the benchmark) for each baseline, VGGT adaptation and D4RT.

## Step 4 : Extension

After these experiments on the 2D and 3D setting, use the results to identify the main limitations of the two approaches, especially focusing on the VGGT and D4RT models. The proposed extension should be discussed in advance with the TA and should answer a clearly defined experimental question regarding the 2 models.

### Guidelines & Constraints
- A **training-based extension is expected**: You must train or fine-tune at least one component of one of the explored approaches to address an identified limitation.
- Due to the computational constraints of Google Colab, training must be lightweight:
  - Fine-tuning only a small subset of the model's parameters or training a small additional module is strongly recommended (training an entire model from scratch or full fine-tuning is prohibitive).
  - Use Automatic Mixed Precision (AMP) training.
  - Use Low-Rank Adaptation (LoRA).
  - Caching: Cache intermediate results or features of the model to save GPU memory and execution time.

### Choose at least one of the following extension directions:
1. **Improving VGGT for 3D Point Tracking:** VGGT is not natively designed for 3D Point Tracking. Develop a method that improves the use of VGGT for 3D point tracking.
2. **Improving D4RT for 2D Point Tracking:** Investigate whether the dynamic 3D representations learned by D4RT can be converted into a competitive 2D tracking method.
3. **Your own ideas:** Propose your own idea on other models you developed in the baselines (must be well motivated).

---

- Q11: What is the specific problem that your extension is trying to solve?
- Q12: How did you implement your extension idea?
- Q13: To the table of the baselines for the task that you are solving, add a new line for each experiment that you made reporting the same metrics with the same experimental setting for evaluation.
- Q14: Comment the results you obtained.

## Deliverables & Oral Exam Requirements

- **Code:** Deliver the PyTorch code for all project steps, with the notebooks or scripts needed to reproduce your results.
- **Report:** Complete and submit the report describing your results, including the required discussions answering all questions (Q1 to Q14).
- **Oral Exam:**
  - Technical questions on reference papers (VGGT, D4RT, TAP-Vid, TAPVid-3D).
  - If using pre-made methods from libraries, you must be able to explain their underlying functioning.

## Essential Code Links

- **TAP-Vid:** https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/README.md
- **TAP-Vid 3D:** https://tapvid3d.github.io/
- **VGGT:** https://github.com/facebookresearch/vggt
- **OpenD4RT:** https://github.com/Lijiaxin0111/Open-d4rt


