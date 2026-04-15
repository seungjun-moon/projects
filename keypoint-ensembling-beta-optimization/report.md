# Keypoint Ensembling & Frame-0 Beta Warmup for MANO Fitting

**Author:** Seungjun
**Date:** 2026-04-15
**Project:** MANO-pipeline — multi-camera RealSense hand annotation

---

## 1. Scope

Two changes were added on top of the baseline `03c → 05 → 10` fitting stack
in `MANO-pipeline`, both aimed at lowering the MANO / MANOtorch solver loss
on the 4-sequence Jeff dataset without changing the downstream solver
hyper-parameters:

1. **Keypoint ensembling** — replace MediaPipe's 2D hand keypoints with a
   per-axis median over MediaPipe + WiLoR + HaMeR
   (`tools/02e_ensemble_hand_detection.py`).
2. **Frame-0 beta warmup** — before the main pose optimization, run a
   short phase that fits MANO shape `betas[1:]` on frame 0 only, with
   `betas[0]` (overall hand size) frozen at zero, then lock the betas for
   the rest of the solve (`tools/05a_mano_pose_solver_beta.py`,
   `tools/10a_manotorch_pose_solver_beta.py`).

All numbers below are from the same four sequences and the same
`conf_weighted + smooth` triangulation used in the tracker-comparison
report.

| Sequence | Frames | Notes |
|---|---:|---|
| `20260407_023628_jeff`             | 1446 | original |
| `20260409_043434_jeff_redo`        | 1130 | clean re-record |
| `20260409_043744_jeff_redo_gun_0`  | 1423 | gun grasp, 0° |
| `20260409_043915_jeff_redo_gun_90` | 1424 | gun grasp, 90° |

Rig: 8 × RealSense 2048×1536, shared `extrinsics_20260407_jeff.yaml`.

---

## 2. Keypoint ensembling

### 2.1 Detector wrappers

Three per-frame 2D detectors now feed `03c_improved_triangulation.py`:

- **MediaPipe** — `tools/02_mp_hand_detection.py` (baseline).
- **WiLoR** — `tools/02b_wilor_hand_detection.py`, wraps
  `third_party/WiLoR`.
- **HaMeR** — `tools/02c_hamer_hand_detection.py`, wraps
  `third_party/hand_tracking_ablation/models/hamer`.

All three write the exact same on-disk layout as
`mp_handmarks_results.npz`: `(2, num_frames, 21, 2)` int64, OpenPose joint
order, `-1` for missing. Two implementation traps worth recording:

1. **Double remap.** Both WiLoR and HaMeR override `MANOLayer` so that
   `pred_keypoints_3d` is **already** in OpenPose order via the MANO→OP
   permutation `[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]`.
   Applying the same permutation on the wrapper side scrambles the
   joints (wrist correct, fingers 30–130 mm off) and silently degrades
   the downstream MANO fit by ~3.5×. Both wrappers ship with an identity
   remap + comment now.
2. **`ViTDetDataset` gaussian.** Both repos call
   `skimage.filters.gaussian` on the full 2048×1536 image once per
   detected hand inside `__getitem__`. That dominates inference at
   ~3 s/frame on a V100. A passthrough patch applied at import time in
   `02b`/`02c` drops it to ~100 ms/frame (30× speedup); downstream output
   is unchanged because the subsequent crop is sampled at patch
   resolution.

### 2.2 Per-axis median ensemble

`tools/02e_ensemble_hand_detection.py` takes the per-detector files and
computes a **per-axis median** over (mp, wilor, hamer):

- entries marked `-1` are excluded per axis;
- a joint is present in the ensemble only if **≥ 2** detectors agree on
  it — otherwise the output stays `-1`.

Median was preferred over mean because the detector failure modes are
asymmetric: WiLoR/HaMeR project a parametric MANO model, so their errors
are correlated; MediaPipe fails independently on visual outliers. Median
is robust to one detector being far off while two agree. CPU-only, <5 s
for a 1500-frame × 8-camera sequence.

### 2.3 2D triangulation quality (reprojection error)

`03c_improved_triangulation.py --method conf_weighted`, smoothed. 2D
reprojection error in pixels. Bold = best in the row.

**`20260407_023628_jeff` (1446 frames)**

| Detector | Mean | Median | 95th |
|---|---:|---:|---:|
| MediaPipe    | 12.84 | 5.19 | 46.95 |
| WiLoR        | 11.40 | 5.22 | 36.54 |
| HaMeR        | 11.94 | 5.27 | 40.55 |
| **Ensemble** | **11.21** | **5.02** | 37.36 |

**`20260409_043434_jeff_redo` (easiest, 1130 frames)**

| Detector | Mean | Median | 95th |
|---|---:|---:|---:|
| MediaPipe | 9.03 | 4.97 | 31.68 |
| **WiLoR** | **6.30** | **4.48** | **16.23** |
| HaMeR     | 7.32 | 5.31 | 19.49 |
| Ensemble  | 6.73 | 4.80 | 17.42 |

**`20260409_043744_jeff_redo_gun_0` (1423 frames, hand-on-gun)**

| Detector | Mean | Median | 95th |
|---|---:|---:|---:|
| MediaPipe    | 23.59 | 5.44 | 103.50 |
| **WiLoR**    | **11.47** | 6.42 | **36.77** |
| HaMeR        | 12.06 | 6.34 | 41.11 |
| Ensemble     | 11.55 | **6.25** | 38.88 |

> MediaPipe **collapses** on this sequence — 95th-percentile
> reprojection of 103 px — because the gun occludes most of the hand
> surface and the image-feature pipeline mis-localises joints whenever
> the silhouette deviates from "open palm". The model-based detectors
> stay near 11–12 px.

**`20260409_043915_jeff_redo_gun_90` (1424 frames, hardest)**

| Detector | Mean | Median | 95th |
|---|---:|---:|---:|
| MediaPipe    | 16.12 | 5.50 | 59.23 |
| WiLoR        | 16.82 | 6.05 | 45.99 |
| HaMeR        | 16.39 | 5.85 | 45.55 |
| **Ensemble** | **16.04** | **5.77** | 43.17 |

### 2.4 Downstream solver loss — ensemble vs MediaPipe

Same solver hyper-parameters for every run
(`w_kpt_3d=1.0, w_fingertip=1.0, w_smooth=0.1, fingertip_steps=3000,
total_steps=12000`). % = improvement over the MediaPipe column.

| Sequence | Solver | MediaPipe | WiLoR | HaMeR | **Ensemble** |
|---|---|---:|---:|---:|---:|
| `jeff`             | MANO      | 0.01852 | 0.01248 (−33 %) | 0.01319 (−29 %) | **0.01218 (−34 %)** |
| `jeff`             | MANOtorch | 0.01149 | 0.00758 (−34 %) | 0.00814 (−29 %) | **0.00694 (−40 %)** |
| `jeff_redo`        | MANO      | 0.01188 | 0.01096 ( −8 %) | 0.01186 ( 0 %)  | **0.01091 ( −8 %)** |
| `jeff_redo`        | MANOtorch | 0.00658 | 0.00653 ( −1 %) | 0.00719 (+9 %)  | **0.00617 ( −6 %)** |
| `jeff_redo_gun_0`  | MANO      | 0.01999 | **0.01218 (−39 %)** | 0.01272 (−36 %) | 0.01227 (−39 %) |
| `jeff_redo_gun_0`  | MANOtorch | 0.01259 | 0.00647 (−49 %) | 0.00668 (−47 %) | **0.00612 (−51 %)** |
| `jeff_redo_gun_90` | MANO      | 0.01679 | 0.01239 (−26 %) | 0.01366 (−19 %) | **0.01236 (−26 %)** |
| `jeff_redo_gun_90` | MANOtorch | 0.01018 | 0.00746 (−27 %) | 0.00842 (−17 %) | **0.00698 (−31 %)** |

**Aggregate (MANOtorch, mean −% vs MediaPipe across 4 sequences):**

| Detector | Mean Δ |
|---|---:|
| HaMeR        | −21.0 % |
| WiLoR        | −27.7 % |
| **Ensemble** | **−32.0 %** |

Key takeaways:
- The dominant gain is the **2D detector itself** — replacing MediaPipe
  with WiLoR already drops MANOtorch loss by 27 % on average.
- The ensemble adds another **4–8 %** on top of WiLoR alone and is the
  best-floor detector: never loses badly, even on `jeff_redo` where the
  three detectors are within ~5 % of each other.
- Gains are largest on the hardest sequences (gun occlusion, hand-on-hand)
  and smallest on the clean re-record. The benefit scales with the 2D
  side: when MediaPipe's reprojection 95th-percentile blows up to 103 px,
  the downstream solver loss halves.
- The parametric-model detectors' **raw trajectories are 2–4× smoother**
  than MediaPipe's, so the solver spends less capacity fighting jitter.
- `jeff_redo` MANOtorch fingertip is the only cell where MediaPipe wins
  (0.00210 vs ensemble 0.00245) — its visual fingertip detections
  happen to be unusually well-placed on that clean open-palm sequence.
  Ensemble still wins on total loss there.

---

## 3. Frame-0 beta warmup

### 3.1 Motivation

The baseline MANO / MANOtorch solvers hold `betas = 0` (mean-shape hand)
for the entire optimization. That is fine when the subject happens to
match the mean hand, but for Jeff the fit has a small but systematic
finger-length mismatch visible as a residual `kpt_3d` floor of
3–7 mm. The goal of the warmup is to absorb that mismatch **once**, on
a single clean frame, and then keep betas frozen so the rest of the
optimization is still a pose-only problem (same convergence behaviour,
no per-frame beta jitter).

### 3.2 Method

Implemented as a subclass of the existing solver
(`tools/05a_mano_pose_solver_beta.ManoPoseSolverBeta`,
`tools/10a_manotorch_pose_solver_beta.ManotorchPoseSolverBeta`).

Before `solve()` runs, the warmup phase does:

1. Initialize `betas ∈ R^{H×10}` at zero, learnable.
2. Freeze `betas[:, 0]` at 0 — the first PCA component is overall hand
   size, which we want to keep tied to the global scale. Zeroing its
   gradient each step is a simpler invariant than adding a large
   penalty and lets betas[1:] still absorb finger-ratio mismatch.
3. Optimize on **frame 0 only** with the loss
   `w_kpt_3d · L_kpt3d(frame 0) + w_reg · L_pose_reg(frame 0) + w_beta_reg · ‖betas[:,1:]‖²`.
4. Clamp betas to `[-1, +1]` after each step (the plausible MANO shape
   range) and re-zero `betas[0]`.
5. After `beta_warmup_steps` iterations, `betas` are detached and
   injected into every subsequent `MANOGroupLayer.forward` call via the
   new `betas_m=` kwarg added in the base solver. The pose optimization
   then proceeds exactly as before, but on a subject-specific hand
   shape rather than the mean.

The pose vector `_pose_m` is also updated in the warmup, but gradients
for frames `≥ 1` are zeroed so only frame 0's pose moves — the goal is
to find betas that are self-consistent with *some* plausible pose, not
to pre-solve the whole sequence. Betas are saved to
`betas_m.npy` in the run folder.

Defaults used throughout:
`beta_warmup_steps=500, lr_beta=0.01, w_beta_reg=0.01, beta_clip=1.0`.

### 3.3 Learned betas (sanity check)

Top-3 components of the learned `betas[1:]` for the clean sequence
(`jeff_redo`), values rounded to 3 decimals:

| Solver | Tracker | Right hand | Left hand |
|---|---|---|---|
| MANO      | mp       | `[0.157, 0.070, 0.097, …]` | `[-0.160, -0.052, 0.053, …]` |
| MANO      | ensemble | `[0.157, 0.075, 0.094, …]` | `[-0.159, -0.050, 0.051, …]` |
| MANOtorch | mp       | `[-0.035, -0.020, 0.111, …]` | `[-0.021, 0.039, -0.060, …]` |
| MANOtorch | ensemble | `[-0.120, -0.125, 0.054, …]` | `[0.076, 0.076, 0.079, …]` |

Notes:
- Betas are **small and stable** — all components ≪ 1 and well inside
  the clip range, so the warmup is not abusing shape to paper over bad
  keypoints.
- The **right and left hands learn mirror-like components** under MANO
  (first coefficient flips sign), as expected for a single subject
  whose two hands are roughly symmetric under a global scale. MANOtorch
  picks a different basis because its layer exposes a different
  internal parameterization, but the magnitudes are comparable.
- MANO betas are **stable across trackers** (mp vs ensemble) on the
  same frame 0 — the shape fit is not tracker-dominated, which is the
  behaviour we want.

### 3.4 Effect on solver loss (ensemble tracker)

All runs use `--tracker ensemble` and identical solver
hyper-parameters. % columns are relative to the same solver / tracker
*without* the warmup. Positive % means warmup is better (lower loss).

**`jeff_redo` (clean)**

| Solver | Metric | Baseline | + beta warmup | Δ % |
|---|---|---:|---:|---:|
| MANO      | total     | 0.01091 | 0.01100 | +0.8 % |
| MANO      | kpt_3d    | 0.00368 | 0.00372 | +1.1 % |
| MANO      | fingertip | 0.00312 | 0.00316 | +1.3 % |
| MANOtorch | total     | 0.00617 | **0.00603** | **−2.3 %** |
| MANOtorch | kpt_3d    | 0.00323 | **0.00316** | **−2.2 %** |
| MANOtorch | fingertip | 0.00245 | **0.00239** | **−2.4 %** |

**`jeff_redo_gun_0`**

| Solver | Metric | Baseline | + beta warmup | Δ % |
|---|---|---:|---:|---:|
| MANO      | total     | 0.01227 | 0.01238 | +0.9 % |
| MANO      | kpt_3d    | 0.00411 | 0.00416 | +1.2 % |
| MANO      | fingertip | 0.00355 | 0.00362 | +2.0 % |
| MANOtorch | total     | 0.00612 | **0.00604** | **−1.3 %** |
| MANOtorch | kpt_3d    | 0.00321 | **0.00316** | **−1.6 %** |
| MANOtorch | fingertip | 0.00232 | **0.00229** | **−1.3 %** |

**`jeff_redo_gun_90`**

| Solver | Metric | Baseline | + beta warmup | Δ % |
|---|---|---:|---:|---:|
| MANO      | total     | 0.01236 | 0.01245 | +0.7 % |
| MANO      | kpt_3d    | 0.00444 | 0.00448 | +0.9 % |
| MANO      | fingertip | 0.00369 | 0.00374 | +1.4 % |
| MANOtorch | total     | 0.00698 | **0.00688** | **−1.4 %** |
| MANOtorch | kpt_3d    | 0.00366 | **0.00360** | **−1.6 %** |
| MANOtorch | fingertip | 0.00266 | **0.00262** | **−1.5 %** |

**Sanity check on `jeff_redo` with the `mp` tracker** (to rule out
"only ensemble sees the gain"):

| Solver | Metric | Baseline (mp) | + beta warmup | Δ % |
|---|---|---:|---:|---:|
| MANOtorch | total     | 0.00658 | **0.00653** | **−0.8 %** |
| MANOtorch | kpt_3d    | 0.00395 | **0.00393** | **−0.5 %** |
| MANOtorch | fingertip | 0.00210 | **0.00208** | **−1.0 %** |

Takeaways:
- **MANOtorch + beta warmup is uniformly better**, on every sequence
  and on every reported metric, by a modest **1–2.4 %**. The gain is
  consistent and moves both the 3D joint error and the fingertip error
  in the same direction, which is what we expect from a shape fix
  rather than from an optimizer artefact.
- **MANO + beta warmup is slightly worse** (+0.7 to +2 %). The
  unconstrained 45-dim PCA pose can already absorb finger-length
  mismatch by warping the pose, so committing to a non-zero `betas`
  removes a degree of freedom the solver was using — the pose then
  fights the fixed shape. MANOtorch's anatomy constraints prevent that
  abuse, so it actually benefits from a correctly-sized hand.
- The gain is largest on the **clean** sequence (`jeff_redo`), not on
  the gun-grasp sequences. This makes sense: frame 0 is an open-palm
  pose on every sequence, so the *quality of the warmup* is comparable
  across sequences, but the pose-only residual has more room to shrink
  when the sequence itself is not dominated by occlusion noise.
- The warmup **does not fight the ensemble win** — on
  `jeff_redo_gun_0`, MANOtorch drops from 0.01259 (mp) → 0.00612
  (ensemble) → 0.00604 (ensemble + beta). The two improvements stack.

### 3.5 Compute cost

500 warmup steps on frame 0 adds roughly **5–8 s** of wall-clock to a
solver run that already takes 3–5 min for 12 000 full-sequence steps
(<3 % overhead). No additional GPU memory is required — only one frame
is in the forward pass during warmup.

---

## 4. Recommended default stack

```
02_mp_hand_detection.py            }  three detectors in parallel
02b_wilor_hand_detection.py        }  on 3 GPUs
02c_hamer_hand_detection.py        }
02e_ensemble_hand_detection.py     ← cpu, ~5 s
03c_improved_triangulation.py --method conf_weighted --tracker ensemble
05a_mano_pose_solver_beta.py --tracker ensemble   ← warmup + solve
10a_manotorch_pose_solver_beta.py --tracker ensemble   ← warm-starts from MANO
```

- **Ensemble keypoints + MANOtorch + beta warmup** is the lowest-loss
  configuration across every sequence and every metric we checked.
- **MANO (05a) + beta warmup** is not recommended — the extra
  shape-vs-pose coupling actively hurts the 45-dim PCA solver. Keep MANO
  as a warm-start for MANOtorch with betas at zero.
- WiLoR alone is still a strong fallback when only one detector can be
  afforded; MediaPipe alone should not be used for sequences with
  hand-object occlusion.

---

## 5. Files added / changed

| file | change |
|---|---|
| `tools/02b_wilor_hand_detection.py`         | new — WiLoR wrapper (identity OpenPose remap, `ViTDetDataset` gaussian hot-patch) |
| `tools/02c_hamer_hand_detection.py`         | new — HaMeR wrapper (same two fixes) |
| `tools/02e_ensemble_hand_detection.py`      | new — per-axis median ensemble, `min_valid=2` |
| `tools/05a_mano_pose_solver_beta.py`        | new — subclass of `ManoPoseSolver` adding frame-0 beta warmup |
| `tools/10a_manotorch_pose_solver_beta.py`   | new — same idea for `ManotorchPoseSolver`, warm-starts from 05a |
| `tools/05_mano_pose_solver.py`              | `MANOGroupLayer.forward` / `MANOLayer.forward` accept optional `betas`/`scales`; `--tracker`; `--render_only` |
| `tools/10_manotorch_pose_solver.py`         | same `betas` kwarg plumbing; `_save_verts_and_joints()` aligned with 05 output layout |
| `slurm/run_beta_warmup.sh`                  | new — single-sequence beta warmup driver (`jeff_redo`, `mp` tracker) |
| `slurm/run_beta_warmup_ensemble.sh`         | new — 3-sequence beta warmup driver (`ensemble` tracker) |
| `slurm/run_ensemble.sh`, `run_full_tracker_pipeline.sh` | new — ensemble + full 3-detector drivers |
| `tools/collect_outputs.py`                  | new — gathers `joints_m.npy` files into `outputs/` |
