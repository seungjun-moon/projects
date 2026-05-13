# Scale-Consistent HaWoR on Pointbridge: Calibration, Noise, and the H5 Build

**Author:** Seungjun
**Date:** 2026-05-13
**Project:** 3D Hand Pose/Shape Estimation Ablation Study → Pointbridge data pipeline

---

## Overview

This report covers how we applied the **scale-consistent HaWoR** variant
(`HAWORStatic`, trained as `hawor_static_warmstart_h200_step20k`) to the
Pointbridge data pipeline, replacing beomjun's vanilla-HaWoR run that powered
the original `0427_pnp_merged` h5. Two concrete advantages drove the swap:

1. **Cleaner depth calibration.** The affine map from stereo depth to
   HaWoR-predicted depth is much closer to the identity. Per-frame
   scale moves from **0.78 → 0.92** and shift from **+13.3 cm → +4.3 cm**
   (Model G, RANSAC 10%, 2 cm threshold; same 720p calibration video, same
   stereo depth source). That is, the model's depth output already sits near
   the metric ground truth — the affine correction is small, so any
   per-frame residual gets blown up much less when we apply
   `z_corrected = (z − shift) / scale` along the camera ray.
2. **Less per-frame jitter on the wrist trajectory.** Across all
   502 PnP episodes of the production sweep, the mean wrist-Z noise amplitude
   (>1 Hz STFT content in a 32-frame window, the standard "real motion is
   ≤1 Hz at 30 fps" decomposition) drops from **12.5 mm → 9.6 mm**, an
   average **1.32×** quieter signal with ours winning **497 / 502 episodes
   (99.0%)**. On the canonical showcase episode (`teddy_ep35`) the gap is
   **1.86×** (14.3 → 7.7 mm) and visually obvious in the 2D overlay.

Both effects compound at finetune time: shape-consistent meshes give the
calibration step a tighter target, and a quieter wrist trajectory means
the `joints_worldspace` we copy into the h5 is closer to a real arm motion
even before any low-pass post-processing.

---

## 1. The Setup

The Pointbridge pipeline is unchanged from `CLAUDE.md` §"End-to-end extraction":

```
Phase 0   YOLO bbox        →  model_tracks.npy   (HaWoR-style square crop)
Phase 1   HaMeR betas      →  betas.npz          (per-stream 10-dim shape)
Phase 2   HaWoR inference  →  world_space_res.pth (21 MANO joints in cam space)
Calib     depth_pairs      →  Model G (scale, shift)
H5 build  hawor → joints_worldspace, proprio[:, :3], with depth correction
H5 v2     init/dest ramp   →  rlwrld_final_initdest_FINAL.h5  (Q=16, 502 eps)
```

The only thing that changes between the **original** build (beomjun's HaWoR
on his bbox) and the **scale-consistent** build (`hawor_static_warmstart_h200_step20k_b0_full`)
is the **Phase 2 inference model**:

- **Original HaWoR** regresses MANO `betas` per frame from local appearance.
  The 10-dim shape wobbles frame to frame, which inflates wrist-translation
  estimates and forces the depth calibration to absorb a ~20% scale and
  ~13 cm shift.
- **`HAWORStatic` (scale-consistent)** takes `betas` as an **input**, locked
  to a single per-stream HaMeR-averaged vector. Shape is stable across the
  stream, so the model's depth prediction is no longer fighting a wobbling
  hand size, and the calibration affine collapses to roughly the identity.

The architecture and warm-start procedure for `HAWORStatic` are documented in
[../models-vid-hawor-static/report.md](../models-vid-hawor-static/report.md);
this report focuses on the **deployment** of that model into the Pointbridge
data pipeline and the resulting downstream quality.

---

## 2. Depth Calibration: Closer to Identity

We re-ran beomjun's calibration script (`scripts/calibrate_depth.py`, a
path-rewritten copy of his `extract_depth_pairs.py`) on the same 720p
calibration video (`ep0_calibrate_seunghoon`, 984 frames) and the same stereo
depth source, swapping only the HaWoR predictions. Both sides see exactly
the same per-frame projected pixel of the five 1st-knuckles
`[1, 5, 9, 13, 17]` and the same 25×25 stereo-depth patch median lookup.

### 2.1 Model G headline numbers

| Variant | scale | shift (m) | distance from (1, 0) | inlier RMSE | N pairs |
|---|---:|---:|---:|---:|---:|
| **HaWoR (original)** | 0.7766 | +0.1332 | 0.265 | 1.12 cm | 4 299 |
| **HaWoR scale-consistent, β=0** | **0.9184** | **+0.0434** | **0.092** | **1.06 cm** | 4 309 |

(Model G = RANSAC, 10% min-inliers, 2 cm residual threshold — the same
production-pick from beomjun's `720right_v2/depth_pairs.npz`.)

The "distance from (1, 0)" column is the Euclidean distance of `(scale,
shift_in_meters)` from the identity calibration; ours is **2.9× closer**
to the no-op affine.

![Calibration scatter — pairs and Model G fit, ours vs original HaWoR](assets/calibration_scatter.png)

Reading the scatter:

- The **red line** (ours fit) lies almost on the dotted **y = x** ideal across the entire 0.30 – 0.55 m depth range that the calibration video covers. The remaining tilt is the ~8% under-prediction at far range that any monocular hand model still has.
- The **black dashed line** (original fit) intersects y = x near 0.6 m but diverges sharply on both ends — predicting hands too close at far range and too far at close range. That curvature is what the (0.78, +0.13) affine has to absorb.
- **Both scatters have similar inlier RMSE** (~1.1 cm). So the win isn't that the original model is noisier *per pair* — it's that the model's depth is already on the right scale, so a single global affine no longer has to swing the predictions by 20% to match metric ground truth.

### 2.2 Stability across the beta sweep

`HAWORStatic` is shape-conditioned, so we can also probe how calibration
behaves as we deliberately shift the input `betas` by a fixed offset. Six
variants `(b0, b±0.5, b±1, b−2)` and a degenerate `b10` were calibrated
identically:

![Calibration scale / shift / RMSE across the beta sweep — ours stays in a tight band near (1, 0); original HaWoR is far](assets/calibration_sweep_bars.png)

| TAG | scale | shift (m) | RMSE (cm) |
|---|---:|---:|---:|
| **β=0** | 0.9184 | 0.0434 | 1.063 |
| β=+0.5 | 0.9111 | 0.0382 | 1.046 |
| β=+1 | 0.9076 | 0.0323 | 1.029 |
| β=−0.5 | 0.9200 | 0.0493 | 1.090 |
| β=−1 | 0.9352 | 0.0526 | 1.094 |
| β=−2 | 0.9686 | 0.0582 | 1.119 |
| HaWoR (original) | 0.7766 | 0.1332 | 1.115 |

Every reasonable `β` choice keeps `scale ∈ [0.91, 0.97]` and `shift ≤ 6 cm`;
the original HaWoR is alone at `(0.78, +0.13)`. Larger β-offsets in the
negative direction (β = −2 → scale 0.969) trade a little extra residual
RMSE for an even closer-to-identity fit. We picked **β=0** (the HaMeR mean
per stream) for the production `_b0_full` h5 build because it's the most
defensible default and has the lowest distance from `(1, 0)` of the small-β
runs.

---

## 3. Per-Frame Wrist Noise

The depth calibration is a global metric correction; it does not fix
per-frame jitter. For that we measure the wrist-Z high-frequency content
in a 32-frame STFT window and read off "RMS amplitude above 1 Hz" in mm.
Real picking motion at 30 fps is below ~1 Hz, so anything above is
mostly sensor / model jitter (cutoff settled after sweeping 0.3 – 5 Hz; see
`CLAUDE.md` "Canonical 'ours is less jittery'" section).

### 3.1 Showcase: `teddy_ep35`

The canonical demo from CLAUDE.md, 101 frames, full PnP cycle. Ours runs
quiet through the reach, the grasp, the transport, and the place; the
original wrist trace oscillates throughout.

![Wrist-Z noise amplitude per frame — ours vs HaWoR (original), teddy ep35](assets/noise_per_frame_teddy_ep35.png)

| Metric (teddy ep35) | HaWoR original | HaWoR scale-consistent |
|---|---:|---:|
| Mean noise amplitude (mm) | 14.30 | **7.69** |
| Improvement | — | **1.86×** |

The accompanying mesh-size jitter (which `HAWORStatic` *eliminates* by
construction — shape is locked) is visible in the companion plot:

![Hand-size per frame — original flickers, ours is essentially flat](assets/hand_size_per_frame_teddy_ep35.png)

The flat red trace is the "echo, don't regress" property of
`HAWORStatic` materializing in the data: pred_shape = input_betas, so
hand size is fixed at the per-stream value while pose and camera adapt
freely. This is the per-frame mechanism behind the calibration gain in §2.

### 3.2 Sweep over all 502 PnP episodes

Repeating the same wrist-Z STFT statistic on every episode in the
production h5:

| Aggregate (502 eps, all PnP except playing-card) | HaWoR original | HaWoR scale-consistent |
|---|---:|---:|
| Mean noise amplitude (mm) | 12.47 | **9.57** |
| Mean orig/ours ratio | — | **1.322×** |
| Median orig/ours ratio | — | **1.304×** |
| Per-ep win rate (ours quieter) | — | **497 / 502 = 99.0%** |
| Mean absolute gap (orig − ours) | — | 2.91 mm |

So even the *worst* episodes for ours are within hair of the original, and
on average we're shaving 3 mm of wrist-Z jitter off every single frame. The
showcase episode is the headline (1.86×), but the median-case behaviour
(~1.30×, ours wins 99/100) is what actually shows up in trained policy
quality.

### 3.3 2D overlay (qualitative)

The 21-MANO-joint reprojection on the source video tells the same story
visually. **Solid colored** skeleton + box = ours; **black dashed** = beomjun's HaWoR.

![Side-by-side 2D overlay, teddy ep35 (ours solid, beomjun's HaWoR dashed black)](assets/viz_teddy_ep35.gif)

The dashed black skeleton hunts around the actual hand position; the solid
overlay tracks the wrist cleanly through the grasp, transport, and place
without the per-frame breathing that the original exhibits.

---

## 4. The H5 Build (`*_b0_full/rlwrld_final_initdest_FINAL.h5`)

The h5 used for the production finetune sweep is built by
`scripts/build_h5_hawor_FINAL.py` and follows the v2 schema documented in
`CLAUDE.md` §"Per-beta FINAL h5 build":

- **502 episodes** (PnP minus playing-card),
- **Length-preserving** (T_h5 == T_video, not T+2),
- **Q = 16 init/dest ramp** via `step_post_initdest_interp.py`:
  `(Q-i)/Q · ROBOT_INIT_6PTS + i/Q · raw[i]` at both ends,
- Depth calibration applied along the camera ray with the variant's full-precision
  `(scale, shift) = (0.91841, 0.04336)` from `calibration_hawor_bneg0/depth_pairs.npz`,
- Top-level attrs `depth_calibration_{applied,scale,shift}`, `hawor_variant`,
  `interp_Q` written so future runs can detect double-application.

Build command:

```bash
/sjw_alinlab2/home/beomjun/miniconda3/envs/point-bridge/bin/python \
  scripts/build_h5_hawor_FINAL.py \
  --src_h5  datasets/h5/original/rlwrld_final_initdest_FINAL.h5 \
  --dst_h5  datasets/h5/hawor_static_warmstart_h200_step20k_b0_full/rlwrld_final_initdest_FINAL.h5 \
  --world_root datasets/hawor_static_warmstart_h200_step20k_b0_full \
  --scale 0.91841 --shift 0.04336 \
  --variant_tag hawor_static_warmstart_h200_step20k_b0
```

The only fields touched relative to the source h5 are
`joints_worldspace`, `right/joints_worldspace`, and `right/proprio[:, :3]`
(wrist xyz only). `object_points_3d` and everything else are copied verbatim
because they come from a different stereo depth source, not from HaWoR.

---

## 5. Reproducing

The full sequence for one episode is:

```bash
# Phase 0 — YOLO bbox (matches beomjun's IoU ≥ 0.9)
/rlwrld3/home/seungjun/hand_tracking_ablation/.venv/bin/python \
  scripts/run_yolo_track.py --video <VID> --out <OUT>/tracks_0_*

# Phase 1 — HaMeR per-stream betas
/rlwrld3/home/seungjun/hand_tracking_ablation/.venv/bin/python \
  scripts/run_hamer_shape.py --video <VID> --tracks <OUT>/tracks_*/model_tracks.npy \
  --out <OUT>/betas.npz

# Phase 2 — hawor_static inference
/rlwrld3/home/seungjun/hand_tracking_ablation/.venv/bin/python \
  scripts/step0_run_hawor_static.py --video <VID> --betas <OUT>/betas.npz \
  --tracks <OUT>/tracks_*/model_tracks.npy --out <OUT>/world_space_res.pth

# Calibration (reuses ep0_calibrate_seunghoon.mp4 + stereo depth)
/sjw_alinlab2/home/beomjun/miniconda3/envs/point-bridge/bin/python \
  scripts/calibrate_depth.py --calib_dir datasets/calibration_hawor_bneg0 \
  --video_name ep0_calibrate_seunghoon --intrinsics assets/intrinsics_720p_right.json \
  --depth_npy datasets/calibration_hawor/ep0_calibrate_seunghoon_depth.npy
```

End-to-end driver scripts (`run.sh`, `run_hawor_ours.sh`) handle the full
703-episode pipeline; the variant tags `_b0_full`, `_b0p5_full`, …,
`_bneg2_full` correspond to the rows in §2.2.

---

## 6. Summary

- **Calibration**: scale-consistent HaWoR pulls the depth-calibration
  affine from **(0.78, +0.13)** to **(0.92, +0.04)** — 2.9× closer to the
  identity transform, with comparable per-pair RMSE. The model's metric
  depth is already correct on first inference; the affine is now a small
  correction, not a 20% rescale.
- **Per-frame noise**: wrist-Z high-frequency amplitude drops on
  **99% of episodes**, averaging **1.32×** quieter (median 1.30×, showcase
  1.86×). Driven by the static-shape head: locking MANO `betas` to a per-stream
  HaMeR mean eliminates the per-frame mesh breathing that drives the
  wrist-translation jitter.
- **Production h5**: `hawor_static_warmstart_h200_step20k_b0_full/rlwrld_final_initdest_FINAL.h5`
  (502 eps, v2 schema with Q=16 init/dest ramp, calibration constants
  `(0.91841, 0.04336)`) is the artefact that downstream finetuning runs
  consume in place of `0427_pnp_merged/rlwrld_final.h5`.
- A six-variant beta sweep confirms the calibration win is robust to the
  shape-input choice; every variant lands within `(scale ∈ [0.91, 0.97],
  shift ≤ 6 cm)` of the identity, vs. the original's `(0.78, 0.13)`.

---

## Key Files

| File | Description |
|---|---|
| `scripts/step0_run_hawor_static.py` | Phase 2 inference: takes per-stream betas + tracks + video, emits `world_space_res.pth` (right hand, cam space) |
| `scripts/build_h5_hawor_FINAL.py` | v2 h5 builder; applies depth calibration along the camera ray and writes the Q=16 init/dest ramp |
| `scripts/calibrate_depth.py` | Path-rewritten copy of beomjun's `extract_depth_pairs.py`; reads per-variant depth_pairs.npz and fits Model A–K |
| `scripts/plot_noise_per_frame.py` | Canonical wrist-Z noise plot — ours vs original, STFT-based |
| `scripts/plot_hand_size_per_frame.py` | Companion plot — palm-to-fingertip length per frame; flat for ours, jittery for original |
| `scripts/visualize_world_space_res.py` | 21-joint + bbox overlay mp4; solid = ours, dashed-black = beomjun's HaWoR |
| `datasets/h5/hawor_static_warmstart_h200_step20k_b0_full/rlwrld_final_initdest_FINAL.h5` | Production h5 for finetune (502 eps, v2 schema) |
| `datasets/calibration_hawor_bneg0/depth_pairs.npz` | β=0 calibration data, Model G: `(scale=0.91841, shift=0.04336, RMSE=1.06 cm)` |
