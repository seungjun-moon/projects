# HaMeR-RlhndV1, PALM, Bbox Unification, and Video-Dataset Migration

**Author:** Seungjun
**Date:** 2026-04-24
**Project:** 3D Hand Pose/Shape Estimation Ablation Study

---

## Overview

This report covers four parallel updates to `hand_tracking_ablation` landed in the past week:

1. **HaMeR-RlhndV1 variant** — a relaxed-constraint IK that preserves the thumb byte-identical and removes all anatomy-angle clamps, replacing the MANOTorch hard-clamp logic for label preprocessing.
2. **IK accuracy comparison** — rlhndv1 vs. MANOTorch, measured as joint-position deviation from GT MANO on 100 samples per dataset.
3. **PALM dataset** — a new in-domain training set recommended by Sejune; 90,356 samples across 91 tar shards, mixed in at weight 0.10.
4. **Unified bbox range via `DATASETS.RESCALE_FACTOR`** — a single config knob now rescales tar/webdataset bboxes to match HaWoR's cropping convention (clip/video datasets already use `rescale=1.2`). Hypothesis: aligning HaMeR and HaWoR crops improves HaWoR training.
5. **ClipDataset → VideoDataset migration** — haptic per-frame JPEGs (21M files) replaced by per-sequence mp4 + `.frames.npy`, decoded via torchcodec. Saves NFS file-handle pressure and boosts GPU utilization.

---

## 1. RlhndV1 IK: Relaxed Constraints

### 1.1 Motivation

MANOTorch's anatomy-aligned Euler decomposition (see the [manotorch-ik-optimization report](../manotorch-ik-optimization/report.md)) clamps every joint to biomechanical ranges (e.g., MCP twist ∈ [−30°, +30°], bend ∈ [−10°, +100°]). In practice these clamps are too tight: real dataset labels routinely contain thumb poses and MCP spreads that fall outside the clamp window, forcing the IK to rewrite the GT and introducing label noise the network has to un-learn.

RlhndV1 (C variant) takes a looser stance:

- **Preserve the thumb chain entirely.** Joints 13/14/15 (Thumb CMC/MCP/IP) are frozen during IK — their rotations survive byte-identical.
- **Drop every anatomy clamp.** No twist/spread/bend ranges; only axis *selection* (zero inactive axes on 1-DoF joints).
- **Narrower IK search space.** IK moves only 20 DoFs (4 non-thumb MCPs × 3 + 8 non-thumb PIP/DIP × 1) vs. MANOTorch's 26 DoFs.

### 1.2 DoF Structure

| Joint group | MANO indices | RlhndV1 (C) | MANOTorch |
|---|---|---|---|
| Global orient | 0 | Frozen | Frozen |
| Non-thumb MCP | 1, 4, 7, 10 | 3-DoF, no clamp | 3-DoF, clamped |
| Non-thumb PIP | 2, 5, 8, 11 | 1-DoF bend, no clamp | 1-DoF bend, clamped |
| Non-thumb DIP | 3, 6, 9, 12 | 1-DoF bend, no clamp | 1-DoF bend, clamped |
| Thumb CMC | 13 | **Preserved** (no IK) | 3-DoF, clamped |
| Thumb MCP | 14 | **Preserved** (no IK) | 2-DoF, clamped |
| Thumb IP | 15 | **Preserved** (no IK) | 1-DoF, clamped |
| **Active IK DoFs** | | **20** | **26** |

Implementation: `models/hamer_rlhndv1/rlhnd_ik_wrapper.py:43-66` (DoF split + active mask). The head (`mano_head.py`) predicts 7 × 3-DoF 6D rotmats (non-thumb MCPs + thumb CMC/MCP/IP) plus 8 × 1-DoF z-bend scalars — matching the stored labels exactly.

### 1.3 Conversion Pipeline

All tar shards and haptic labels were re-projected onto the rlhndv1 DoF structure at build time, mirroring the earlier MANOTorch pipeline:

| Script | Target |
|---|---|
| `scripts/convert_tars_rlhndv1.py` | Webdataset tar shards (training + eval) |
| `scripts/convert_haptic_labels_rlhndv1.py` | Haptic clip labels (`_DATA/haptic_training_label_rlhndv1/`) |
| `scripts/convert_npz_rlhndv1.py` | MoCap npz (FreiHAND discriminator data) |

Output layout: `_DATA/hamer_training_data/dataset_tars_rlhndv1/` and `_DATA/haptic_training_label_rlhndv1/<dataset>/`. SLURM array launchers live under `slurm/l40s/convert_*_rlhndv1*.sh`.

---

## 2. IK Accuracy: RlhndV1 vs MANOTorch

Measured as per-joint 3D deviation (21 joints including fingertips) between the GT MANO forward and the IK-projected pose, in millimeters. **100 samples per dataset** (20 clips × 5 frames). Computed with `scripts/stats_rlhndv1_ik_error.py`. Left hands run through MANO_RIGHT via axis-angle y/z mirror (preserves joint distances).

### 2.1 Per-Dataset

| Dataset | N | MANOTorch mean | MANOTorch p95 | MANOTorch max | RlhndV1 mean | RlhndV1 p95 | RlhndV1 max |
|---|---:|---:|---:|---:|---:|---:|---:|
| arctic_resize | 100 | 2.36 | 9.54 | 45.52 | **1.06** | **4.51** | **19.83** |
| dexycb | 100 | 1.64 | 6.59 | 29.00 | **0.71** | **3.04** | **6.30** |
| ho2o | 100 | 5.66 | 35.18 | 90.00 | **0.24** | **1.45** | **7.16** |
| ho3d | 100 | 0.53 | 2.25 | 15.01 | **0.19** | **1.21** | **2.40** |
| interhand | 100 | 5.53 | 34.37 | 114.67 | **0.71** | **2.86** | **7.43** |
| hot3d | 100 | 9.38 | 47.93 | 106.85 | **2.16** | **10.36** | 64.05 |
| **Overall** | **600** | **4.17** | — | 114.67 | **0.84** | — | 64.05 |

### 2.2 Overall Percentile Summary

Aggregated over the 500-sample run excluding arctic_resize (which was run separately):

| Percentile | MANOTorch (mm) | RlhndV1 (mm) | Ratio (mt/rl) |
|---|---:|---:|---:|
| p50 | 0.44 | 0.02 | 22× |
| p90 | 11.91 | 1.84 | 6.5× |
| p95 | 27.26 | 2.94 | 9.3× |
| p99 | 65.49 | 10.35 | 6.3× |
| p99.9 | 101.52 | 45.94 | 2.2× |

### 2.3 Reading

- **RlhndV1 reconstructs GT far more faithfully** — mean error drops by **~5×** (4.55 → 0.80 mm), p95 by **~9×** (27.26 → 2.94 mm).
- The gap is largest on **hand-object datasets (ho2o, interhand)** where grasp poses routinely violate MANOTorch's MCP twist/spread clamps.
- **HO3D** is small either way — its labels were already near-anatomical.
- **HOT3D tails** remain heavy even for rlhndv1 (max 64 mm). Root cause likely lives in the label projection step, not in the IK itself — worth a follow-up.
- **Interpretation**: MANOTorch's anatomy clamps are not a free regularizer. On noisy real-hand labels they overwrite the GT and leak noise into the supervision target.

### 2.4 Training Integration

`DATASETS_CONFIG: datasets_tar_rlhndv1.yaml` points the training pipeline at the pre-projected shards. The experiment config is `models/configs_hydra/experiment/hamer_rlhndv1.yaml`; SLURM launcher is `slurm/l40s/hamer_rlhndv1.sh`. A zoom-in variant `hamer_rlhndv1_2.yaml` (RESCALE_FACTOR=0.6) is also queued.

---

## 3. PALM Dataset

New in-domain training corpus recommended by Sejune. Integrated into both the standard and rlhndv1 tar pipelines.

| Attribute | Value |
|---|---|
| Samples | **90,356** |
| Tar shards | **91** (`{000000..000090}.tar`) |
| Location (standard) | `_DATA/hamer_training_data/dataset_tars/palm-train/` |
| Location (rlhndv1) | `_DATA/hamer_training_data/dataset_tars_rlhndv1/palm-train/` |
| Sampling weight (mix_all.yaml) | **0.10** |

Mix-weight rebalancing in `models/configs_hydra/data/mix_all.yaml` to make room without changing total weight:

| Dataset | Old weight | New weight |
|---|---:|---:|
| ARCTIC-TRAIN | 0.10 | 0.05 |
| COCOW-TRAIN | 0.10 | 0.05 |
| **PALM-TRAIN** | — | **0.10** |

No dedicated conversion script — PALM tars ship through the generic `convert_tars_rlhndv1.py` pipeline for the rlhndv1 variant and are consumed directly by the standard HaMeR training path.

---

## 4. Unified Bbox Range: `DATASETS.RESCALE_FACTOR`

### 4.1 Motivation

The two data pipelines historically disagreed on bbox sizing:

- **Tar pipeline (HaMeR/image):** `bbox_size = expand_to_aspect_ratio(scale * 200).max()` — no rescale; tight around the keypoint extent.
- **Clip/Video pipeline (HaWoR):** `bbox_size = rescale_factor * scale.max() * 200` with `rescale_factor = 1.2` default — ~20% context padding around the hand.

**This change unifies the two under a single knob.** When HaMeR and HaWoR train on mixed batches (e.g., `hawor_mt_hamer_mt`), the frame-level crops now follow the same convention. The conjecture — supported by the preliminary mixed-training runs — is that aligning HaMeR's cropping to HaWoR's (slightly more context, consistent across pipelines) should improve HaWoR performance, since HaWoR was the variant whose crop style the temporal model was trained on.

### 4.2 Implementation

Added `DATASETS.RESCALE_FACTOR` (default `1.0`) to the tar-webdataset path. See `models/datasets/image_dataset.py:347`:

```python
rescale_factor = cfg.DATASETS.get('RESCALE_FACTOR', 1.0)
# ...
bbox_size = rescale_factor * expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
```

Values < 1 zoom in on the hand; values > 1 add context. The default (1.0) preserves pre-existing behavior.

### 4.3 Zoom-Out-Only Augmentation Gate

Stacking scale-augmentation's zoom-in on top of a pre-shrunk RESCALE_FACTOR routinely crops fingertips. Added a gate in `models/datasets/utils.py:42` and `image_dataset.py:420`:

```python
# image_dataset.py:418-420
# Skip zoom-in scale-aug when bbox is already pre-shrunk via RESCALE_FACTOR<0.9;
# stacking another zoom-in factor often crops fingertips.
zoom_out_only_scale_aug = rescale_factor < 0.9
```

When enabled, `do_augmentation` restricts scale aug to `[1, 1 + SCALE_FACTOR]` (pure zoom-out; no additional zoom-in).

### 4.4 Experiment Settings

| Experiment | RESCALE_FACTOR | Effective behavior |
|---|---:|---|
| `hamer_rlhndv1.yaml` | 0.8 | 20% zoom-in (aligns with HaWoR context when combined with label scale) |
| `hamer_rlhndv1_2.yaml` | 0.6 | 40% zoom-in (tighter — experimental) |
| Other HaMeR variants | 1.0 (default) | Unchanged |
| HaWoR (clip/video) | 1.2 (built-in) | Unchanged |

---

## 5. ClipDataset → VideoDataset Migration

### 5.1 Motivation

The haptic training pipeline originally stored 21M per-frame JPEGs across 5 datasets. Each sample opens 16 files on NFS — an I/O pattern that saturated the filesystem and bottlenecked training at ~4× slower than the GPU. The `hawor_mt_hamer_mt` job (job 47410, now at 212K steps / 112h) lived with this bottleneck; transient NFS failures also caused intermittent crashes requiring PIL-truncated-image fallbacks.

### 5.2 Design

Replace per-frame JPEGs with per-sequence mp4 + a frame-name→index sidecar:

```
_DATA/haptic_training_videos/<dataset>/<seq>/<clip>.mp4
_DATA/haptic_training_videos/<dataset>/<seq>/<clip>.frames.npy
```

- **1 file open per sample** (not 16).
- **Frame map**: `.frames.npy` maps the existing `imgname` field to a frame index — no label schema changes needed.
- **Decoder**: torchcodec (FFmpeg-backed), fresh decoder per sample + bounded LRU on the frame map. Decord was tried first but leaked /dev/shm buffers on full-res frames (arctic 2800×2000, hot3d 1408×1408), OOM-killing DataLoader workers.

See `models_clip/datasets/video_dataset.py` and `scripts/convert_images_to_video.py`.

### 5.3 Scope of Migration

All `TYPE: ClipDataset` entries in `models_clip/configs/datasets_clip_manotorch.yaml` flipped to `TYPE: VideoDataset`:

| Dataset | Status |
|---|---|
| ARCTIC-Ego train/val | VideoDataset |
| ARCTIC-Exo train/val | VideoDataset |
| ARCTIC-Ego-Resize / Exo-Resize (half-res, 30-frame trim) | VideoDataset |
| DexYCB train | VideoDataset |
| HO3D-Clip train | VideoDataset |
| H2O-Clip train | VideoDataset |
| InterHand26M-Clip train | VideoDataset |
| **HOT3D-Clip train (new)** | VideoDataset |

HOT3D is a fresh addition — 187 Aria recordings converted via `scripts/convert_hot3d_clipdataset.py` + `aggregate_hot3d_index.py`; SLURM array launchers under `slurm/cpu/hot3d_*.sh`.

### 5.4 API Parity

`VideoDataset` and `ClipDataset` share the same constructor signature (`cfg, dataset_file, label_dir, img_dir|video_dir, train, rescale_factor, ...`) and return identical sample dicts (`imgname, center, scale, hand_pose, betas, keypoints_2d/3d, ...`). Downstream `_prepare_batch` needs no changes.

### 5.5 Side Benefits

- **DexYCB left-hand fix**: clip labels un-flipped in place so `left_hand_preflipped` can be deprecated (commit `3d0a7c5`). MANO FK now reconstructs both hands at 0.00 mm mean error.
- **VideoDataset cTw → global_orient parity fix**: shared helper `apply_cTw_to_global_orient()` in `models_clip/datasets/utils.py` applied in both ClipDataset and VideoDataset `_parse_label`. Previously only ClipDataset did this; VideoDataset was feeding world-frame GO to the model, producing noisy `loss_global_orient`.
- **arctic_resize**: half-resolution re-encode with first/last 30 frames trimmed. Useful for fitting arctic on L40S memory and dodging the tails where the capture rig comes into view.

### 5.6 What's Pending

- GPU-side benchmark under realistic multi-worker contention (CPU benchmark was inconclusive — 0.4 vs. 0.8 samples/s single-threaded).
- All-intra (`-g 1`) encode experiment to cut keyframe-seek overhead.
- Hand-region crop-then-encode to shrink video dimensions further on arctic/hot3d.

---

## Key Files

| File | Description |
|---|---|
| `models/hamer_rlhndv1/rlhnd_ik_wrapper.py` | RlhndV1 IK (C variant) — 20 DoF, no clamps, thumb preserved |
| `models/hamer_rlhndv1/rlhnd_wrapper.py` | RLHNDv1 MANO wrapper |
| `models/hamer_rlhndv1/mano_head.py` | Head: 7 × 3-DoF 6D + 8 × 1-DoF bend |
| `scripts/stats_rlhndv1_ik_error.py` | Per-dataset GT vs IK deviation stats |
| `scripts/vis_rlhndv1_vs_manotorch.py` | Side-by-side visualization |
| `scripts/convert_tars_rlhndv1.py` | Tar shard re-projection |
| `scripts/convert_haptic_labels_rlhndv1.py` | Haptic clip label re-projection |
| `models/configs/datasets_tar_rlhndv1.yaml` | rlhndv1 tar dataset config (inc. PALM-TRAIN) |
| `models/configs_hydra/experiment/hamer_rlhndv1.yaml` | RlhndV1 training experiment |
| `models/configs_hydra/experiment/hamer_rlhndv1_2.yaml` | Tighter-crop variant (RESCALE_FACTOR=0.6) |
| `models/configs_hydra/data/mix_all.yaml` | Dataset mix with PALM@0.10 |
| `models/datasets/image_dataset.py` | Tar webdataset + RESCALE_FACTOR wire-up |
| `models/datasets/utils.py` | `zoom_out_only_scale_aug` gate |
| `models_clip/datasets/video_dataset.py` | VideoDataset (torchcodec + mp4 + frames.npy) |
| `scripts/convert_images_to_video.py` | JPEGs → mp4 converter |
| `models_clip/configs/datasets_clip_manotorch.yaml` | All clip entries flipped to VideoDataset |
