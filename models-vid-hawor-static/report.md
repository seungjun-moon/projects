# `models_vid`: Trimmed Video Stack and `HAWORStatic` for Egocentric Hand Tracking

**Author:** Seungjun
**Date:** 2026-05-06
**Project:** 3D Hand Pose/Shape Estimation Ablation Study

---

## Overview

`models_vid/` is a focused fork of `models_clip/`, kept to just the two model
variants we are still actively training video on top of:

1. **`HAWOR`** — the canonical HaWoR architecture (ViT backbone + space-time
   attention + motion module + MANO head), restored to the
   "DexYCB-style" batch convention (`gt_cam_full_pose`, `gt_cam_betas`,
   `gt_cam_j2d`, `do_flip`, real per-frame `img_focal`) instead of the
   `_prepare_batch` adapter that `models_clip` had grown.
2. **`HAWORStatic`** — a new `HAWOR` subclass that **takes MANO shape
   (`betas`) as an input** rather than regressing it. Pose, camera, and
   keypoints are still predicted; shape is held to a per-stream value supplied
   by the caller. This gives stable, plausible egocentric video tracking, where
   per-frame shape regression is the most flicker-prone component of the
   prediction.

Everything else — `hamer*`, `wilor*`, `hawor_manotorch*`, `hawor_rlhndv1*`,
their associated configs and data YAMLs, all the obsolete SLURM launchers, and
both `clip_dataset.py` / `hawor_dataset.py` — has been pruned. The pruned
items still live in `models_clip/` for the experiments that depend on them;
`models_vid/` is the new home for video work going forward.

---

## 1. Folder-Level Differences

| `models_clip/` | `models_vid/` | Notes |
|---|---|---|
| `hamer/`, `hamer_manotorch_fix_shape_euler/`, `hawor_manotorch_fix_shape_euler/`, `hawor_rlhndv1*/` (×3) | — | dropped; image- and clip-only variants stay in `models_clip/` |
| `hawor/` (`heads/`, `mano_torch_wrapper.py`, `rotation.py`, …) | `hawor/` (`modules.py`, `track_dataset.py`, `imutils.py`, `utils/`, `constants.py`) | flatter layout; `heads/modules.py` collapsed to top-level `modules.py`; pulls in HaWoR-public `track_dataset.py` + `imutils.py` for inference; drops MANOTorch glue (no longer needed at the video layer) |
| — | `hawor_static/` (`hawor_static.py`, `modules.py`) | new shape-conditioned head + subclass |
| `datasets/{clip_dataset, hawor_dataset, image_dataset, json_dataset, mocap_dataset, utils, video_dataset}.py` | `datasets/{image_dataset, mocap_dataset, utils, video_dataset}.py` | `clip_dataset.py`, `hawor_dataset.py`, `json_dataset.py` removed — VideoDataset is now the only training data path |
| `configs/datasets_clip.yaml`, `datasets_clip_manotorch.yaml`, `datasets_clip_rlhndv1.yaml`, `datasets_eval.yaml`, `datasets_hawor_rlhndv1.yaml` | `configs/datasets_vid.yaml` (single file) | one unified dataset YAML for video training/eval |
| `configs_hydra/experiment/*.yaml` (10 files) | `configs_hydra/experiment/{default, hawor, hawor_static, hawor_static_v2}.yaml` | one experiment per variant, plus a warm-start variant |
| `configs_hydra/data/mix_all.yaml` | `mix_all.yaml` + new `mix_quest.yaml` | adds Quest-only dataset mix |

The deletions are easy to verify against the current `git status` for the
working tree, where the corresponding `slurm/`, `run_vid_*`, and
`scripts/train_vid_static.py` files are also being retired in favour of the
single `scripts/train_vid.py` entrypoint.

---

## 2. `HAWOR` (`models_vid/hawor/hawor.py`) vs `models_clip/hawor/hawor.py`

The video HaWoR is closer to the **public HaWoR** convention than the
`models_clip` version. The clip variant adapted incoming standard
`ClipDataset` batches via `_prepare_batch()`, supplying a constant focal
length, reading `box_center` / `box_size`, etc. The video variant skips this
adapter entirely:

| Concern | `models_clip/hawor/hawor.py` | `models_vid/hawor/hawor.py` |
|---|---|---|
| Batch adapter | `_prepare_batch()` builds `center, scale, img_focal, img_center` from `box_center / box_size / img_size` and a constant `cfg.EXTRA.FOCAL_LENGTH` | None — `VideoDataset` directly emits `center, scale, img_focal, img_center` per frame |
| Per-frame focal | constant (`EXTRA.FOCAL_LENGTH`, typ. 5000) | **real** per-frame focal from intrinsics (mean of fx, fy), with `√(W²+H²)` fallback |
| GT loss inputs | `keypoints_2d`, `keypoints_3d`, `mano_params` | `gt_cam_j2d`, `gt_j3d_wo_trans`, `gt_cam_full_pose`, `gt_cam_betas` (HaWoR public layout) |
| Left-hand handling | flipped pre-emptively in `ClipDataset` (`left_hand_preflipped`) | `do_flip` flag travels with the sample; predictions un-flipped post-forward via `do_flip` mask before exposing `pred_cam_full` / `trans_full` |
| Robustness knobs | none beyond the GAN-variant `iteration` counter | `pred_cam` scale floor at `1e-3` to stop `s ≈ 0` from blowing up `tz_full = 2·f / (b·s)` and tainting all DDP ranks; warm-up LR ramp (`WARMUP_STEPS`); `error_if_nonfinite=False` clip so transient fp16 inf grads get absorbed by `GradScaler`; `train/grad_scale` logged for steady-state diagnosis |
| `MANOTransformerDecoderHead` location | `hawor/heads/modules.py` | `hawor/modules.py` (flatter) |
| Inference | reuses the `models_clip` patched `inference` that pulls a `TrackDatasetEval` from `lib.datasets.track_dataset` | bundles `track_dataset.TrackDatasetEval` and `imutils` directly under `models_vid/hawor/`, so video inference is self-contained — no dependency on the legacy `lib/` tree |

The `forward_step` shape is otherwise the same: ViT backbone → (optional)
space-time attention → MANO head → `(pred_pose, pred_shape, pred_cam)` →
`mano.query` → (optional) motion module on pose → 2D projection.

---

## 3. `HAWORStatic` (new): shape as an input, not an output

### 3.1 Why

For egocentric video, MANO **shape** (the 10-dim PCA `betas`) is by definition
constant across the stream — the hand belongs to one wearer. Per-frame shape
regression nonetheless wobbles frame-to-frame (because the head sees only
local appearance) and that wobble:

- **distorts mesh size** → contact / depth estimates drift,
- **leaks into pose** → fingers compensate for a too-small or too-large palm,
- **kills temporal consistency** → meshes flicker even when joints are stable.

`HAWORStatic` accepts the per-stream shape as a control input. This decouples
shape from per-frame appearance noise and gives a markedly more **stable and
plausible** mesh under egocentric conditions, while leaving pose and camera
free to adapt frame-by-frame.

### 3.2 What it changes

`models_vid/hawor_static/hawor_static.py` is a **48-line subclass** of
`HAWOR`. It overrides exactly two things:

1. The MANO head is swapped for `MANOTransformerDecoderHeadStaticShape`
   (`models_vid/hawor_static/modules.py`).
2. `forward_step` reads `batch['gt_cam_betas']` and forwards it into the new
   head.

Everything else — backbone, `st_module`, `motion_module`, `mano` wrapper,
optimizer, `compute_loss`, `tensorboard_logging`, projection / un-flip logic
— is inherited unchanged.

### 3.3 The static-shape head

```python
# Original head:
#   pred_pose  = decpose(token_out) + init_hand_pose
#   pred_cam   = deccam(token_out)  + init_cam
#   pred_shape = decshape(token_out) + init_betas

# Static-shape head:
token_out  = transformer(query=zeros, context=ViT_features)   # (B, dim)
shape_feat = encshape(input_betas)                            # (B, dim)
fused      = token_out + shape_feat                           # (B, dim)
pred_pose  = decpose(fused) + init_hand_pose
pred_cam   = deccam(fused)  + init_cam
pred_shape = input_betas                                      # echoed
```

Three architectural notes:

1. **`encshape` replaces `decshape`.** A 2-layer MLP `Linear(10 → 1024) →
   GELU → Linear(1024 → 1024)` projects the 10-dim betas into the decoder's
   1024-dim representation space, then sums into the cross-attended token
   before pose / camera readout. Sum-fusion (rather than concat) keeps
   `decpose` / `deccam` shape-compatible with the original head.
2. **Small-gain Xavier init** on `encshape` (`gain=0.01`, bias `0`). At step 0
   the static-shape contribution is ~0, so the model behaves like the source
   HaWoR checkpoint and learns to use shape over training.
3. **Echo, don't regress.** `pred_shape = input_betas` makes the betas branch
   of `compute_loss` identically zero, so only pose / cam / keypoints train.
   Downstream code (`mano.query`, the loss dict, the wandb visualizer) sees
   the same `(pred_pose, pred_shape, pred_cam)` triple as before.

### 3.4 Warm-start from a vanilla HaWoR checkpoint

Loading a pretrained HaWoR ckpt would normally fail strict `load_state_dict`
because of the head divergence. `HAWORStatic.__init__` filters the source
checkpoint:

| Source key | Action |
|---|---|
| `mano_head.decshape.{weight, bias}` | dropped (no betas regression) |
| `mano_head.init_betas` | dropped (no betas readout) |
| `mano_head.encshape.*` | kept at xavier init (no source) |
| everything else (backbone, st_module, motion_module, mano_head.transformer / decpose / deccam) | loaded |

`encshape`'s small-gain init means the warm-started model behaves like the
source ckpt at step 0; gradient flow into `encshape` lets it discover shape
fusion over training. The two flavours of warm start are committed as
configs:

- `experiment/hawor_static.yaml` — clip 1.0 grad clip, no warmup
- `experiment/hawor_static_v2.yaml` — `WARMUP_STEPS: 1000`, `GRAD_CLIP_VAL: 0`
  (linear LR ramp; rely on the GradScaler skip path to absorb early-step inf
  grads instead of clipping them away)

### 3.5 What "shape input" means at inference

At training time, `gt_cam_betas` comes from the per-frame label (so the
model is supervised that `pred_shape == GT betas`).

At inference time on a new egocentric stream, the caller picks a single
per-stream betas vector — for example:
- a one-pass HaMeR prediction averaged over the first N frames (the
  per-sequence mean strategy already proven for `hawor_rlhndv1_gt_shape` in
  `scripts/eval_video.py`),
- a calibrated betas from a short setup capture, or
- defaults (zeros) for unknown wearers.

The model then locks shape to that vector for every frame of the stream while
pose and camera adapt freely. This is the egocentric-friendly counterpart of
the gt-shape eval path: it gives the same temporal stability without
requiring GT betas at inference.

---

## 4. Datasets and Mixes

### 4.1 `VideoDataset` parity gains over `models_clip`

Both repos ship a `video_dataset.py`, but the `models_vid` version is the one
HaWoR loss code expects (522 lines vs 425). It adds the **HaWoR-style alias
fields** that `compute_loss` consumes directly:

```python
item['gt_cam_j2d']        = keypoints_2d[..., :2]        # (T, 21, 2)
item['gt_j3d_wo_trans']   = keypoints_3d[..., :3]        # (T, 21, 3)
item['gt_cam_full_pose']  = [global_orient, hand_pose]   # (T, 48) axis-angle
item['gt_cam_betas']      = mano_params['betas']         # (T, 10)
item['do_flip']           = right_label_inverted         # (T,) float
```

plus per-frame `img_focal` (mean of fx, fy from the intrinsics matrix; falls
back to the frame diagonal when the focal field is missing) and the
`img_center = (W/2, H/2)` principal point. With these in place, both `HAWOR`
and `HAWORStatic` consume the dataset directly — no `_prepare_batch`
indirection.

The torchcodec-based mp4 decode path, the `.frames.npy` sidecar, the
bounded LRU on the frame map, and the `_skip_sample` retry/log loop are all
inherited from the migration described in
[hamer-rlhndv1-and-data-updates §5](../hamer-rlhndv1-and-data-updates/report.md).

### 4.2 New dataset entries in `datasets_vid.yaml`

The video-dataset YAML now lists every clip dataset in a single file, plus a
new entry for the **Quest egocentric data**:

```yaml
HOT3D-QUEST-CLIP-TRAIN:
    TYPE: VideoDataset
    DATASET_FILE: _DATA/haptic_training_label/hot3d_quest/clip/...
    video_dir: _DATA/haptic_training_videos/hot3d_quest/
    label_dir: _DATA/haptic_training_label/hot3d_quest/clip/
```

### 4.3 New `mix_quest.yaml`

A Quest-leaning mix: `HOT3D-QUEST-CLIP-TRAIN` at weight 0.20 alongside the
ARCTIC / DexYCB / HO3D / H2O / HOT3D mix. Useful for `HAWORStatic` runs that
should bias toward egocentric headset footage, where shape-input control
matters most.

---

## 5. Training Entrypoint

`scripts/train_vid.py` mirrors `scripts/train_clip.py` but the model registry
collapses from seven entries to two:

```python
# train_clip.py
model_cls = {
    'hawor': HAWOR,
    'hamer': HAMER,
    'hamer_manotorch_fix_shape_euler': HAMERMANOTorchFixShapeEuler,
    'hawor_manotorch_fix_shape_euler': HAWORMANOTorchFixShapeEuler,
    'hawor_rlhndv1': HAWORRlhndv1,
    'hawor_rlhndv1_fix_shape': HAWORRlhndv1FixShape,
    'hawor_rlhndv1_gt_shape': HAWORRlhndv1GtShape,
}

# train_vid.py
model_cls = {
    'hawor': HAWOR,
    'hawor_static': HAWORStatic,
}
```

Default dataset config is `datasets_vid.yaml`, hydra entrypoint is
`models_vid/configs_hydra/train.yaml`, wandb project is `260503` (vs
`20260401` for the clip stack).

---

## 6. Summary

- `models_vid/` is a slim, video-only fork: two model classes, one dataset
  pipeline (`VideoDataset` over mp4), one training script.
- `HAWOR` is reverted to the public HaWoR DexYCB-style batch convention so
  the dataset can feed it directly without an adapter, and gains scale-floor
  / warmup / grad-scale-logging hardening for fp16 DDP stability.
- **`HAWORStatic`** introduces shape as a control input rather than a
  regression target: the per-stream betas are encoded by a fresh MLP and fused
  with the cross-attended decoder token, while `pred_shape` echoes the input
  so loss / MANO / logging stay byte-compatible. Shape is then locked across
  the egocentric stream, eliminating per-frame mesh-size flicker and
  delivering more stable, plausible egocentric video hand tracking.
- A new `HOT3D-QUEST-CLIP-TRAIN` source and a `mix_quest.yaml` mix are
  available for the egocentric-leaning runs that benefit most from the
  shape-input control.

---

## Key Files

| File | Description |
|---|---|
| `models_vid/hawor/hawor.py` | Restored DexYCB-style HaWoR with real per-frame focal, `do_flip` un-flip path, scale floor, warmup, grad-scale logging |
| `models_vid/hawor/modules.py` | Flattened `MANOTransformerDecoderHead` + `temporal_attention` (formerly `hawor/heads/modules.py`) |
| `models_vid/hawor/track_dataset.py`, `imutils.py` | Self-contained inference dataset + crop utils (no `lib/` dep) |
| `models_vid/hawor_static/hawor_static.py` | `HAWOR` subclass; swaps head + reads `gt_cam_betas`; warm-start from vanilla HaWoR ckpt |
| `models_vid/hawor_static/modules.py` | `MANOTransformerDecoderHeadStaticShape`: `encshape` MLP + sum-fusion + `pred_shape = input_betas` |
| `models_vid/datasets/video_dataset.py` | VideoDataset with HaWoR alias fields (`gt_cam_*`, `do_flip`, real per-frame focal) |
| `models_vid/configs/datasets_vid.yaml` | Single unified dataset YAML, including HOT3D-QUEST entry |
| `models_vid/configs_hydra/data/mix_all.yaml` | Standard mix with HOT3D-QUEST @ 0.20 |
| `models_vid/configs_hydra/data/mix_quest.yaml` | Quest-leaning mix (new) |
| `models_vid/configs_hydra/experiment/hawor.yaml` | Vanilla HaWoR experiment |
| `models_vid/configs_hydra/experiment/hawor_static.yaml` | Static-shape, GRAD_CLIP=1.0 |
| `models_vid/configs_hydra/experiment/hawor_static_v2.yaml` | Static-shape, WARMUP_STEPS=1000, GRAD_CLIP=0 |
| `scripts/train_vid.py` | Entrypoint (hawor / hawor_static), hydra config under `models_vid/configs_hydra/` |
