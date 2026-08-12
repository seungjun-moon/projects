# Force Training Report: OpenTouch Label Correction & Loss Rebalancing

*2026-08-12 — prepared for weekly review. Covers the 08-10 ~ 08-12 work on the
HOPE-protocol force/contact models (hand pressure estimation from egocentric /
third-person video onto MANO vertices).*

---

## 1. Executive summary

1. **All OpenTouch force labels used before 08-11 were polarity-inverted.**
   The released sensor stream encodes *high pressure as low counts*; our
   conversion assumed the opposite. Every prior force model was trained to
   predict the *absence* of pressure. Found via visual auditing (credit:
   repeated challenges in review), confirmed by distribution statistics over
   all 26 recordings, fixed with a one-line formula change, all labels
   regenerated, all force models retrained.

2. **The training loss carried two structural biases** that suppressed
   pressure-magnitude learning: (i) a reduction inconsistency that made the
   contact term outweigh the force term by 10³–10⁴× on shared parameters, and
   (ii) OpenTouch dominating PVDB in force-gradient mass by ~5–15×. Both are
   now measured, fixed, and validated by a controlled rerun.

3. **Standing vs the HOPE paper** (its exact PVDB split, its metrics, faithful
   PressureVision IoU protocol): we now lead every contact metric on both
   datasets, tie contact IoU, and trail only PVDB pressure-magnitude metrics —
   with the rebalanced run closing that gap (0.525 → 0.496 kPa MAE at 42% of
   training; HOPE: 0.449).

---

## 2. OpenTouch label extraction (corrected)

### 2.1 Sensor model

The OpenTouch glove carries a 16×16 taxel array (256 elements) over the palm
side of the right hand. Each taxel reports an ADC count

  c ∈ [0, 3072],  with **c ≈ 3072 at rest and c → 0 under pressure**

(i.e., the released `right_pressure` stream is sensor-native and *inverted*).
The dataset authors state a full-scale pressure of 50 kPa. The correct
per-taxel conversion is therefore

  **p [kPa] = (1 − c/3072) · 50 = (3072 − c) · 50/3072.**

Two properties matter downstream: taxels idle at c ≈ 0.64–0.88 of range (a
per-taxel idle offset of ~2–6 kPa after conversion, not exactly zero), and the
encoding hard-caps at 50 kPa — OpenTouch cannot represent harder presses.

### 2.2 What went wrong before

The original pipeline treated counts as press-*positive* and applied a
per-clip baseline:

  p_old = clip(c − P10_clip(c), 0) · s   (s: an interim scale, later 50/3072)

With inverted counts, P10_clip(c) selects the *most-pressed* frames as the
"baseline," so p_old is largest when the hand is idle and ~0 under hard press
— a fully inverted label that nevertheless *looks* plausible in visualization,
because the per-clip baseline preserves the correct **spatial** pattern (the
right vertices light up) while flipping the **temporal** one (at the wrong
moments). This is why frame-by-frame visual auditing, not aggregate metrics,
caught it.

### 2.3 How the inversion was proven

- **Distribution statistic**: for every active taxel, locate its median within
  its own [min, max] range. Across all 26 recordings the median sits at
  0.64–0.88 of range — signals *rest* near the top. A press-positive signal
  would idle near the bottom.
- **Video-anchored events**: frames where the hand is verifiably airborne show
  c ≈ 2600–3072; verifiable grips plunge the contact taxels to ≈ 0.
- **Cross-dataset consistency after the fix**: in-contact pressure medians
  become OpenTouch 3.4 kPa vs PVDB 2.5 kPa (physically compatible), where the
  inverted labels had implied ~45 kPa on idle hands.

### 2.4 Taxel → MANO-vertex mapping

The original annotation projected taxels onto the posed MANO mesh with
pose-dependent weights; that pipeline is not recoverable. We rebuilt a
**static mapping** by correlating each sensing vertex's label series against
all 256 taxel series per recording (per-demo standardized), taking the
majority vote across the 26 recordings: 267 sensing vertices → 46 distinct
taxels, 194/267 with an absolute majority. Resolution reality check: each
fingertip is covered by only **2–6 physical taxels** (upsampled ~10× onto ~36
mesh vertices), versus **50–100 sensels** under a fingertip on PVDB's pad —
vertex-level metrics on OpenTouch therefore flatter localization, and
"summed pressure" values are resolution artifacts (pressure is intensive;
only per-vertex means/maxima, or area-weighted sums approximating Newtons,
compare across sensors).

### 2.5 Supervision masks and contact

Force supervision exists only inside a coverage set Ω (where the sensor can
measure): Ω_OT is the static 267-vertex glove area (100% of frames);
Ω_PVDB is the per-frame pad-visible band (median 28 vertices, empty in 33% of
frames). Contact ground truth is derived from force: contact = 1[p > τ],
τ = 1 kPa. Outside Ω the force loss is silent; predictions there are
disciplined only indirectly, through the contact gate (§3.1).

---

## 3. Training loss: diagnosis and changes

### 3.1 The loss as it was

With per-vertex contact logits ĉ_v and normalized pressure p̂_v (scale
s = 110 kPa), gated output p̂ = σ(ĉ)·relu(p̃)·s, the stage-2 loss was

  L = λ_c · L_contact + λ_p · L_force,  λ_c = λ_p = 0.01

  L_contact(frame) = **Σ**_{v=1..778} BCE(ĉ_v, c_v)      (a SUM)
  L_force(frame)  = (1/|Ω|) **Σ**_{v∈Ω} | p̂_v − p_v | / s  (a MEAN)

### 3.2 Bias 1 — reduction inconsistency (contact ≫ force)

The sum/mean mismatch makes the two terms live on different scales:

  L_contact(frame) ≈ 778 · BCE̅ ≈ O(10²–10³) · λ  vs
  L_force(frame)  ≈ |err|̅ / 110 ≈ O(10⁻²) · λ

so on **shared** parameters (the temporal module and decoder both heads read
from) the contact gradient outweighs force by **~10³–10⁴×**. The HOPE paper's
Eq. 6 defines *both* losses as means with λ_c = λ_p = 1; our sum-style BCE was
an inherited house convention, and the imbalance silently rendered the nominal
1 : 1 weighting meaningless.

### 3.3 Bias 2 — OpenTouch dominance over PVDB

Sampling is an even 50/50 draw per training example, but the force-gradient
mass is not:

- **Coverage**: 100% of OT frames carry force supervision vs **64%** of PVDB
  frames (hand off the pad ⇒ Ω empty ⇒ zero force gradient) → OT : PVDB
  effective force frames ≈ 61 : 39.
- **Magnitude**: per-frame mean |p| over Ω is 3.90 kPa (OT) vs 1.13 kPa
  (PVDB); median per-vertex loss-unit values differ ~10×. Since L1 gradient
  mass scales with target magnitude, an average OT frame contributes **3–10×**
  the force gradient of a PVDB frame.
- Combined: force learning is **~5–15× OT-dominated**, and the >50 kPa regime
  — which only PVDB can teach (OT is capped) — is taught by the faintest
  voice in the mix. This precisely matched the observed symptom set: OT force
  at paper level early, PVDB magnitude metrics persistently lagging.

### 3.4 The corrected loss (the "balanced" run)

  L_contact(frame) = **(1/778) Σ**_v BCE(ĉ_v, c_v)      (mean; HOPE-faithful)
  L_force = Σ_f w_f · L_force(f) / Σ_f w_f · 1[Ω_f ≠ ∅],  **w_f = 4 if f ∈ PVDB else 1**
  λ_c = λ_p = 1;  PVDB sampling weight ×1.6 (cancels the coverage deficit)

Additionally, a **scene-level OpenTouch holdout** was carved (4 recordings,
249 clips, 9.8%) — the first honest OpenTouch validation set; all previous OT
numbers were train-seen.

### 3.5 Validation of the fixes (controlled rerun, same architecture)

| PVDB (held-out) | unbalanced @70k | balanced @85k | HOPE (paper) |
|---|---|---|---|
| force MAE, paper regime (kPa) | 0.525 | **0.496** | 0.449 |
| force MAE on contact region | 9.19* | **8.78** | — |
| vertex contact F1 | 0.721 | 0.716 | — |
| frame contact F1 | 0.926 | 0.924 | 0.905 |

| OpenTouch | unbalanced @70k (train-seen) | balanced @85k (**held-out**) | HOPE (their test) |
|---|---|---|---|
| force MAE (kPa) | 1.801 | 1.646 | 1.808 |
| force MAE on contact region | 4.27* | **4.05** | 5.84 |
| vertex contact F1 | 0.686 | **0.648 (honest)** | 0.660 |

*\* unbalanced values are from its final 200k checkpoint.*

Force improved on **both** datasets without any contact cost, and the honest
held-out OpenTouch contact F1 (0.648) lands within 0.012 of HOPE's test-set
figure — the earlier train-seen numbers were not a mirage.

---

## 4. Evaluation fairness (established alongside)

- **Split**: our PVDB train partition matches HOPE's at exactly 1,672
  sequences; test = the official val_fold_5 (~99% sequence overlap; the delta
  is calibration routines and two-handed sequences excluded by all
  single-hand pipelines).
- **Denominator honesty**: published PVDB vertex-MAE numbers live on a
  zero-diluted vertex set (in HOPE's own table a near-silent model scores
  best-in-column 0.248 kPa). We report both a strict per-frame-Ω MAE
  (predict-zero costs 4.2 kPa) and a paper-comparable loose-Ω MAE
  (predict-zero: 0.595), each against its trivial baseline.
- **Pixel IoU**: reimplemented PressureVision's protocol faithfully (contact
  IoU at 1 kPa and volumetric IoU Σmin/Σmax, dataset-summed) against the raw
  Sensel images, with mesh rasterization of per-vertex predictions. A
  **representation ceiling** was measured by projecting ground-truth vertex
  labels through the same chain: contact/vol IoU 0.516/0.422 — nearly equal to
  pad-native PressureVision's 0.547/0.411, quantifying the intrinsic cost of
  the vertex-space formulation that HOPE also pays.

## 5. Current standing (final 200k checkpoints; balanced still training)

- Contact: lead or tie every metric vs HOPE on both datasets, including
  faithful contact IoU (0.327–0.341 vs 0.328) and PVDB frame F1 within 0.001
  of pad-native PressureVision.
- Force: contact-region MAE better than HOPE on OpenTouch (4.05 vs 5.84);
  PVDB magnitude metrics (MAE/RMSE/vol IoU) still trail and are the target of
  the balanced run and of the next iteration (full-image context for the
  force head; HOPE-style loss gating).
- Architecture ablation (matched budget): the per-vertex readout more than
  doubles the global-token design on pad localization (contact IoU 0.327 vs
  0.140); the global-token variant never meaningfully beats a predict-zero
  baseline on PVDB force.

## 6. Open items

1. Author follow-ups: her evaluation Ω definition (removes the last
   denominator asterisk) and her OpenTouch train/test split.
2. Balanced-run finals (due today) → final report table.
3. Next design iteration for the PVDB pressure tail (>50 kPa regime).
