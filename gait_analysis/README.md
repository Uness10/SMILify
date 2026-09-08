# Gait analysis for the SMILySTICKS single-view runs

Two things live here:

1. **`find_gait_windows.py`** — picks clean gait-cycle windows *out of the footage*,
   not out of thin air. It decodes the keypoint markers that are already drawn into
   the collage MP4s and scores every candidate window on step-period regularity,
   step-amplitude regularity and heading stability.
2. **`analyse_gait.py`** — the plots your supervisor asked for, computed from the
   per-frame pose that `run_singleview_inference.py --export_animation` writes.

Everything is numpy/scipy/matplotlib + opencv. No torch needed for the analysis.

---

## What the footage says (already done, `SMILySTICKS_..._sv_ref.mp4`)

500 frames, 6 legs tracked from the ground-truth (panel-0) markers:

| | |
|---|---|
| median step period | **41 frames** (all six legs within ±1 frame) |
| coordination | tripod: **{R1, L2, R3}** alternating with **{R2, L3, L1}**, L3 lagging R2 by ~0.15 cycle |
| body heading | varies by only **13°** over the whole clip — the animal walks straight throughout |
| head–abdomen distance | CV **1.0 %** — no appreciable out-of-plane body rotation |
| model fit (GT vs predicted markers, leg joints) | **2.9 px** median for `sv_ref`/`1e-1`, **2.5 px** for `1e-4`, on a 512 px crop where the body is 169 px long |
| L1 (left front) | marker identity flips between the tarsus and neighbouring joints — **excluded from window scoring**; the model itself tracks it, so it is still plotted from the pose export |

**Recommended windows** (dataset frame numbers, both verified by eye on the
overlay panel):

| window | frames | cycles | why |
|---|---|---|---|
| **W1 (primary)** | **32 – 196** | 4 | best-scoring 4-cycle stretch; heading flat, all five reliable legs stepping regularly |
| **W2 (replicate)** | **372 – 495** | 3 | best-scoring late window — an independent stretch, so any conclusion can be checked twice |
| (cleanest 3-cycle) | 62 – 185 | 3 | sits inside W1; use if you want the tightest possible example |

To redo this for another clip:

```bash
python gait_analysis/find_gait_windows.py <collage>.mp4 --panel-width 512 --exclude l_1_l
```

`--panel 0` reads the ground-truth markers, `--panel 1` the model's own — the
difference between the two is a free per-frame reprojection-error check.

> Heads up: the uploaded `..._sv_ref.mp4` and `..._1e1.mp4` are **byte-identical**
> (same md5). Whatever produced them, one of the two exports did not get the run it
> was named after.

---

## Running the analysis

Add one flag to the inference command you already use:

```bash
python smal_fitter/neuralSMIL/run_singleview_inference.py ... \
    --export_animation inference_out/SMILySTICKS_..._sv_ref
```

That writes `<stem>.npz` + `<stem>.json` (`poses` `(F, 55, 3)` axis-angle,
`trans`, `betas`, `betas_per_frame`, `log_beta_scales`, `betas_trans`, `fps`).
Then:

```bash
python gait_analysis/analyse_gait.py \
    --npz  inference_out/SMILySTICKS_..._sv_ref.npz \
    --smal-file 3D_model_prep/SMILy_STICK.pkl \
    --windows 32-196 372-495 \
    --label sv_ref --out gait_figs
```

`run_on_cluster.sh` does both steps for the three checkpoints; fill in the
checkpoint/dataset paths at the top.

Flags worth knowing:

* `--propagate-scaling` — set it if the checkpoint was trained/rendered with
  `propagate_scaling=True`, so the per-joint scales compose the same way.
* `--frame-offset N` — if the export does not start at dataset frame 0; the
  `--windows` above are dataset frame numbers.
* `--foot-segment ta|pt` — tarsus (default) or pretarsus tip.

---

## What comes out

| file | what it shows |
|---|---|
| `fig1_ThC_two_dominant_axes_<run>_w<i>.png` | protraction/retraction **α** and levation/depression **β** of all six thorax–coxa joints, swing phases shaded |
| `fig2_hinge_angles_<run>_w<i>.png` | CTr (“C-TF”) interior angle **δ** and FTi interior angle **γ** — the downstream joints now constrained to one DoF |
| `fig3_axis_dominance_<run>.png` | every joint ranked by how much it actually rotates, plus whether its dominant axis lines up with the anatomical one |
| `fig4_tarsus_horizontal`, `fig5_tarsus_sagittal` | tarsus position relative to the root, global rotation removed, one trace per step cycle so successive steps overlay, with the cycle-mean on top |
| `fig6_gait_diagram_<run>_w<i>.png` | swing/stance bars + foot position along the body axis |
| `angles_*.csv`, `tarsi_*.csv`, `steps_*.csv`, `axis_dominance_*.csv` | everything above as numbers, per frame |

### Conventions

Taken from the rig itself and from the repo's own validation
(`diagnostics/probe_stick_joint_axes_PROBE.py`,
`diagnostics/joint_dof_VERIFY_PROBE.log`), not assumed:

* Model frame **+x anterior, +y left, +z dorsal**. All output is in the animal's
  own body frame: the predicted global rotation is zeroed and the root joint
  (`b_t`) put at the origin, so a foot trajectory is comparable across steps.
* **ThC** (`l_N_co_*`) is the 3-DoF ball joint. Its relative rotation is decomposed
  `R = Rz(α) Rx(β) Ry(γ)` in the model frame, with the per-side sign flips the rig
  probe measured (+z is protraction on the right, retraction on the left; +x is
  depression on the right, levation on the left), then the rest-pose values are
  added so the plotted angles are absolute and comparable to the literature.
  Because they come from the pose parameter, they do not depend on what the
  joints downstream are doing.
* **CTr** (`l_N_tr_*`) and **FTi** (`l_N_ti_*`) are the 1-DoF hinges. Their axis is
  the least-squares normal of the resting leg plane, signed so that positive =
  flexion — determined by rotating the joint and watching the interior angle,
  never assumed. The off-axis residual is reported: for a true hinge it stays at 0.
* **`l_N_fe_*`** (trochanter–femur) is locked to zero on all three axes — the
  segments are fused (Theunissen et al. 2015). It appears in the ranking as a
  control: it should sit at the bottom.
* Distances are in **% body length** (rest-pose head to abdomen-tip distance), so
  they are independent of the predicted mesh scale.
* Swing/stance from the sign of the foot's longitudinal velocity in the body
  frame, with minimum run lengths — this recovers the duty factor, a fixed
  percentile threshold does not.

Caveat worth keeping in mind: the rig is not perfectly left/right symmetric
(front-leg rest δ is 154° on the right, 138° on the left), so left and right
absolute angles are not expected to coincide exactly even for perfectly
symmetric motion.

---

## Verification

```bash
python gait_analysis/test_fk.py .           # from the repo root
python gait_analysis/test_roundtrip.py 3D_model_prep/SMILy_STICK.pkl
```

* `test_fk.py` checks the numpy forward kinematics against the repo's own
  `smal_model/batch_lbs.batch_global_rigid_transformation` (with and without
  `propagate_scaling`, with per-joint scales and translations) — max error 1.4e-7 —
  and that a zero pose reproduces the shape-dependent rest joints exactly.
* `test_roundtrip.py` synthesises a walk with known ThC amplitudes and midpoints
  via `make_synthetic_export.py`, runs the full analysis and checks the numbers
  come back: midpoints and half-ranges recovered to <0.5°, the fused
  trochanter–femur reads ~0, and the FTi off-hinge-axis residual stays ~0.
  `make_synthetic_export.py` also gives you a dummy `.npz` to preview the figures
  before a checkpoint is available.
