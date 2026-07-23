# Joint-angle prior study — baseline pipeline

Quantifies **how and in what form** benefits arise from adding a user-defined
joint-angle prior to the neural stick-insect model.

**Hypothesis (supervisor):** adding authored per-joint rotation ranges should yield
more *realistic* recovered poses while retaining — or improving — 3D position (MPJPE)
and 2D reprojection (PCK) accuracy.

This folder produces the **"before" (unconstrained) column** of that comparison. Re-run
it later against a constrained checkpoint and diff the two output folders.

---

## Key finding to discuss first

The stick model `3D_model_prep/SMILy_STICK.pkl` **has no `joint_limits` key yet** (probed:
55 joints, keys include `J_names` but not `joint_limits`). So the "user-defined prior"
ranges have not been authored into this model. Two consequences:

1. The unconstrained model is genuinely unconstrained — `joint_limits_prior.py` falls back
   to wide-open ±π, and `joint_limit_regularization` is off by default in the configs.
2. **The baseline empirical angle distributions this pipeline produces are exactly the data
   you'd use to author sensible ranges** (e.g. bound each joint/axis around the observed
   distribution of clean poses, or to biological limits). That authoring step is the natural
   bridge between this baseline and the constrained run.

Until limits are authored, the violation table is skipped. You can still enable it early by
passing `--limits <file>` (an `(J,3,2)` `.npy` in radians, or a `{joint_name: [[min,max]x3]}`
`.json`) — e.g. a first-draft biological range — to see how far the current model strays.

---

## What it measures

**Accuracy (what we must not lose)** — scraped from `benchmark_model`'s report:
MPJPE (mean + median, mm) and PCK@5px at native (1530px) and input (224px) resolution.

**Pose realism (what the prior should improve)** — from the exported per-frame parameters:

- **Joint-angle distributions** — per joint, per axis-angle component, with authored
  limit lines overlaid (`joint_angle_distributions.png`, `per_axis_stats.csv`).
- **Range of motion** — per-joint rotation magnitude ‖axis-angle‖ (`range_of_motion.png`,
  `magnitude_stats.csv`).
- **Limit violations** — % of frames outside the authored range and mean/max overshoot in
  degrees, per joint/axis (`limit_violations.csv`; needs authored limits).
- **Trajectories** — global body translation over time and leg-tip joint angle over time
  (`trajectories/`), to spot jitter or implausible excursions.

Everything carries a `--label` column so the constrained run drops straight into a
side-by-side table.

> **Comparison axis:** authored limits are defined per axis-angle *component*, and the
> training-time hinge loss clamps each component independently — so per-axis comparison of
> `poses[:, j, axis]` against `limits[j, axis]` is the correct apples-to-apples measure. The
> magnitude ‖axis-angle‖ is reported alongside as a representation-robust intuition only.

The six featured joints default to the leg tips
(`l_1_pt_l/r`, `l_2_pt_l/r`, `l_3_pt_l/r`) — the `joint_importance` set in
`multiview_sticks_UNET_optimal.json`. Override with `--important-joints`.

---

## Run it

Prereqs: conda env `pytorch3d` active, run from repo root, and the example checkpoint +
dataset downloaded to the repo root per `GETTING_STARTED.md`
(`SMILySTICKS_ViT_model.pth`, `SMILySTICKS_centred_reprojected_FIXED.h5`).

```bash
# full run (benchmark + inference + analysis)
bash scripts/prior_study/run_baseline_study.sh

# override paths / label / quick pass
CHECKPOINT=my.pth DATASET=my.h5 LABEL=unconstrained MAX_FRAMES=300 \
  bash scripts/prior_study/run_baseline_study.sh

# later: the constrained model, then diff prior_study_results/{unconstrained,constrained}
LABEL=constrained CHECKPOINT=constrained.pth LIMITS=authored_limits.npy \
  bash scripts/prior_study/run_baseline_study.sh
```

Outputs land in `prior_study_results/<label>/` (analysis subfolder + copied benchmark report).

### Analysis engine alone

If you already have an exported clip (`run_multiview_inference --export_animation <stem>`):

```bash
python scripts/prior_study/analyze_baseline_pose.py \
    --npz clip.npz --json clip.json \
    --smal-file 3D_model_prep/SMILy_STICK.pkl \
    --benchmark benchmark_.../benchmark_report.txt \
    --label unconstrained --out prior_study_results/unconstrained/analysis
```

### Validate without data

```bash
python scripts/prior_study/analyze_baseline_pose.py --self-test --out /tmp/selftest
```

Fabricates a synthetic 55-joint clip with a deliberate out-of-range excursion and runs the
whole pipeline — confirms plotting/tables work before committing GPU time.

---

## Suggested discussion agenda (for the sit-down)

1. Where does the unconstrained model already produce implausible angles (violation table /
   distribution tails)? Those joints are where a prior has the most to gain.
2. What is the current accuracy floor (MPJPE / PCK) we must protect?
3. How should the ranges be authored — biological limits, or data-driven bounds fit to the
   clean subset of these distributions?
4. Apply via soft `joint_limit_regularization` (already implemented, Issue #56) or hard
   clamps? Re-run and diff.
