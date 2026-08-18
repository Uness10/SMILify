# Joint-limit prior study — single-view vs multi-view

**Question.** Does adding authored per-joint rotation limits produce more
*realistic* recovered poses while retaining 3D position (MPJPE) and 2D
reprojection (PCK) accuracy — and does the answer differ between single-view and
multi-view reconstruction?

**Design.** Constrained arm only. Each modality's downloaded checkpoint is
fine-tuned for **10 more epochs** with `joint_limit_regularization = 100.0`
against `3D_model_prep/SMILy_STICK_limits_authored.pkl`. The reference is that
same downloaded checkpoint evaluated as-is. Four evaluations, two comparisons:

| arm | checkpoint | limits during training | limits when scoring |
|---|---|---|---|
| `sv_reference` | downloaded single-view `.pth` | no | **yes** |
| `sv_constrained` | `sv_reference` + 10 epochs | **yes**, w=100 | **yes** |
| `mv_reference` | downloaded multi-view `.pth` | no | **yes** |
| `mv_constrained` | `mv_reference` + 10 epochs | **yes**, w=100 | **yes** |

All four are scored against the same authored ranges. The reference arms were
trained without limits but must still be *measured* against them — that
violation rate is the "before" number the study rests on.

Runs on **RWTH CLAIX-2023** (`c23g`, 4× H100).

---

## Read this first — four things that will bite you

**1. The fine-tuning confound is real and is not controlled for.**
The reference arms get zero additional epochs, so every difference in the final
table is *the prior plus 10 epochs of continued training*, not the prior alone.
The control that would separate them — the same 10 epochs at `w_limit = 0` — was
deliberately dropped from this study. Say so when presenting the numbers. If a
reviewer pushes back, that control is one extra `prepare_resume_config.py` call
with `--joint-limit-weight 0.0` and two more array tasks.

**2. `training.num_epochs` is an absolute END epoch, not a count.**
Both trainers set `start_epoch = checkpoint["epoch"] + 1` and loop
`range(start_epoch, num_epochs)`. A checkpoint at epoch 240 with
`num_epochs: 250` gives you 9 more epochs; with `num_epochs: 10` it trains
**zero** and exits looking successful. `prepare_resume_config.py` reads the epoch
out of the checkpoint and writes `num_epochs = epoch + 1 + N`; the training
sbatch re-checks it before allocating GPUs.

**3. `batch_size` is PER PROCESS under DDP.**
On 4 GPUs the checkpoint's batch size is quadrupled, so the continuation runs at
a different effective batch — and therefore a different gradient-noise scale —
than the run that produced the checkpoint, while inheriting that run's
learning-rate curriculum unchanged. Either divide `training.batch_size` by 4 in
the prepared config to match the original regime, or accept 4× knowingly. The
training job prints the arithmetic in its pre-flight block.

**4. The penalty is averaged over ALL non-root axes, not just the authored ones.**
`_joint_limit_penalty` is a `torch.mean` over the full `(N, N_POSE, 3)` hinge
(`smil_image_regressor.py:1445`). Free axes contribute exact zeros, so the
*fraction* of authored axes scales the effective prior strength. At the current
**79.6 %** coverage, `w = 100` delivers an effective ≈ 80 — near its nominal
value. This is a recent change: the earlier limit set covered 18.5 % of axes, so
the same `w = 100` was diluted 4.3×. **Do not carry over weight recommendations
written against the old file.**

---

## The model file

**`3D_model_prep/SMILy_STICK_limits_authored.pkl`** — the only model to use here.
`3D_model_prep/SMILy_STICK.pkl` has no `joint_limits` at all, and the trainer
raises rather than training a silent no-op.

Probed contents (`diagnostics/limits_coverage_PROBE.txt`):

| Check | Result |
|---|---|
| `joint_limits` shape | `(55, 3, 2)`, float32, radians — matches `len(J_names)` |
| `min <= max`, all finite | pass (`LimitPrior._ranges_from_joint_limits` validation) |
| all axes within `[-π, π]` | pass — no representation-ambiguous authoring |
| rest pose inside limits | pass — 0 ∈ `[min, max]` on every axis |
| Root `b_t` | `[0,0]` on all 3 axes — `LimitPrior` pins it, the fitter drops it via the `[3:]` slice |
| **Constrained axes** | **129 of 162 non-root (79.6 %)**, across **43 of 54** joints |
| Constrained ranges | 20°–160°, median 20° |
| Deliberately free | abdomen `b_a_1..5` and all six pretarsi `l_{1,2,3}_pt_{r,l}` (33 axes at ±π) |

The limits are authored from the stick-insect literature (Theunissen+2015,
Guschlbauer+2022, Dallmann+2016) — see `docs/joint_dofs/` for the derivation, the
per-axis CSV and the build script. The `.pkl` matches that CSV exactly.

---

## Files

| File | What it does |
|---|---|
| `prepare_resume_config.py` | Derives a continuation config from a checkpoint. **Mode-aware**: reads the modality off the checkpoint's weights and refuses to proceed if the base config disagrees. Sets `resume_checkpoint`, the absolute `num_epochs`, the joint-limit weight, and isolated output dirs. |
| `export_poses.py` | Runs a **single-view or multi-view** checkpoint over the same test split the benchmark scores and writes the `.npz`/`.json` pair `analyze_baseline_pose.py` expects. Needed because neither inference entry point accepts an HDF5. |
| `run_prior_study.sh` | One arm, end to end: benchmark → export → analyse into `prior_study_results/<label>/`. Detects the modality from the checkpoint. |
| `compare_arms.py` | Joins the four arm folders into `comparison.md` + `arms.csv` — the study's actual deliverable. |
| `preflight_study.py` | Login-node guard for the two silent invalidations: config/checkpoint architecture drift (which resets the epoch counter to 0) and seed/ratio disagreement between arms (which scores them on different frames). |
| `analyze_baseline_pose.py` | The analysis engine: angle distributions, ROM, limit violations, trajectories. Unchanged. |
| `../../hpc_files/rwth/run_prior_study_train.sbatch` | Array `0-1` (SV, MV), 4× H100 each, `torchrun --standalone`. |
| `../../hpc_files/rwth/run_prior_study_eval.sbatch` | Array `0-3` (the four arms), 1 GPU each, runs `compare_arms.py` at the end. |

### What was removed

`README_singleview.md`, `run_singleview_study.sh`, `run_baseline_study.sh`,
`export_singleview_poses.py`, `submit_baseline.sbatch`,
`run_singleview_training_RWTH.sbatch` and `run_singleview_study_RWTH.sbatch` are
superseded by the mode-agnostic scripts above. They are in git history if needed.

---

## Run it on RWTH CLAIX-2023

Substitute your project for `<proj>` (`rwth####` / `p0######`).

### 0. One-time setup on the login node

```bash
cd "$HPCWORK/SMILify"
conda activate "$HPCWORK/conda_envs/pytorch3d"
python hpc_files/download_backbone_weights.py   # ViT weights; c23g nodes are offline
mkdir -p logs configs_runs
```

Keep the repo on `$HPCWORK` — `$HOME`'s quota will not hold a conda env plus the
21 GB dataset. If a job later reports `could not locate conda.sh`, pass it
explicitly: `CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh" sbatch --export=ALL,CONDA_SH ...`

### 1. Prepare both configs (login node)

```bash
# single-view
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint  "$SV_REF" \
    --base-config <the JSON that produced it> \
    --extra-epochs 10 --label constrained \
    --joint-limit-weight 100.0 \
    --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
    --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
    --out configs_runs/singleview_constrained.json

# multi-view
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint  "$MV_REF" \
    --base-config smal_fitter/neuralSMIL/configs/examples/multiview_SMILySTICKS_3D_ViT_Large_AUG_FIXED.json \
    --extra-epochs 10 --label constrained \
    --joint-limit-weight 100.0 \
    --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
    --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
    --out configs_runs/multiview_constrained.json
```

`--base-config` must be **the JSON that produced the checkpoint**. Omit it and
the script auto-discovers from the `<name>_checkpoints/` → `<name>.json` naming
convention.

> **Never let it fall back to the checkpoint's embedded config.** That block is
> serialized from the runtime config, so it carries stale defaults. On job
> 15521637 it yielded a **mouse** `data_path` paired with the 55-joint stick
> model, plus wrong `hidden_dim`/`freeze_backbone`. Training ran without error.
> The fallback is behind `--allow-embedded-config` and off by default, and the
> script cross-checks the model's joint count against the HDF5's `n_joints`.

Check the arithmetic in the output before submitting:

```
[prepare] checkpoint epoch: 240
[prepare] checkpoint kind : singleview (from the state dict)
[prepare] mode      : singleview
[prepare] resume from epoch 241, train 10 epoch(s) -> training.num_epochs = 251 (was 250)
[prepare] arm: CONSTRAINED (w=100.0)
[prepare] model/data cross-check: 55 joints on both sides — OK
[prepare] load_config round-trip: OK
```

### 2. Pre-flight (login node) — do not skip

```bash
export SV_REF=<path to the downloaded single-view .pth>
export MV_REF=<path to the downloaded multi-view .pth>

python scripts/prior_study/preflight_study.py \
    --config configs_runs/singleview_constrained.json --reference "$SV_REF" \
    --config configs_runs/multiview_constrained.json  --reference "$MV_REF"
```

This catches the two failures that **do not crash**:

- **Architecture drift.** `load_checkpoint` drops every tensor whose shape
  disagrees with the freshly-built model and then calls
  `load_state_dict(strict=False)`, returning `epoch = 0` when anything was
  dropped. A wrong `--base-config` therefore silently re-initialises layers *and*
  resets the epoch counter — so `num_epochs: 251` trains **251 epochs from
  scratch**, not 10, and the job eats the whole wall clock. The epoch arithmetic
  in the training job reads the checkpoint's stored epoch and is blind to this.
- **Split mismatch.** The benchmark and exporter derive the test split from
  `seed` + `train_ratio` + `val_ratio`. The reference arm takes them from the
  downloaded checkpoint's embedded config, the constrained arm from the config we
  prepared. If they disagree the two arms are scored on **different frames** and
  every number still looks plausible.

Exit 0 means safe to submit. The architecture half runs again inside the training
job, but fixing it here takes seconds instead of a queue wait.

### 3. Submit

```bash

jid=$(sbatch --parsable --account=<proj> \
        hpc_files/rwth/run_prior_study_train.sbatch)
echo "training array: $jid"

sbatch --dependency=afterok:$jid --account=<proj> \
    --export=ALL,SV_REF,MV_REF \
    hpc_files/rwth/run_prior_study_eval.sbatch
```

The reference arms don't depend on training, so you can score them immediately
and in parallel with the fine-tuning:

```bash
sbatch --array=0,2 --account=<proj> --export=ALL,SV_REF,MV_REF \
    hpc_files/rwth/run_prior_study_eval.sbatch
```

Monitor: `squeue --me` / `tail -f logs/jl_train_<jobid>_<task>.out`

### 4. Results

```
runs/{singleview,multiview}_constrained/
  checkpoints/  best_model.pth, checkpoint_epoch_*.pth, config.json
  plots/  visualizations/

prior_study_results/
  {sv,mv}_{reference,constrained}/
    arm.json                                <- provenance for compare_arms.py
    analysis/baseline_summary.md            <- per-arm tables
    analysis/limit_violations.csv           <- the before/after number
    analysis/joint_angle_distributions.png  <- dashed red = authored limits; features the
                                               COXAE, not analyze_baseline_pose's default
                                               pretarsi (those are free in this .pkl)
    analysis/range_of_motion.png
    benchmark_*/benchmark_report.txt        <- MPJPE mm + PCK
  comparison/
    comparison.md                           <- HEADLINE: both modalities + deltas
    arms.csv
```

`comparison.md` reports deltas as `constrained - reference`, annotated
per-metric (lower is better for violations and MPJPE, higher for PCK).

---

## Quick local checks (no GPU, no data)

```bash
# analysis engine end-to-end on a synthetic clip with a deliberate out-of-range joint
python scripts/prior_study/analyze_baseline_pose.py --self-test --out /tmp/selftest

# epoch arithmetic + mode handling without reading a checkpoint
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint /dev/null --assume-epoch 240 --extra-epochs 10 --mode multiview \
    --base-config smal_fitter/neuralSMIL/configs/examples/multiview_SMILySTICKS_3D_ViT_Large_AUG_FIXED.json \
    --label dryrun --out /tmp/dryrun.json --no-validate

# sbatch syntax
bash -n hpc_files/rwth/run_prior_study_train.sbatch
bash -n hpc_files/rwth/run_prior_study_eval.sbatch
bash -n scripts/prior_study/run_prior_study.sh
```

---

## If the prior shows no effect

In order of likelihood, before concluding the hypothesis is wrong:

1. **Nothing was violating those joints anyway.** Check the violation table in
   each `*_reference` arm. If the model was already inside the authored ranges,
   the hinge is near-zero by construction and there is nothing for it to fix.
   This is the single most informative diagnostic, and it costs no GPU time
   beyond the reference arms.
2. **10 epochs at a decayed LR.** The curriculum is inherited from the original
   run, so the tail LR may be too small to move the pose much. Raising
   `--extra-epochs` is cheaper than raising `w`.
3. **Weight too low.** At 79.6 % coverage the dilution factor is only 1.26×, so
   `w = 100` is close to nominal. If the hinge is firing but not winning against
   the keypoint losses, compare its magnitude against `keypoint_2d`/`keypoint_3d`
   in the training log before changing it.
4. **Multi-view has less to gain by construction.** Multi-view already resolves
   depth ambiguity geometrically, so a small multi-view delta beside a large
   single-view one is a *result*, not a failure.
