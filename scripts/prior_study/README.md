# Joint-limit prior study — single-view lambda sweep

**Question.** Does adding authored per-joint rotation limits produce more
*realistic* recovered poses while retaining 3D position (MPJPE) and 2D
reprojection (PCK) accuracy — and **at what weight**? The modality comparison
(single-view vs multi-view) is the follow-up; this pass sweeps the weight on
single-view, where the problem is underdetermined and the prior has the most to
do.

**Design — lambda sweep, single-view.** The single-view checkpoint is fine-tuned
for **50 more epochs**, once per weight in **{1e-4, 1e-3, 1e-2, 1e-1}**, against
`3D_model_prep/SMILy_STICK_limits_authored.pkl`. The reference is that same
downloaded checkpoint evaluated as-is. Five evaluations, one curve:

| arm | checkpoint | limits during training | limits when scoring |
|---|---|---|---|
| `sv_reference` | downloaded single-view `.pth` | no | **yes** |
| `sv_constrained` @ 1e-4 | `sv_reference` + 50 epochs | **yes**, w=1e-4 | **yes** |
| `sv_constrained` @ 1e-3 | `sv_reference` + 50 epochs | **yes**, w=1e-3 | **yes** |
| `sv_constrained` @ 1e-2 | `sv_reference` + 50 epochs | **yes**, w=1e-2 | **yes** |
| `sv_constrained` @ 1e-1 | `sv_reference` + 50 epochs | **yes**, w=1e-1 | **yes** |

All five are scored against the same authored ranges. The reference arm was
trained without limits but must still be *measured* against them — that
violation rate is the "before" number the study rests on.

Multi-view is **not** part of this sweep. Multi-view already resolves depth
geometrically, so the prior has least to do there; tasks 4-7 of the training
array and 5-9 of the eval array run the same four lambdas on multi-view when
that becomes worth the node-hours.

**Read reference-vs-lambda and lambda-vs-lambda differently.** The reference got
zero additional epochs, so any reference-vs-lambda delta is *the prior plus 50
epochs of continued training*. Comparisons **between** lambdas share the epoch
count and are clean. Read the sweep as a curve first; use the reference row as
context, not as the headline.

**The sweep sits far below the previous w=100.** At 79.6 % axis coverage the
effective weight is `lambda × 0.80`, so even the top of this sweep is ~1000×
weaker than the earlier setting. If every lambda comes back flat, that is the
most likely reason and the fix is to sweep upward, not to conclude the
hypothesis is wrong.

Runs on **RWTH CLAIX-2023** (`c23g`, 4× H100). Account `rwth2151` caps wall time
at **24 h**, which 50 epochs may not fit — see *Wall time* below.

---

## Read this first — six things that will bite you

**1. The fine-tuning confound is real and is not controlled for.**
The reference arm gets zero additional epochs, so every reference-vs-lambda
difference is *the prior plus 50 epochs of continued training*, not the prior
alone. Say so when presenting the numbers. The control that would separate them
is one more `prepare_lambda_sweep.sh` call with `LAMBDAS="0"` and one more array
task — cheap, and worth having ready before a reviewer asks. Lambda-vs-lambda
comparisons do not carry this confound.

**1b. Every arm of a sweep must train the same number of epochs.**
Otherwise the table varies lambda *and* training length at once and neither can
be attributed. `prepare_lambda_sweep.sh` passes one `EXTRA_EPOCHS` to every arm
for exactly this reason, and each training task prints the count it will run so
the logs can be checked against each other. If one arm is killed at the wall
clock and continued while the others were not, note it — or re-run them all.

**2. `training.num_epochs` is an absolute END epoch, not a count.**
Both trainers set `start_epoch = checkpoint["epoch"] + 1` and loop
`range(start_epoch, num_epochs)`. A checkpoint at epoch 354 with
`num_epochs: 404` gives you 50 more epochs; with `num_epochs: 50` it trains
**zero** and exits looking successful. This is why "50 epochs" is always passed
as `--extra-epochs 50` and never typed into the JSON.
`prepare_resume_config.py` reads the epoch out of the checkpoint and writes
`num_epochs = epoch + 1 + N`; the training sbatch re-checks it before allocating
GPUs.

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

**5. Nothing may overwrite the swept weight mid-run.**
`loss_curriculum.curriculum_stages` replaces weights at given epochs. A stage
inside the continuation's epoch range that sets `joint_limit_regularization`
would discard the lambda partway through and two sweep arms would silently
converge. `prepare_lambda_sweep.sh` strips those keys and the training sbatch
refuses to launch if any survive.

**6. Two arms can collide in the benchmark directory.**
`benchmark_model` derives its output directory from `getcwd()` plus the
checkpoint stem (`benchmark_model.py:1085`) with no override, and every swept
arm resumes into a file called `best_model.pth`. Two arms benchmarking
concurrently would interleave into the same
`benchmark_singleview_best_model_on_*/`. `run_prior_study.sh` now points the
benchmark at a per-arm-named symlink (`BENCH_TAG`, set by the eval sbatch to
`<label>_lam<lambda>`); the weights are identical, only the output path changes.

---

## Wall time

Account `rwth2151` allows **MaxWall 1-00:00:00**, and no longer QOS is attached
to this association — 24 h is the ceiling for a single job. The earlier
+10-epoch runs were budgeted at 12 h. Before submitting four 50-epoch tasks:

```bash
grep -h "Epoch\|epoch" logs/jl_train_*_*.out | grep -i "time\|elapsed\|completed" | tail -20
```

Multiply the per-epoch time by 50. If it does not fit, in order of preference:

1. **Lower `EXTRA_EPOCHS`.** The sweep is about lambda; the comparison holds as
   long as every arm gets the same count.
2. **Drop `dataset.dataset_fraction`** for the sweep and restore it for the
   winning lambda.
3. **Chain a continuation** with `--dependency=afterany` and a second
   `prepare_resume_config.py` against the newest `checkpoint_epoch_*.pth`.

`prepare_lambda_sweep.sh` sets `save_checkpoint_every = 2`, so a job killed at
the wall clock loses at most two epochs. The eval and render jobs chain with
`afterany`, not `afterok`, so a timed-out arm is still scored from its newest
periodic checkpoint.

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
| `prepare_lambda_sweep.sh` | Builds every config of the sweep in one call — `prepare_resume_config.py` once per lambda, all at the same `--extra-epochs`, plus the two post-patches that matter at 50 epochs (`save_checkpoint_every`, curriculum hygiene). |
| `compare_arms.py` | Joins a reference/constrained pair into `comparison.md` + `arms.csv`. Unchanged; the sweep gives each lambda its own root so this keeps working on the fixed pair names. |
| `pick_segments.py` | CPU only. Chooses which frame windows of the exported test split every arm renders — once, from the reference — so the panels line up. `--select worst` aims them at the frames the reference actually violates. |
| `render_clip_npz.py` | Renders one arm's exported `clip_*.npz` to MP4. No inference: SMAL forward + rasterisation of the predictions that were already scored. Burns a per-frame out-of-range counter into each frame. |
| `stack_renders.sh` | ffmpeg-only, login node. Stacks the per-arm segment renders into one comparison video per window. |
| `preflight_study.py` | Login-node guard for the two silent invalidations: config/checkpoint architecture drift (which resets the epoch counter to 0) and seed/ratio disagreement between arms (which scores them on different frames). |
| `analyze_baseline_pose.py` | The analysis engine: angle distributions, ROM, limit violations, trajectories. Unchanged. |
| `../../hpc_files/rwth/run_prior_study_train.sbatch` | Array `0-3` = the single-view lambda sweep (`4-7` = the same lambdas on multi-view), 4× H100 each, `torchrun`. Refuses to launch if the config's weight is not the lambda its task id is for. |
| `../../hpc_files/rwth/run_prior_study_eval.sbatch` | Array `0-4` = reference + the four lambdas (`5-9` = multi-view), 1 GPU each. Writes each lambda to its own root and rebuilds `sweep/sweep.md` at the end. |
| `../../hpc_files/rwth/run_prior_study_render.sbatch` | Array `0-4`, 1 GPU each. Renders every arm's exported poses over the shared windows. Refuses to start without `segments.json`. |

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
conda activate pytorch3d
python hpc_files/download_backbone_weights.py   # ViT weights; c23g nodes are offline
mkdir -p logs configs_runs
```

Keep the repo on `$HPCWORK` — `$HOME`'s quota will not hold a conda env plus the
21 GB dataset. If a job later reports `could not locate conda.sh`, pass it
explicitly: `CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh" sbatch --export=ALL,CONDA_SH ...`

**This matters more at 50 epochs than it did at 10.** `REPO_DIR` defaults to
`SLURM_SUBMIT_DIR`, so `runs/singleview_lam*/checkpoints/` is written relative to
wherever you submit from. Four lambdas × ~25 periodic checkpoints (50 epochs at
`save_checkpoint_every = 2`) of a ViT-Large model is on the order of 100 GB. From
a `$HOME` checkout the sweep dies partway through on a write error, not a clean
failure. Either submit from `$HPCWORK`, or repoint `output.*` in the prepared
configs — see the quota block at the top of `cmnds.txt`.

**The env prefix.** `ENV_PREFIX` is the env's full *prefix* (a directory), not
its name, and the layout differs per install: `conda/envs/<name>` for a conda
rooted at `$HPCWORK`, `conda_envs/<name>` if `envs_dirs` was pointed there. All
three sbatch files probe both and verify `bin/python` exists before calling
`conda activate`, so a wrong guess fails on the login node's terms rather than
after a queue wait. Override explicitly when you keep several envs:

```bash
ENV_PREFIX="$(conda info --envs | awk '$1=="pytorch3d"{print $NF}')" \
    sbatch --export=ALL,ENV_PREFIX ...
```

### 1. Build the sweep configs (login node)

```bash
export SV_REF=<path to the downloaded single-view .pth>
export SV_BASE_CONFIG=<the JSON that produced it>

EXTRA_EPOCHS=50 bash scripts/prior_study/prepare_lambda_sweep.sh singleview
```

That is `prepare_resume_config.py` once per lambda, writing
`configs_runs/singleview_lam{1e-4,1e-3,1e-2,1e-1}.json` — the exact names the
training array derives from its task id. Equivalent single call:

```bash
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint  "$SV_REF" \
    --base-config "$SV_BASE_CONFIG" \
    --extra-epochs 50 --label lam1e-3 \
    --joint-limit-weight 1e-3 \
    --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
    --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
    --out configs_runs/singleview_lam1e-3.json
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

Check the arithmetic printed for **every** lambda before submitting:

```
[prepare] checkpoint epoch: 354
[prepare] checkpoint kind : singleview (from the state dict)
[prepare] resume from epoch 355, train 50 epoch(s) -> training.num_epochs = 405
[prepare] arm: CONSTRAINED (w=0.001)
[prepare] model/data cross-check: 55 joints on both sides — OK
[patch]   save_checkpoint_every: 5 -> 2
```

The four `num_epochs` values must be identical, and each `w` must match its
filename. The training sbatch checks both, but fixing it here takes seconds.

```bash
grep -H '"num_epochs"\|joint_limit_regularization' configs_runs/singleview_lam*.json
```

### 2. Pre-flight (login node) — do not skip

```bash
python scripts/prior_study/preflight_study.py \
    --config configs_runs/singleview_lam1e-4.json --reference "$SV_REF" \
    --config configs_runs/singleview_lam1e-3.json --reference "$SV_REF" \
    --config configs_runs/singleview_lam1e-2.json --reference "$SV_REF" \
    --config configs_runs/singleview_lam1e-1.json --reference "$SV_REF"
```

(`prepare_lambda_sweep.sh` prints this command with the paths filled in.)

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

**Score the reference first.** It needs no training, so it runs immediately and
in parallel with everything else — and its violation table decides whether the
sweep can find anything at all. If the reference is already inside the authored
ranges, the hinge is near-zero by construction at every lambda.

```bash
sbatch --array=0 --account=rwth2151 --export=ALL,SV_REF \
    hpc_files/rwth/run_prior_study_eval.sbatch

# when it lands:
column -s, -t prior_study_results/sv_reference/analysis/limit_violations.csv | head -20
```

Then the sweep. `%2` throttles to two concurrent tasks — each takes a whole
4-GPU node, and a mistake caught on the first task costs half as much:

```bash
jid=$(sbatch --parsable --array=0-3%2 --account=rwth2151 \
        hpc_files/rwth/run_prior_study_train.sbatch)
echo "training array: $jid"

sbatch --array=1-4 --dependency=afterany:$jid --account=rwth2151 \
    --export=ALL,SV_REF,EXTRA_EPOCHS=50 \
    hpc_files/rwth/run_prior_study_eval.sbatch
```

`afterany`, not `afterok`: an arm killed at the 24 h wall still left periodic
checkpoints, and `run_prior_study.sh` falls back to the newest one when
`best_model.pth` was never written.

One lambda only: `sbatch --array=1 ...`. The multi-view sweep later:
`sbatch --array=4-7 ...` (train) and `--array=6-9` (eval).

Monitor: `squeue --me` / `tail -f logs/jl_train_<jobid>_<task>.out`

### 3b. Qualitative renders

The tables say *how much* the violation rate moved; they do not say whether the
result looks like a stick insect.

**Render the exported poses, not raw video.** `export_poses.py` already ran every
arm over the test split and wrote `prior_study_results/<arm>/clip_*.npz`. Those
are the predictions behind every MPJPE / PCK / violation number in the tables, in
the same frame order for every arm. Rendering them means the pictures and the
numbers describe one thing — and it costs minutes on one GPU, with no checkpoint,
dataset or backbone loaded. Re-running inference on raw clips would answer a
different question and line up with nothing.

```bash
# 1. Pick the windows ONCE, from the REFERENCE arm (login node, CPU, seconds).
#    Every arm must render the same frames or the grid is not a comparison.
python scripts/prior_study/pick_segments.py \
    --npz prior_study_results/sv_reference/clip_sv_reference.npz \
    --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
    --segments 3 --length 300 --select worst \
    --out prior_study_results/renders/segments.json

# 2. Render every arm over those windows.
sbatch --array=0-4 --account=rwth2151 hpc_files/rwth/run_prior_study_render.sbatch

# 3. Stack into one comparison video per window (login node, ffmpeg).
bash scripts/prior_study/stack_renders.sh
```

**The earlier `w = 100` arm is array task 5.** Its poses predate the sweep and
live in the flat pre-sweep layout `prior_study_results/sv_constrained/`, so it
gets its own task rather than a `lam<L>/` root — which means the two arms that
exist before any sweep training can be compared today:

```bash
sbatch --array=5 --account=rwth2151 hpc_files/rwth/run_prior_study_render.sbatch
bash scripts/prior_study/stack_renders.sh     # -> reference | lambda = 100
```

It is **not epoch-matched to the sweep**: it continued the reference for the few
epochs its old 12 h budget allowed (epoch 386 → 394, and `best_model.pth` was
never written so the newest periodic checkpoint was used), while every sweep arm
gets 50. Its burned-in label says so, and `stack_renders.sh` prints a warning
whenever it shares a grid with the sweep arms. Read it against `sv_reference`,
which it was continued from; read the sweep arms against each other.

`ARM`, `NPZ` and `LABEL` override any task's defaults, for an arm stored
somewhere else:

```bash
ARM=lam100_rerun NPZ=some/other/clip.npz LABEL="lambda = 100 (rerun)" \
  sbatch --array=5 --account=rwth2151 --export=ALL,ARM,NPZ,LABEL \
    hpc_files/rwth/run_prior_study_render.sbatch
```

**`--select worst`** ranks windows by how far the *reference* strays outside the
authored ranges and takes the top ones. A window where the reference was already
in range cannot show the prior doing anything, however good the prior is —
picking windows at random is the most common way a qualitative comparison comes
back "looks the same" for a reason that has nothing to do with the prior. Run
`--select even` alongside it for an unbiased look.

If `pick_segments.py` reports that the reference never leaves the authored ranges
anywhere in the clip, stop: no render can show a correction, and the sweep's
answer is already in the violation table.

**Cameras.** The default `--camera orbit` is a fixed camera on the *root-centred*
mesh: the predicted translation is dropped, so the animal stays in frame and only
its articulation varies between panels. That is what a joint-limit prior changes,
and holding the body still is what makes two arms comparable frame by frame. One
distance is computed across all segments and reused for every arm, so identical
poses cannot look different because a panel zoomed. `CAMERA=predicted` uses the
sidecar camera and the predicted translation — the model's own view, better for
checking the fit still lands on the animal, worse for judging pose.

**The HUD.** Every frame carries the arm label, the frame index, and how many of
the 129 authored joint-axes are out of range *in that frame*, plus the worst
offender and its overshoot. Five panels of a stick insect are otherwise very hard
to tell apart, and the eye invents differences the numbers do not support. The
counter is the per-frame version of what `limit_violations.csv` aggregates, so
the video and the table can be read against each other.

Output:

```
prior_study_results/renders/
  segments.json                       <- the shared windows
  sv_reference/seg00_f012345.mp4      <- one MP4 per window per arm
  sv_reference/render.json            <- provenance + mean axes out of range
  lam1e-4/ lam1e-3/ lam1e-2/ lam1e-1/
  comparison/seg00_f012345_compare.mp4  <- all arms, identical frames, labelled
```

Two arms side by side (easier to judge than five):
`ARMS="sv_reference lam1e-2" bash scripts/prior_study/stack_renders.sh`

**If a `predicted`-camera render comes out at visibly the wrong size**, the npz
cannot record which vertex-scaling branch the trainer used (`use_ue_scaling` vs
the `mesh_scale` head). `SCALING=ue` is the other option. The `orbit` camera
auto-frames, so it is unaffected.

### 4. Results

```
runs/singleview_lam{1e-4,1e-3,1e-2,1e-1}/
  checkpoints/  best_model.pth, checkpoint_epoch_*.pth, config.json
  plots/  visualizations/

prior_study_results/
  sv_reference/                             <- scored once, reused by every lambda
    arm.json                                <- provenance for compare_arms.py
    analysis/baseline_summary.md            <- per-arm tables
    analysis/limit_violations.csv           <- the "before" number
    analysis/joint_angle_distributions.png  <- dashed red = authored limits; features the
                                               COXAE, not analyze_baseline_pose's default
                                               pretarsi (those are free in this .pkl)
    analysis/range_of_motion.png
    benchmark_*/benchmark_report.txt        <- MPJPE mm + PCK
  lam1e-4/
    sv_reference -> ../sv_reference         <- symlink, so compare_arms sees the pair
    sv_constrained/                         <- same layout as above
    comparison/comparison.md                <- reference vs this lambda
  lam1e-3/ lam1e-2/ lam1e-1/                <- same
  sweep/
    sweep.md                                <- HEADLINE: the lambda curve
    sweep.csv                               <- one row per (mode, lambda)
  renders/
    segments.json                           <- the windows every arm renders
    sv_reference/seg00_f012345.mp4          <- one MP4 per window, HUD burned in
    sv_reference/render.json                <- provenance + axes out of range
    lam1e-4/ ...
    comparison/seg00_f012345_compare.mp4    <- all arms, identical frames
```

Each lambda gets its own root because `compare_arms.py` builds its delta table
from the fixed pair names (`sv_reference`, `sv_constrained`); giving every arm
that label inside a per-lambda folder keeps it working unchanged. `sweep.md` is
rebuilt from all of them by the eval job.

Deltas are `constrained - reference`, annotated per-metric (lower is better for
violations and MPJPE, higher for PCK).

**Win condition:** a lambda that lowers the violation rate while leaving MPJPE
and PCK flat. A lambda that lowers violations *and* accuracy is the prior
overriding the data — expect that at the top of the sweep, if anywhere. Flat
everywhere, with a flat reference violation table, means the model was already
inside the authored ranges.

---

## Quick local checks (no GPU, no data)

```bash
# analysis engine end-to-end on a synthetic clip with a deliberate out-of-range joint
python scripts/prior_study/analyze_baseline_pose.py --self-test --out /tmp/selftest

# epoch arithmetic + mode handling without reading a checkpoint
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint /dev/null --assume-epoch 354 --extra-epochs 50 --mode multiview \
    --base-config smal_fitter/neuralSMIL/configs/examples/multiview_SMILySTICKS_3D_ViT_Large_AUG_FIXED.json \
    --label dryrun --out /tmp/dryrun.json --no-validate

# sbatch + script syntax
bash -n hpc_files/rwth/run_prior_study_train.sbatch
bash -n hpc_files/rwth/run_prior_study_eval.sbatch
bash -n hpc_files/rwth/run_prior_study_render.sbatch
bash -n scripts/prior_study/run_prior_study.sh
bash -n scripts/prior_study/prepare_lambda_sweep.sh
bash -n scripts/prior_study/stack_renders.sh

# segment picking on the real reference export — CPU, no GPU, seconds
python scripts/prior_study/pick_segments.py \
    --npz prior_study_results/sv_reference/clip_sv_reference.npz \
    --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
    --segments 3 --length 300 --out /tmp/segments.json

# the stacking filtergraph, on synthetic clips — no GPU, no checkpoints
mkdir -p /tmp/r/{sv_reference,lam1e-2}
for a in sv_reference lam1e-2; do
  ffmpeg -y -loglevel error -f lavfi -i "testsrc=size=512x512:rate=30:duration=2" \
      -pix_fmt yuv420p /tmp/r/$a/seg00_f000000.mp4
done
RENDER_ROOT=/tmp/r ARMS="sv_reference lam1e-2" bash scripts/prior_study/stack_renders.sh
```

---

## If the prior shows no effect

In order of likelihood, before concluding the hypothesis is wrong:

1. **Nothing was violating those joints anyway.** Check the violation table in
   each `*_reference` arm. If the model was already inside the authored ranges,
   the hinge is near-zero by construction and there is nothing for it to fix.
   This is the single most informative diagnostic, and it costs no GPU time
   beyond the reference arms.
2. **Every lambda in this sweep is far below the previous `w = 100`.** At
   79.6 % coverage the effective weight is `lambda × 0.80`, so `1e-1` is still
   ~1000× weaker than the setting the earlier study used. Each training task
   prints the effective weight as a ratio against `keypoint_3d`; if that ratio
   is ≪ 1 at every point, the sweep is measuring a prior that cannot move the
   pose — and the answer is to sweep *upward* (`LAMBDAS="1 10 100 1000"`), not
   to conclude the hypothesis is wrong.
3. **50 epochs at a decayed LR.** The curriculum is inherited from the original
   run, so the tail LR may be too small to move the pose much even when the
   hinge is firing. Check the LR the training log reports at resume.
4. **The hinge is firing but losing.** Compare its magnitude against
   `keypoint_2d`/`keypoint_3d` in the training log before changing anything
   else — that comparison distinguishes "too weak" from "not firing".
5. **Multi-view has less to gain by construction.** Multi-view already resolves
   depth ambiguity geometrically, so a small multi-view delta beside a large
   single-view one is a *result*, not a failure. That is why this sweep is
   single-view only.

If the tables move but the renders look identical — or the reverse — trust the
renders for *plausibility* and the tables for *magnitude*. A prior that fixes a
handful of frames deep in the test split will show in the violation rate and
never appear in a 300-frame window; a prior that freezes the legs will be obvious
on screen while the mean violation rate looks fine. Because the renders come from
the same `.npz` the tables are computed from, a disagreement between them is
always about *which frames you are looking at*, never about two different models
— check the per-frame counter in the HUD against `limit_violations.csv`.
