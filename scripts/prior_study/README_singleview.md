# Single-view prior study — constrained run (RWTH CLAIX-2023)

The single-view counterpart of [`README.md`](README.md) (which is multi-view).
Resume the downloaded single-view checkpoint for 10 more epochs with
`joint_limit_regularization > 0` against the authored per-joint limits, then
measure accuracy *and* pose realism.

**Runs on RWTH CLAIX-2023 (`c23g`, 4× H100).** The JSC/JURECA scripts under
`hpc_files/sc_venv_pytorch3d/` are superseded — see [Cluster port](#cluster-port).

**Scope note:** this study now has a **single arm**. The unconstrained
fine-tuning arm was dropped; the reference point is the downloaded checkpoint
itself, not a separately fine-tuned baseline. Anything below that reads as a
two-column comparison is stale.

---

## Read this first — three things that will bite you

**1. `training.num_epochs` is an absolute END epoch, not a count.**
On resume the trainer sets `start_epoch = checkpoint["epoch"] + 1`
(`train_smil_regressor.py:1185`) and loops `range(start_epoch, num_epochs)`. A
checkpoint at epoch 240 with `num_epochs: 250` gives you 9 more epochs; with
`num_epochs: 10` it trains **zero** and exits looking successful.
`prepare_resume_config.py` reads the epoch out of the checkpoint and writes
`num_epochs = epoch + 1 + N` for you, and the training sbatch re-checks it
before allocating GPUs.

**2. `batch_size` is PER PROCESS under DDP.**
`train_smil_regressor.py:1737` prints `Total effective batch size: batch_size *
world_size`. On 4 GPUs the checkpoint's batch size is quadrupled, so the
continuation runs at a different effective batch (and therefore a different
noise scale) than the run that produced the checkpoint — while inheriting that
run's learning-rate curriculum unchanged. Either divide `training.batch_size` by
4 in the prepared config to match the original regime, or accept 4× knowingly.
The training job prints the arithmetic in its pre-flight block before it starts.

**3. The penalty is averaged over ALL non-root axes, not the constrained ones.**
`_joint_limit_penalty` is a `torch.mean` over the full `(N, N_POSE, 3)` hinge
(`smil_image_regressor.py:1445`). Free axes contribute exact zeros to that mean,
so the *fraction* of authored axes directly scales the effective prior strength.
With the current 30-of-162 non-root coverage, `w_limit = 100` delivers roughly
**4.3× less** pull than the same number would on a densely authored set. The
training pre-flight prints the coverage percentage; read it before concluding
"the prior does nothing."

---

## The model file

**`3D_model_prep/SMILy_STICK_limits_authored.pkl`** — this is the one to use.

- `3D_model_prep/SMILy_STICK.pkl` has no `joint_limits` at all; the trainer
  raises rather than training a silent no-op.
- `3D_model_prep/SMILy_STICK_authored.pkl` is an earlier, densely authored
  variant (132/165 axes). Geometry is identical to the file above — only
  `joint_limits` differs — so the two are interchangeable as far as the mesh,
  skinning and joint regressor are concerned.

Verified contents of `SMILy_STICK_limits_authored.pkl`:

| Check | Result |
|---|---|
| `joint_limits` shape | `(55, 3, 2)`, float64, radians — matches `len(J_names)` |
| `min <= max`, all finite | pass (`LimitPrior._ranges_from_joint_limits` validation) |
| all axes within `[-π, π]` | pass — no representation-ambiguous authoring |
| rest pose inside limits | pass — 0 ∈ `[min, max]` on every axis |
| Constrained axes | **33 of 165**, across **11 of 55** joints (132 axes at ±π = free) |
| Constrained joints | `l_{1,2,3}_co_{r,l}` (coxae), `w_1_{r,l}`, `w_2_r`, `b_h` |
| Root `b_t` | `[0,0]` on all 3 axes — correct; `LimitPrior` pins it, the fitter drops it via the `[3:]` slice |
| `_check_bounds_can_bite` | passes — not wide-open, so the hinge is not a silent no-op even in `6d` |
| Non-root constrained ranges | 20°–120°, median 57° |

> **Coverage is deliberate and narrow.** Limits are authored on the coxae, wings
> and head only; the trochanter/femur/tibia/tarsus chain and the
> antennae/mandibles are intentionally left free. So the prior constrains
> **proximal leg placement**, not distal leg articulation — violations in the
> femur/tibia/tarsus chain are not penalised and should not be expected to
> improve. See also caveat 3 above on how coverage dilutes the effective weight.

---

## Files

| File | What it does |
|---|---|
| `prepare_resume_config.py` | Derives the run config from the checkpoint's `config.json`: sets `resume_checkpoint`, computes the absolute `num_epochs`, sets the joint-limit weight, isolates output dirs, validates via `load_config`. |
| `export_singleview_poses.py` | Runs a single-view checkpoint over the **same test split the benchmark scores** and writes the `.npz`/`.json` pair `analyze_baseline_pose.py` expects. (Needed because `run_singleview_inference --export_animation` only accepts a folder/video, not an HDF5.) |
| `run_singleview_study.sh` | benchmark → export → analyse, into `prior_study_results/singleview_<label>/`. |
| `../../hpc_files/rwth/run_singleview_training_RWTH.sbatch` | `c23g`, 1 node × 4 H100, `torchrun --standalone`, resumes from the checkpoint. |
| `../../hpc_files/rwth/run_singleview_study_RWTH.sbatch` | `c23g`, 1 GPU, runs the study wrapper. |
| `analyze_baseline_pose.py` | Shared with the multi-view arm — unchanged. |

### Cluster port

The RWTH scripts were ported from `hpc_files/sc_venv_pytorch3d/run_singleview_*_JURECA.sbatch`.
What changed and why:

| | JURECA-DC | RWTH CLAIX-2023 |
|---|---|---|
| `--partition` | `dc-gpu` | `c23g` |
| GPUs/node | 4× A100 40GB | 4× H100 80GB — `--nproc_per_node` stays **4**, so effective batch is unchanged |
| `--cpus-per-task` | 128 | **96** (2× Xeon 8468) — 24 for the 1-GPU study job |
| `--account` | hardcoded `cias-7` | **not hardcoded** — RWTH accounts are per-project; the script aborts unless you pass `--account=` |
| env / data root | `/p/project1/...` | `$HPCWORK` (`$HOME` quota won't hold a conda env + 21 GB dataset) |
| launcher | `torchrun --standalone` | unchanged (never used `torchrun_jsc`; that lives on the `jsc-hpc` branch) |
| compute-node network | none | none — `HF_*_OFFLINE=1` and the login-node weight prefetch both stay |

Two latent bugs were fixed in the port rather than carried over:

- The study script's `SMAL_FILE` defaulted to `SMILy_STICK.pkl` (**no limits**),
  so `analyze_baseline_pose.load_limits` found nothing and silently skipped the
  limit-violation table — the headline number of the study. It now defaults to
  the authored `.pkl`, and the job prints the limits source before running.
- The JSC study script used `--account=ias-7` while training used `cias-7`.
  Moot now that the account is supplied at submit time.

`LIMITS` stays unset on purpose: `load_limits` prefers an explicit `--limits`
file but falls back to the `joint_limits` key inside `SMAL_FILE`, so pointing at
the authored `.pkl` is sufficient.

---

## Run it on RWTH CLAIX-2023

Substitute your project for `<proj>` throughout (`rwth####` / `p0######`;
`squeue --me --format="%a"` or your project portal shows it).

### 0. One-time setup on the login node

```bash
cd "$HPCWORK/SMILify"
conda activate "$HPCWORK/conda_envs/pytorch3d"
python hpc_files/download_backbone_weights.py   # ViT-L weights; c23g nodes are offline
mkdir -p logs configs_runs
```

If a job later reports `could not locate conda.sh`, batch jobs aren't picking up
conda from your `~/.bashrc`. Pass it explicitly:

```bash
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh" \
  sbatch --account=<proj> --export=ALL,CONDA_SH \
    hpc_files/rwth/run_singleview_training_RWTH.sbatch
```

### 1. Checkpoint and dataset

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/15sjoulpXQmkbXopS1-ULp1mrWZeQzq4k \
      -O singleview_SMILySTICKS_3D_ViT_checkpoints
ls singleview_SMILySTICKS_3D_ViT_checkpoints/
```

You want `best_model.pth` (or a specific `checkpoint_epoch_*.pth`) **and**
`config.json` — the trainer writes the resolved config next to the checkpoints,
and reusing it guarantees the architecture, split seed and loss curriculum match
the run being continued.

The dataset (`SMILySTICKS_centred_reprojected_FIXED.h5`, ~21 GB) goes in the repo
root per `GETTING_STARTED.md`. On RWTH keep the repo itself on `$HPCWORK` —
`$HOME` will not hold it.

```bash
gdown 1wlVPe1ZwGmFkS9KhLODpIzvfi3DsqgQL -O SMILySTICKS_centred_reprojected_FIXED.h5
```

### 2. Prepare the run config (login node)

```bash
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth \
    --base-config singleview_SMILySTICKS_3D_ViT.json \
    --extra-epochs 10 --label constrained \
    --joint-limit-weight 100.0 \
    --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
    --out configs_runs/singleview_constrained.json
```

`--base-config` must be **the JSON that produced the checkpoint** — for this run
that is `singleview_SMILySTICKS_3D_ViT.json` (whose `output.checkpoint_dir` is
`singleview_SMILySTICKS_3D_ViT_checkpoints`). Omit the flag and the script
auto-discovers it from that naming convention.

> **Never let it fall back to the checkpoint's embedded config.** That block is
> serialized from the runtime `TrainingConfig`, so it carries stale defaults, not
> what the run used. On job 15521637 it yielded `data_path:
> RealSMILyMouseFalknerFROM3D_no_crop.h5` (a **mouse** dataset) paired with the
> 55-joint stick model, plus `hidden_dim 1024`/`freeze_backbone true` instead of
> the real `512`/`false`. Training ran without error. The fallback is now behind
> `--allow-embedded-config` and off by default, and the script cross-checks the
> model's joint count against the HDF5's `n_joints` before writing anything.

Expected output — check the epoch arithmetic before submitting:

```
[prepare] checkpoint epoch: 240
[prepare] resume from epoch 241, train 10 epoch(s) -> training.num_epochs = 251 (was 250)
[prepare] arm: CONSTRAINED (w=100.0)
[prepare] outputs -> runs/singleview_constrained/
[prepare] smal file : 3D_model_prep/SMILy_STICK_limits_authored.pkl
[prepare] load_config round-trip: OK
```

### 3. Submit training + evaluation as a chain

```bash
jid=$(sbatch --parsable --account=<proj> \
        hpc_files/rwth/run_singleview_training_RWTH.sbatch)
echo "training job: $jid"

sbatch --dependency=afterok:$jid --account=<proj> \
    hpc_files/rwth/run_singleview_study_RWTH.sbatch
```

Both scripts default to the constrained arm, so no `CONFIG`/`LABEL`/`CHECKPOINT`
exports are needed. The training job's pre-flight block aborts before allocating
GPUs if `joint_limit_regularization` is 0, if the `.pkl` carries no usable
`joint_limits`, or if the epoch arithmetic would train zero epochs.

Monitor / cancel:

```bash
squeue --me
tail -f logs/sv_constrained_${jid}.out
scancel $jid
```

### 4. Results

```
runs/singleview_constrained/
  checkpoints/  best_model.pth, checkpoint_epoch_*.pth, config.json
  plots/  visualizations/  visualizations_train/
prior_study_results/singleview_constrained/
  analysis/baseline_summary.md            <- headline tables
  analysis/joint_angle_distributions.png  <- dashed red = authored limits
  analysis/range_of_motion.png
  analysis/per_axis_stats.csv, magnitude_stats.csv
  analysis/trajectories/*.png
  benchmark_singleview_*/benchmark_report.txt   <- MPJPE mm + PCK
  clip_constrained.npz / .json            <- exported poses
```

Every table carries a `--label` column, so additional runs (a higher `w_limit`,
a denser limit set) concatenate directly.

---

## Quick local checks (no GPU, no data)

```bash
# analysis engine end-to-end on a synthetic clip with a deliberate out-of-range joint
python scripts/prior_study/analyze_baseline_pose.py --self-test --out /tmp/selftest

# epoch arithmetic without reading the checkpoint
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint /dev/null --assume-epoch 240 --extra-epochs 10 \
    --base-config smal_fitter/neuralSMIL/configs/examples/getting_started_singleview.json \
    --label dryrun --out /tmp/dryrun.json --no-validate

# sbatch syntax
bash -n hpc_files/rwth/run_singleview_training_RWTH.sbatch
bash -n hpc_files/rwth/run_singleview_study_RWTH.sbatch
```

---

## If the prior shows no effect

In order of likelihood, before concluding the hypothesis is wrong:

1. **Coverage dilution** (caveat 3) — raise `--joint-limit-weight` to ~430 to
   match the gradient scale a densely authored set would give at 100.
2. **Nothing was violating those joints anyway** — check the violation table in
   `analysis/baseline_summary.md`. If the coxae/wings/head were already inside
   their ranges, the hinge is near-zero by construction and the free
   femur/tibia/tarsus chain is where the implausibility lives.
3. **10 epochs at a decayed LR** — the curriculum is inherited from the original
   run, so the tail LR may be too small to move the pose much.
