# Single-view prior study — unconstrained arm

The single-view counterpart of [`README.md`](README.md) (which is multi-view).
This folder's job here is the **"before" (unconstrained) column**: resume the
downloaded single-view checkpoint for a few more epochs with
`joint_limit_regularization = 0`, then measure accuracy *and* pose realism so a
constrained run drops straight into a side-by-side table.

---

## Read this first — two things that will bite you

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
4 in the prepared config to match the original regime, or accept 4× and keep it
**identical across both arms** so the comparison stays clean. The training job
prints the arithmetic in its pre-flight block before it starts.

**3. `3D_model_prep/SMILy_STICK.pkl` has no authored `joint_limits`.**
Only `3D_model_prep/OmniAnt_25PCs_joint_limited.pkl` does (added in `978c69b`:
99 of 165 axes constrained across 33 of 55 joints). So:

- the **unconstrained** run below works today;
- the **constrained** run on the stick model does not exist yet — the trainer
  raises if `joint_limit_regularization > 0` and the `.pkl` has no usable limits.
  Author the stick limits in Blender (`docs/joint_limits_user_guide.md`) first.
  The angle distributions this study produces are exactly the data you'd use to
  pick those ranges.

---

## Files

| File | What it does |
|---|---|
| `prepare_resume_config.py` | Derives the run config from the checkpoint's `config.json`: sets `resume_checkpoint`, computes the absolute `num_epochs`, sets the joint-limit weight, isolates output dirs, validates via `load_config`. |
| `export_singleview_poses.py` | Runs a single-view checkpoint over the **same test split the benchmark scores** and writes the `.npz`/`.json` pair `analyze_baseline_pose.py` expects. (Needed because `run_singleview_inference --export_animation` only accepts a folder/video, not an HDF5.) |
| `run_singleview_study.sh` | benchmark → export → analyse, into `prior_study_results/singleview_<label>/`. |
| `../../hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch` | 1 node × 4 GPUs, `torchrun_jsc`, resumes from the checkpoint. |
| `../../hpc_files/sc_venv_pytorch3d/run_singleview_study_JURECA.sbatch` | 1 GPU, runs the study wrapper. |
| `analyze_baseline_pose.py` | Shared with the multi-view arm — unchanged. |

> **These sbatch scripts use conda, not the JSC `sc_venv` setup.** The
> `activate.sh` / `torchrun_jsc` combination in
> `run_multiview_training_JURECA.sbatch` comes from the **`jsc-hpc`** branch and
> is not in this working tree — sourcing it fails and `torchrun_jsc: command not
> found` follows. These scripts activate the conda env directly and use plain
> `torchrun --standalone`, which is all a single node needs.
>
> Defaults are `ENV_PREFIX=/p/project1/cias-7/anouar1/conda_envs/pytorch3d` and
> `--account=cias-7`. Override without editing:
> `ENV_PREFIX=... sbatch --export=ALL,ENV_PREFIX ...`

---

## Run it on JURECA-DC

### 0. One-time setup on the login node

```bash
cd /p/scratch/share/anouar1/SMILify
conda activate /p/project1/cias-7/anouar1/conda_envs/pytorch3d
python hpc_files/download_backbone_weights.py   # ViT-L weights; compute nodes are offline
mkdir -p logs configs_runs
```

If the job later reports `could not locate conda.sh`, batch jobs aren't picking
up conda from your `~/.bashrc`. Pass it explicitly:

```bash
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh" \
  sbatch --export=ALL,CONDA_SH hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch
```

### 1. Download the checkpoint folder

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/15sjoulpXQmkbXopS1-ULp1mrWZeQzq4k \
      -O singleview_SMILySTICKS_3D_ViT_checkpoints
ls singleview_SMILySTICKS_3D_ViT_checkpoints/
```

You want `best_model.pth` (or a specific `checkpoint_epoch_*.pth`) **and**
`config.json` — the trainer writes the resolved config next to the checkpoints,
and reusing it guarantees the architecture, split seed and loss curriculum match
the run being continued. If `config.json` is missing, `prepare_resume_config.py`
falls back to the config block embedded in the `.pth` and tells you which
sections it had to guess.

The dataset (`SMILySTICKS_centred_reprojected_FIXED.h5`, ~21 GB) goes in the
repo root per `GETTING_STARTED.md`:

```bash
gdown 1wlVPe1ZwGmFkS9KhLODpIzvfi3DsqgQL -O SMILySTICKS_centred_reprojected_FIXED.h5
```

### 2. Prepare the continuation config (login node)

```bash
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth \
    --base-config singleview_SMILySTICKS_3D_ViT_checkpoints/config.json \
    --extra-epochs 10 \
    --label unconstrained \
    --joint-limit-weight 0.0 \
    --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
    --smal-file 3D_model_prep/SMILy_STICK.pkl \
    --out configs_runs/singleview_unconstrained.json
```

It prints the epoch arithmetic — check it before submitting:

```
[prepare] checkpoint epoch: 240
[prepare] resume from epoch 241, train 10 epoch(s) -> training.num_epochs = 251 (was 250)
[prepare] arm: UNCONSTRAINED
[prepare] outputs -> runs/singleview_unconstrained/
[prepare] load_config round-trip: OK
```

### 3. Submit training + evaluation as a chain

```bash
jid=$(sbatch --parsable hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch)
echo "training job: $jid"

LABEL=unconstrained \
CHECKPOINT=runs/singleview_unconstrained/checkpoints/best_model.pth \
  sbatch --dependency=afterok:$jid --export=ALL,LABEL,CHECKPOINT \
    hpc_files/sc_venv_pytorch3d/run_singleview_study_JURECA.sbatch
```

Monitor / cancel:

```bash
squeue --me
tail -f logs/sv_unconstrained_${jid}.out
scancel $jid
```

### 4. Results

```
runs/singleview_unconstrained/
  checkpoints/  best_model.pth, checkpoint_epoch_*.pth, config.json
  plots/  visualizations/  visualizations_train/
prior_study_results/singleview_unconstrained/
  analysis/baseline_summary.md            <- headline tables
  analysis/joint_angle_distributions.png
  analysis/range_of_motion.png
  analysis/per_axis_stats.csv, magnitude_stats.csv
  analysis/trajectories/*.png
  benchmark_singleview_*/benchmark_report.txt   <- MPJPE mm + PCK
  clip_unconstrained.npz / .json          <- exported poses
```

Every table carries a `--label` column, so the constrained arm concatenates
directly.

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
```

---

## Later: the constrained arm

Once the stick model carries authored limits, the whole change is one flag:

```bash
python scripts/prior_study/prepare_resume_config.py \
    --checkpoint singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth \
    --base-config singleview_SMILySTICKS_3D_ViT_checkpoints/config.json \
    --extra-epochs 10 --label constrained \
    --joint-limit-weight 100.0 \
    --smal-file 3D_model_prep/SMILy_STICK_joint_limited.pkl \
    --out configs_runs/singleview_constrained.json
```

(`w_limit=100` is the value that removed every violation at no surface cost in
the `fitter_3d` ant study — a starting point, not a tuned value for this model.)

Then re-run steps 3–4 with `LABEL=constrained` and diff
`prior_study_results/singleview_{unconstrained,constrained}/`.

Both arms start from the **same** checkpoint and train the **same** number of
epochs, so the only difference is the prior.
