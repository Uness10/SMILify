#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c23g
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --job-name=sv_constrained
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# RWTH CLAIX-2023 (c23g) — single-node, 4-GPU DDP continuation of the
# SINGLE-VIEW SMIL regressor, resuming from a downloaded checkpoint.
# CONSTRAINED arm of the joint-limit prior study
# (joint_limit_regularization > 0 against the authored per-joint limits).
#
# Ported from hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch.
# Differences from the JSC version, and why:
#   --partition=c23g       CLAIX-2023 GPU partition (JSC: dc-gpu)
#   --cpus-per-task=96     a c23g node is 2x Xeon 8468 = 96 cores (JSC: 128)
#   --account              RWTH accounts are per-project (rwth####/p0######) and
#                          there is no sensible default, so ACCOUNT is REQUIRED
#                          and passed with `sbatch --account=` (see below)
#   4x H100 80GB           vs 4x A100 40GB; --nproc_per_node stays 4, so the DDP
#                          effective batch size is unchanged
#
# ACCOUNT is not hardcoded. Submit with --account, or export ACCOUNT and let the
# guard below fail loudly rather than queue against the wrong project.
#
# Prepare the config on the LOGIN NODE first (it reads the checkpoint's epoch and
# turns "N more epochs" into the absolute end epoch the trainer expects):
#
#   mkdir -p logs configs_runs
#   conda activate "$HPCWORK/conda_envs/pytorch3d"
#   python scripts/prior_study/prepare_resume_config.py \
#       --checkpoint singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth \
#       --base-config singleview_SMILySTICKS_3D_ViT.json \
#       --extra-epochs 10 --label constrained \
#       --joint-limit-weight 100.0 \
#       --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
#       --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
#       --out configs_runs/singleview_constrained.json
#
# Then submit FROM THE REPO ROOT (the #SBATCH --output path resolves at submit
# time, so logs/ must already exist):
#
#   sbatch --account=<your_project> \
#     hpc_files/rwth/run_singleview_training_RWTH.sbatch
#
# Override the config without editing this file:
#   CONFIG=configs_runs/singleview_other.json \
#     sbatch --account=<proj> --export=ALL,CONFIG --job-name=sv_other \
#       hpc_files/rwth/run_singleview_training_RWTH.sbatch
#
# Monitor:  squeue --me   /   tail -f logs/<jobname>_<jobid>.out
# Cancel :  scancel <jobid>
#
# NOTE: intentionally no `set -u`. Conda's activate hooks (libblas_mkl_activate.sh)
# reference unset vars like MKL_INTERFACE_LAYER and abort under set -u.
set -eo pipefail
# ---------------------------------------------------------------------------

# --- account guard -----------------------------------------------------------
# SLURM_JOB_ACCOUNT is set by the scheduler once the job is accepted, so an
# empty value here means neither --account nor a default association applied.
if [[ -z "${SLURM_JOB_ACCOUNT:-}" ]]; then
    echo "ERROR: no SLURM account. Submit with:" >&2
    echo "  sbatch --account=<your_rwth_project> $0" >&2
    exit 1
fi

# Backbone weights must already be cached (hpc_files/download_backbone_weights.py
# on the login node); c23g compute nodes have no outbound network.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
# RWTH: $HPCWORK is the large per-user filesystem; $HOME has a hard quota that a
# conda env plus the 21 GB dataset will blow through.
ENV_PREFIX="${ENV_PREFIX:-$HPCWORK/conda_envs/pytorch3d}"

# --- activate conda ----------------------------------------------------------
# Batch jobs don't reliably source ~/.bashrc, so locate conda.sh explicitly.
# Override with CONDA_SH=/path/to/conda.sh if this guess is wrong.
if [[ -z "${CONDA_SH:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    else
        for base in "$HPCWORK/miniconda3" "$HPCWORK/miniforge3" \
                    "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
            [[ -f "$base/etc/profile.d/conda.sh" ]] && { CONDA_SH="$base/etc/profile.d/conda.sh"; break; }
        done
    fi
fi
if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: could not locate conda.sh. Submit with:" >&2
    echo "  CONDA_SH=/path/to/conda/etc/profile.d/conda.sh sbatch --export=ALL,CONDA_SH ..." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_PREFIX"

cd "$REPO_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "  cluster: RWTH CLAIX-2023 | partition ${SLURM_JOB_PARTITION:-?} | account $SLURM_JOB_ACCOUNT"
echo "  repo   : $REPO_DIR"
echo "  env    : $CONDA_PREFIX"
echo "  python : $(which python)"
python -c "import torch; print(f'  torch  : {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s) visible')"

# --- config ------------------------------------------------------------------
CONFIG="${CONFIG:-configs_runs/singleview_constrained.json}"
echo "  config : $CONFIG"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "       Run scripts/prior_study/prepare_resume_config.py on the login node first." >&2
    exit 1
fi

# --- pre-flight: the num_epochs trap ----------------------------------------
# training.num_epochs is an ABSOLUTE end epoch and the trainer resumes at
# checkpoint_epoch + 1. If num_epochs <= that, the loop body never executes and
# the job "succeeds" after training nothing. Catch it before burning the GPUs.
#
# Also asserts the constrained arm is actually constrained: a weight of 0 here
# means prepare_resume_config.py was run without --joint-limit-weight and the
# job would silently reproduce the baseline.
python - "$CONFIG" <<'PY' || exit 1
import json, sys, os, pickle, torch
import numpy as np

cfg = json.load(open(sys.argv[1]))
tr = cfg.get("training", {})
ckpt_path = tr.get("resume_checkpoint")
num_epochs = tr.get("num_epochs")
if not ckpt_path:
    sys.exit("ERROR: training.resume_checkpoint is null — this job is meant to RESUME.")
if not os.path.isfile(ckpt_path):
    sys.exit(f"ERROR: resume_checkpoint does not exist: {ckpt_path}")
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
start = int(ck.get("epoch", -1)) + 1
print(f"  resume checkpoint : {ckpt_path} (epoch {start - 1})")
print(f"  start epoch       : {start}")
print(f"  num_epochs (end)  : {num_epochs}")
if num_epochs is None or num_epochs <= start:
    sys.exit(f"ERROR: num_epochs ({num_epochs}) <= start epoch ({start}) — this would train 0 epochs.\n"
             f"       Re-run prepare_resume_config.py with --extra-epochs N.")
print(f"  -> will train {num_epochs - start} epoch(s)")

w = (cfg.get("loss_curriculum", {}).get("base_weights", {}) or {}).get("joint_limit_regularization", 0.0)
print(f"  joint_limit_regularization: {w}")
if not w:
    sys.exit("ERROR: joint_limit_regularization is 0 — this is the CONSTRAINED arm.\n"
             "       Re-run prepare_resume_config.py with --joint-limit-weight 100.0,\n"
             "       or submit with CONFIG=<the config you actually mean>.")

# The penalty is a silent no-op unless the model .pkl carries usable limits;
# smil_image_regressor raises for that case, but only after the GPUs are held.
smal_file = (cfg.get("smal_model", {}) or {}).get("smal_file")
print(f"  smal_file         : {smal_file}")
if not smal_file or not os.path.isfile(smal_file):
    sys.exit(f"ERROR: smal_model.smal_file missing on disk: {smal_file}")
dd = pickle.load(open(smal_file, "rb"), encoding="latin1")
jl = dd.get("joint_limits")
if jl is None:
    sys.exit(f"ERROR: {smal_file} has no 'joint_limits' key, but the limit weight is {w}.\n"
             f"       The hinge would be a guaranteed no-op. Use the authored .pkl.")
jl = np.asarray(jl, dtype=np.float64)
n_j = len(dd["J_names"])
if jl.shape != (n_j, 3, 2):
    sys.exit(f"ERROR: joint_limits shape {jl.shape}, expected {(n_j, 3, 2)}.")
pi = np.pi
free = (jl[..., 0] <= -pi + 1e-4) & (jl[..., 1] >= pi - 1e-4)
n_con = int((~free).sum())
if n_con == 0:
    sys.exit("ERROR: every joint/axis is wide open ([-pi,pi] = free) — the hinge can never fire.")
# Exclude the root: LimitPrior pins J_names[0] to [0,0] and the fitter drops it
# via the [3:] slice, so it contributes nothing to the penalty.
n_con_nonroot = int((~free[1:]).sum())
n_axes_nonroot = free[1:].size
print(f"  authored limits   : {n_con} constrained axes ({n_con_nonroot} non-root) "
      f"across {int((~free).any(axis=1).sum())} joints")
print(f"  NOTE: _joint_limit_penalty is a mean over ALL {n_axes_nonroot} non-root axes, "
      f"so only {100.0 * n_con_nonroot / n_axes_nonroot:.0f}% of the average is "
      f"constrained -> effective prior strength is diluted accordingly.")

bs = tr.get("batch_size")
print(f"  batch_size {bs} per GPU x 4 GPUs -> effective {bs * 4 if bs else '?'} "
      f"(train_smil_regressor.py:1737 — batch_size is PER PROCESS under DDP)")
print(f"  checkpoints -> {cfg.get('output', {}).get('checkpoint_dir')}")
PY

# --- launch ------------------------------------------------------------------
# Single node: --standalone gives torchrun its own rendezvous on localhost, so no
# MASTER_ADDR/MASTER_PORT/rdzv plumbing is needed. The single-view trainer detects
# RANK/LOCAL_RANK/WORLD_SIZE (is_torchrun_launched) and skips its own mp.spawn
# path, so --num_gpus is intentionally not passed.
#
# Run torchrun directly rather than under srun: the batch script already runs on
# the allocated node with all 4 GPUs visible, and srun+torchrun double-launching
# is a common source of duplicated ranks.
TORCHRUN="torchrun"
command -v torchrun >/dev/null 2>&1 || TORCHRUN="python -m torch.distributed.run"

$TORCHRUN \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m smal_fitter.neuralSMIL.train_smil_regressor --config "$CONFIG"

echo "Done — $(date)"
