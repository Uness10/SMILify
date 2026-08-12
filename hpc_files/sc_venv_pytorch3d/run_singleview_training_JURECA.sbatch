#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=cias-7
#SBATCH --partition=dc-gpu
#SBATCH --time=04:00:00
#SBATCH --job-name=sv_unconstrained
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Single-node, 4-GPU DDP continuation of the SINGLE-VIEW SMIL regressor,
# resuming from a downloaded checkpoint. Unconstrained arm of the joint-limit
# prior study (joint_limit_regularization = 0).
#
# Uses the conda env directly (NOT the sc_venv/torchrun_jsc setup from the
# jsc-hpc branch — that venv isn't in this working tree).
#
# Prepare the config on the LOGIN NODE first (it reads the checkpoint's epoch
# and turns "N more epochs" into the absolute end epoch the trainer expects):
#
#   mkdir -p logs configs_runs
#   conda activate /p/project1/cias-7/anouar1/conda_envs/pytorch3d
#   python scripts/prior_study/prepare_resume_config.py \
#       --checkpoint singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth \
#       --base-config singleview_SMILySTICKS_3D_ViT_checkpoints/config.json \
#       --extra-epochs 10 --label unconstrained \
#       --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
#       --out configs_runs/singleview_unconstrained.json
#
# Then submit FROM THE REPO ROOT (the #SBATCH --output path resolves at submit
# time, so logs/ must already exist):
#
#   sbatch hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch
#
# Override without editing this file:
#   CONFIG=configs_runs/singleview_constrained.json \
#     sbatch --export=ALL,CONFIG --job-name=sv_constrained \
#       hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch
#
# Monitor:  squeue --me   /   tail -f logs/<jobname>_<jobid>.out
# Cancel :  scancel <jobid>
#
# NOTE: intentionally no `set -u`. Conda's activate hooks (libblas_mkl_activate.sh)
# reference unset vars like MKL_INTERFACE_LAYER and abort under set -u.
set -eo pipefail
# ---------------------------------------------------------------------------

# Backbone weights must already be cached (hpc_files/download_backbone_weights.py
# on the login node); compute nodes have no outbound network.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
ENV_PREFIX="${ENV_PREFIX:-/p/project1/cias-7/anouar1/conda_envs/pytorch3d}"

# --- activate conda ----------------------------------------------------------
# Batch jobs don't reliably source ~/.bashrc, so locate conda.sh explicitly.
# Override with CONDA_SH=/path/to/conda.sh if this guess is wrong.
if [[ -z "${CONDA_SH:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
    else
        for base in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" \
                    /p/project1/cias-7/anouar1/miniconda3 /p/project1/cias-7/anouar1/miniforge3; do
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
echo "  repo   : $REPO_DIR"
echo "  env    : $CONDA_PREFIX"
echo "  python : $(which python)"
python -c "import torch; print(f'  torch  : {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s) visible')"

# --- config ------------------------------------------------------------------
CONFIG="${CONFIG:-configs_runs/singleview_unconstrained.json}"
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
python - "$CONFIG" <<'PY' || exit 1
import json, sys, os, torch
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
print(f"  joint_limit_regularization: {w}  ({'UNCONSTRAINED' if not w else 'CONSTRAINED'})")
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
