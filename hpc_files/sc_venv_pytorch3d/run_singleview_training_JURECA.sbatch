#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --account=ias-7
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
# Prepare the config on the LOGIN NODE first (it reads the checkpoint's epoch
# and turns "N more epochs" into the absolute end epoch the trainer expects):
#
#   mkdir -p logs configs_runs
#   source hpc_files/sc_venv_pytorch3d/activate.sh
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
# Override the config without editing this file:
#   CONFIG=configs_runs/singleview_constrained.json \
#     sbatch --export=ALL,CONFIG --job-name=sv_constrained \
#       hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch
#
# Monitor:  squeue --me   /   tail -f logs/<jobname>_<jobid>.out
# Cancel :  scancel <jobid>
# ---------------------------------------------------------------------------

# Backbone weights are pre-fetched on the login node (hpc_files/download_backbone_weights.py);
# keep HF/timm off the network on the compute nodes.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# One task per node; expose the task's CPU allocation to srun (DataLoader workers).
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun rendezvous: first node of the allocation is the master.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7011

# Activate the venv (config.sh exports SMILIFY_DIR = repo root), then run from the
# repo root so the absolute `smal_fitter.*` / `config` imports resolve.
source "${SLURM_SUBMIT_DIR}/hpc_files/sc_venv_pytorch3d/activate.sh"
cd "${SMILIFY_DIR}"

# --- edit for your run -------------------------------------------------------
CONFIG="${CONFIG:-configs_runs/singleview_unconstrained.json}"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "Config: $CONFIG"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "       Run scripts/prior_study/prepare_resume_config.py on the login node first." >&2
    exit 1
fi

# --- pre-flight: the num_epochs trap ----------------------------------------
# training.num_epochs is an ABSOLUTE end epoch and the trainer resumes at
# checkpoint_epoch + 1. If num_epochs <= that, the loop body never executes and
# the job "succeeds" after training nothing. Catch it before allocating GPUs.
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
print(f"  checkpoints -> {cfg.get('output', {}).get('checkpoint_dir')}")
PY

# --- launch ------------------------------------------------------------------
# One torchrun agent per node, one process per GPU. The single-view trainer
# detects RANK/LOCAL_RANK/WORLD_SIZE (is_torchrun_launched) and skips its own
# mp.spawn path, so --num_gpus is intentionally not passed.
srun --export=ALL --cpu_bind=none bash -c "torchrun_jsc \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=gpu \
    --rdzv_backend c10d \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    -m smal_fitter.neuralSMIL.train_smil_regressor --config $CONFIG"

echo "Done — $(date)"
