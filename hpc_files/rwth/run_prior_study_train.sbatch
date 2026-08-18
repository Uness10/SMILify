#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c23g
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --job-name=jl_train
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# JOINT-LIMIT PRIOR STUDY — training stage (RWTH CLAIX-2023, c23g)
#
# Fine-tunes BOTH modalities from their downloaded checkpoints with the joint-
# limit penalty enabled, as a 2-task array:
#
#     SLURM_ARRAY_TASK_ID=0  ->  singleview  (configs_runs/singleview_constrained.json)
#     SLURM_ARRAY_TASK_ID=1  ->  multiview   (configs_runs/multiview_constrained.json)
#
# Each task gets its own node with 4x H100 and runs torchrun --standalone.
# The study has ONLY the constrained arm: the reference is each downloaded
# checkpoint evaluated as-is, which the eval sbatch handles. Nothing here trains
# an unconstrained arm.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING — prepare both configs on the LOGIN NODE
# ---------------------------------------------------------------------------
# prepare_resume_config.py reads each checkpoint's epoch and turns "10 more
# epochs" into the absolute end epoch the trainers expect (num_epochs is an END
# epoch, not a count — see the pre-flight below).
#
#   cd "$HPCWORK/SMILify"
#   conda activate "$HPCWORK/conda_envs/pytorch3d"
#   mkdir -p logs configs_runs
#
#   # single-view
#   python scripts/prior_study/prepare_resume_config.py \
#       --checkpoint  <path to your singleview .pth> \
#       --base-config <the JSON that produced it> \
#       --extra-epochs 10 --label constrained \
#       --joint-limit-weight 100.0 \
#       --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
#       --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
#       --out configs_runs/singleview_constrained.json
#
#   # multi-view
#   python scripts/prior_study/prepare_resume_config.py \
#       --checkpoint  <path to your multiview .pth> \
#       --base-config smal_fitter/neuralSMIL/configs/examples/multiview_SMILySTICKS_3D_ViT_Large_AUG_FIXED.json \
#       --extra-epochs 10 --label constrained \
#       --joint-limit-weight 100.0 \
#       --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
#       --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
#       --out configs_runs/multiview_constrained.json
#
# The script refuses to write if the checkpoint's modality disagrees with the
# base config, or if the model and dataset describe different skeletons.
#
# ---------------------------------------------------------------------------
# SUBMIT (from the repo root — the #SBATCH --output path resolves at submit
# time, so logs/ must already exist)
# ---------------------------------------------------------------------------
#   jid=$(sbatch --parsable --account=<your_project> \
#           hpc_files/rwth/run_prior_study_train.sbatch)
#   sbatch --dependency=afterok:$jid --account=<your_project> \
#           hpc_files/rwth/run_prior_study_eval.sbatch
#
# Run one modality only:
#   sbatch --array=0 --account=<proj> hpc_files/rwth/run_prior_study_train.sbatch
#
# Override a config without editing this file:
#   SV_CONFIG=configs_runs/other.json sbatch --account=<proj> \
#       --export=ALL,SV_CONFIG hpc_files/rwth/run_prior_study_train.sbatch
#
# Monitor:  squeue --me   /   tail -f logs/jl_train_<jobid>_<taskid>.out
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

# --- which arm is this array task? -------------------------------------------
TASK="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK" in
    0) MODE="singleview"; CONFIG="${SV_CONFIG:-configs_runs/singleview_constrained.json}"
       TRAIN_MODULE="smal_fitter.neuralSMIL.train_smil_regressor" ;;
    1) MODE="multiview";  CONFIG="${MV_CONFIG:-configs_runs/multiview_constrained.json}"
       TRAIN_MODULE="smal_fitter.neuralSMIL.train_multiview_regressor" ;;
    *) echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$TASK (expected 0 or 1)." >&2; exit 1 ;;
esac

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

echo "=================================================================="
echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $TASK on $(hostname) — $(date)"
echo "  arm    : CONSTRAINED / $MODE"
echo "  cluster: RWTH CLAIX-2023 | partition ${SLURM_JOB_PARTITION:-?} | account $SLURM_JOB_ACCOUNT"
echo "  repo   : $REPO_DIR"
echo "  env    : $CONDA_PREFIX"
echo "  config : $CONFIG"
echo "=================================================================="
python -c "import torch; print(f'  torch  : {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s) visible')"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "       Run scripts/prior_study/prepare_resume_config.py on the login node first" >&2
    echo "       (see the header of this file for the exact two commands)." >&2
    exit 1
fi

# --- pre-flight: everything that can silently no-op --------------------------
# Each check below corresponds to a way this job can "succeed" while producing a
# result that does not answer the question. All of them are cheap and run before
# the GPUs are touched.
python - "$CONFIG" "$MODE" <<'PY' || exit 1
import json, os, pickle, sys
import numpy as np
import torch

cfg_path, expected_mode = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))

# 1. mode -------------------------------------------------------------------
mode = cfg.get("mode")
if mode != expected_mode:
    sys.exit(f"ERROR: config mode is '{mode}' but this array task trains '{expected_mode}'.\n"
             f"       Array task 0 = singleview, task 1 = multiview.")

# 2. the num_epochs trap ----------------------------------------------------
# training.num_epochs is an ABSOLUTE end epoch and the trainers resume at
# checkpoint_epoch + 1, looping range(start, num_epochs). If num_epochs <= that,
# the loop body never executes and the job "succeeds" after training nothing.
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

# 3. the checkpoint really is this modality ---------------------------------
sd = ck.get("model_state_dict", ck)
ckpt_mode = "multiview" if "view_embeddings.weight" in sd else "singleview"
if ckpt_mode != expected_mode:
    sys.exit(f"ERROR: {ckpt_path} is a {ckpt_mode} checkpoint but the config says {expected_mode}.")

# 4. the arm really is constrained ------------------------------------------
w = (cfg.get("loss_curriculum", {}).get("base_weights", {}) or {}).get("joint_limit_regularization", 0.0)
print(f"  joint_limit_regularization: {w}")
if not w:
    sys.exit("ERROR: joint_limit_regularization is 0 — this study has only a CONSTRAINED arm.\n"
             "       Re-run prepare_resume_config.py with --joint-limit-weight 100.0.")

# 5. the model carries limits for the hinge to act on -----------------------
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
if not np.all(jl[..., 0] <= jl[..., 1]):
    sys.exit("ERROR: joint_limits has min > max somewhere; LimitPrior will raise.")
if not np.isfinite(jl).all():
    sys.exit("ERROR: joint_limits contains non-finite values.")

pi = np.pi
free = (jl[..., 0] <= -pi + 1e-4) & (jl[..., 1] >= pi - 1e-4)
# Exclude the root: LimitPrior pins J_names[0] to [0,0] and the fitter drops it
# via the [3:] slice, so it contributes nothing to the penalty.
n_con_nonroot = int((~free[1:]).sum())
n_axes_nonroot = free[1:].size
if n_con_nonroot == 0:
    sys.exit("ERROR: every non-root joint/axis is wide open ([-pi,pi] = free) — the hinge can never fire.")
coverage = 100.0 * n_con_nonroot / n_axes_nonroot
print(f"  authored limits   : {n_con_nonroot}/{n_axes_nonroot} non-root axes constrained ({coverage:.0f}%), "
      f"across {int((~free[1:]).any(axis=1).sum())} joints")

# 6. coverage dilution ------------------------------------------------------
# _joint_limit_penalty is a torch.mean over the FULL (N, N_POSE, 3) hinge
# (smil_image_regressor.py:1445). Free axes contribute exact zeros to that mean,
# so the fraction of authored axes scales the effective prior strength.
print(f"  NOTE: the penalty is a mean over ALL {n_axes_nonroot} non-root axes, so the effective")
print(f"        prior strength is w x {coverage / 100.0:.2f}. At w={w} that is an effective {w * coverage / 100.0:.1f}.")

# 7. DDP batch arithmetic ---------------------------------------------------
# batch_size is PER PROCESS under DDP, so 4 GPUs quadruple the effective batch
# relative to the run that produced the checkpoint, while inheriting that run's
# learning-rate curriculum unchanged.
bs = tr.get("batch_size")
print(f"  batch_size {bs} per GPU x 4 GPUs -> effective {bs * 4 if bs else '?'} "
      f"(batch_size is PER PROCESS under DDP)")
print(f"  checkpoints -> {cfg.get('output', {}).get('checkpoint_dir')}")
PY

# --- launch ------------------------------------------------------------------
# Single node: --standalone gives torchrun its own rendezvous on localhost, so no
# MASTER_ADDR/MASTER_PORT/rdzv plumbing is needed. Both trainers detect
# RANK/LOCAL_RANK/WORLD_SIZE (is_torchrun_launched) and skip their own mp.spawn
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
    -m "$TRAIN_MODULE" --config "$CONFIG"

echo "Done ($MODE) — $(date)"
