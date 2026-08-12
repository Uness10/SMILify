#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --time=02:00:00
#SBATCH --job-name=sv_study
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Evaluation half of the single-view joint-limit prior study:
#   benchmark (MPJPE/PCK) -> pose export over the same test split -> analysis.
#
# Single GPU on purpose: benchmark_model has no DDP path (it pins
# config.GPU_IDS), and the export/analysis are cheap relative to training.
#
# Submit FROM THE REPO ROOT, after the training job has finished:
#
#   mkdir -p logs
#   LABEL=unconstrained \
#   CHECKPOINT=runs/singleview_unconstrained/checkpoints/best_model.pth \
#     sbatch --export=ALL,LABEL,CHECKPOINT \
#       hpc_files/sc_venv_pytorch3d/run_singleview_study_JURECA.sbatch
#
# Chain it directly after training instead:
#   jid=$(sbatch --parsable hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch)
#   sbatch --dependency=afterok:$jid --export=ALL,LABEL,CHECKPOINT \
#       hpc_files/sc_venv_pytorch3d/run_singleview_study_JURECA.sbatch
#
# Later, the constrained arm (needs a model .pkl with authored joint_limits):
#   LABEL=constrained CHECKPOINT=runs/singleview_constrained/checkpoints/best_model.pth \
#   LIMITS=authored_limits.npy sbatch --export=ALL,LABEL,CHECKPOINT,LIMITS ...
# ---------------------------------------------------------------------------

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source "${SLURM_SUBMIT_DIR}/hpc_files/sc_venv_pytorch3d/activate.sh"
cd "${SMILIFY_DIR}"

# --- edit / override at submit time ------------------------------------------
export LABEL="${LABEL:-unconstrained}"
export CHECKPOINT="${CHECKPOINT:-runs/singleview_unconstrained/checkpoints/best_model.pth}"
export DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
export SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK.pkl}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MAX_FRAMES="${MAX_FRAMES:-0}"
export LIMITS="${LIMITS:-}"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
bash scripts/prior_study/run_singleview_study.sh

echo "Done — $(date)"
