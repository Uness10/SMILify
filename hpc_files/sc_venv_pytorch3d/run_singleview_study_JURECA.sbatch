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
# Submit FROM THE REPO ROOT, after the training job finishes:
#
#   mkdir -p logs
#   LABEL=unconstrained \
#   CHECKPOINT=runs/singleview_unconstrained/checkpoints/best_model.pth \
#     sbatch --export=ALL,LABEL,CHECKPOINT \
#       hpc_files/sc_venv_pytorch3d/run_singleview_study_JURECA.sbatch
#
# Chain it directly after training instead:
#   jid=$(sbatch --parsable hpc_files/sc_venv_pytorch3d/run_singleview_training_JURECA.sbatch)
#   LABEL=unconstrained CHECKPOINT=runs/singleview_unconstrained/checkpoints/best_model.pth \
#     sbatch --dependency=afterok:$jid --export=ALL,LABEL,CHECKPOINT \
#       hpc_files/sc_venv_pytorch3d/run_singleview_study_JURECA.sbatch
#
# NOTE: intentionally no `set -u` (conda activate hooks reference unset vars).
set -eo pipefail
# ---------------------------------------------------------------------------

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
ENV_PREFIX="${ENV_PREFIX:-/p/project1/cias-7/anouar1/conda_envs/pytorch3d}"

# --- activate conda ----------------------------------------------------------
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
echo "  env    : $CONDA_PREFIX"
echo "  python : $(which python)"

# --- edit / override at submit time ------------------------------------------
export LABEL="${LABEL:-unconstrained}"
export CHECKPOINT="${CHECKPOINT:-runs/singleview_unconstrained/checkpoints/best_model.pth}"
export DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
export SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK.pkl}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MAX_FRAMES="${MAX_FRAMES:-0}"
export LIMITS="${LIMITS:-}"

bash scripts/prior_study/run_singleview_study.sh

echo "Done — $(date)"
