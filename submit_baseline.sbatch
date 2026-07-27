#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=smil_baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=prior_study_results/slurm_%j.log
#SBATCH --error=prior_study_results/slurm_%j.log
#
# Background baseline study on JURECA-DC. Submit with:
#     sbatch scripts/prior_study/submit_baseline.sbatch
# Monitor:
#     squeue --me
#     tail -f prior_study_results/slurm_<jobid>.log
# Cancel:
#     scancel <jobid>
#
# Override paths/label at submit time, e.g.:
#     LABEL=unconstrained sbatch --export=ALL,LABEL scripts/prior_study/submit_baseline.sbatch
#
# NOTE: intentionally no set -u. Conda activate hooks (libblas_mkl_activate.sh)
# reference unset vars like MKL_INTERFACE_LAYER and abort under set -u.
set -eo pipefail

# ---- locations (edit if your repo / conda live elsewhere) -------------------
REPO_DIR="${REPO_DIR:-/p/scratch/share/anouar1/SMILify}"
CONDA_SH="${CONDA_SH:-/p/scratch/cias-7/anouar1/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-pytorch3d}"

echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
mkdir -p "$REPO_DIR/prior_study_results"

# ---- activate env -----------------------------------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

cd "$REPO_DIR"

# ---- run the study (wrapper does pre-flight + benchmark + inference + analysis)
bash scripts/prior_study/run_baseline_study.sh

echo "Done — $(date)"
