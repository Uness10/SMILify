#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --job-name=jl_eval
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# JOINT-LIMIT PRIOR STUDY — evaluation stage (RWTH CLAIX-2023, c23g)
#
# Scores all four arms, as a 4-task array, one GPU each:
#
#     0  sv_reference     the downloaded single-view checkpoint, unmodified
#     1  sv_constrained   its +10-epoch continuation with the limit penalty
#     2  mv_reference     the downloaded multi-view checkpoint, unmodified
#     3  mv_constrained   its +10-epoch continuation with the limit penalty
#
# Each task runs benchmark -> pose export -> analysis into
# prior_study_results/<arm>/. The LAST task to finish also runs compare_arms.py,
# which joins whatever arms are on disk into the headline table.
#
# ALL FOUR arms are scored against the SAME authored .pkl. The reference arms
# were trained without limits but must still be *measured* against those ranges —
# that violation rate is the "before" number the study rests on.
#
# ---------------------------------------------------------------------------
# CHECKPOINT PATHS — set these before submitting
# ---------------------------------------------------------------------------
# The reference checkpoints live wherever you downloaded them, so they are NOT
# hardcoded. Export them at submit time (the job aborts loudly if a path is
# missing rather than silently scoring three arms):
#
#   SV_REF=singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth \
#   MV_REF=SMILySTICKS_ViT_model.pth \
#     sbatch --account=<proj> --export=ALL,SV_REF,MV_REF \
#       hpc_files/rwth/run_prior_study_eval.sbatch
#
# Chained after training (the usual case):
#
#   jid=$(sbatch --parsable --account=<proj> hpc_files/rwth/run_prior_study_train.sbatch)
#   SV_REF=... MV_REF=... sbatch --dependency=afterok:$jid --account=<proj> \
#       --export=ALL,SV_REF,MV_REF hpc_files/rwth/run_prior_study_eval.sbatch
#
# Score the reference arms alone (no training needed — they are just the
# downloaded checkpoints, so this can run immediately):
#
#   SV_REF=... MV_REF=... sbatch --array=0,2 --account=<proj> \
#       --export=ALL,SV_REF,MV_REF hpc_files/rwth/run_prior_study_eval.sbatch
#
# NOTE: intentionally no `set -u` (conda activate hooks reference unset vars).
set -eo pipefail
# ---------------------------------------------------------------------------

if [[ -z "${SLURM_JOB_ACCOUNT:-}" ]]; then
    echo "ERROR: no SLURM account. Submit with:" >&2
    echo "  sbatch --account=<your_rwth_project> $0" >&2
    exit 1
fi

# --- which arm is this array task? -------------------------------------------
# The constrained checkpoints are written by the training job into
# runs/<mode>_constrained/checkpoints/; run_prior_study.sh falls back to the
# newest periodic checkpoint if best_model.pth was never written (val loss need
# not beat the resumed best in a short continuation).
TASK="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK" in
    0) LABEL="sv_reference"
       CHECKPOINT="${SV_REF:-}"
       WHAT="downloaded single-view checkpoint (set SV_REF=<path>)" ;;
    1) LABEL="sv_constrained"
       CHECKPOINT="${SV_CON:-runs/singleview_constrained/checkpoints/best_model.pth}"
       WHAT="single-view constrained continuation" ;;
    2) LABEL="mv_reference"
       CHECKPOINT="${MV_REF:-}"
       WHAT="downloaded multi-view checkpoint (set MV_REF=<path>)" ;;
    3) LABEL="mv_constrained"
       CHECKPOINT="${MV_CON:-runs/multiview_constrained/checkpoints/best_model.pth}"
       WHAT="multi-view constrained continuation" ;;
    *) echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$TASK (expected 0-3)." >&2; exit 1 ;;
esac

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
ENV_PREFIX="${ENV_PREFIX:-$HPCWORK/conda_envs/pytorch3d}"

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

# The authored .pkl is the scoring reference for EVERY arm, including the
# references. Pointing this at SMILy_STICK.pkl (no joint_limits) would make
# analyze_baseline_pose silently skip the violation table.
export SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK_limits_authored.pkl}"
export DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"

echo "=================================================================="
echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $TASK on $(hostname) — $(date)"
echo "  arm        : $LABEL — $WHAT"
echo "  checkpoint : ${CHECKPOINT:-<UNSET>}"
echo "  dataset    : $DATASET"
echo "  smal file  : $SMAL_FILE"
echo "=================================================================="

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: no checkpoint for arm '$LABEL'." >&2
    echo "       This arm needs $WHAT." >&2
    echo "       Re-submit with --export=ALL,SV_REF,MV_REF and the paths set." >&2
    exit 1
fi

# run_prior_study.sh re-checks existence (and applies the best_model.pth
# fallback), but failing here gives a clearer message per arm.
if [[ ! -e "$CHECKPOINT" && "$(basename "$CHECKPOINT")" != "best_model.pth" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

LABEL="$LABEL" CHECKPOINT="$CHECKPOINT" \
  bash scripts/prior_study/run_prior_study.sh

# --- join the arms -----------------------------------------------------------
# Every task attempts this; compare_arms.py tolerates missing arms and simply
# reports what it found, so whichever task finishes last produces the complete
# table. Cheap enough (pure CSV/text parsing) that racing is not worth avoiding.
echo
echo "[eval] refreshing the comparison table from whatever arms are on disk..."
python scripts/prior_study/compare_arms.py \
    --results-root prior_study_results \
    --out prior_study_results/comparison \
    --extra-epochs "${EXTRA_EPOCHS:-10}" \
    --joint-limit-weight "${JOINT_LIMIT_WEIGHT:-100.0}" || \
  echo "[eval] (comparison incomplete — expected until all four arms have run)"

echo "Done ($LABEL) — $(date)"
