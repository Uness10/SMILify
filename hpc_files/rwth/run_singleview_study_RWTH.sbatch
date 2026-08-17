#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --job-name=sv_study
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# RWTH CLAIX-2023 (c23g) — evaluation half of the single-view joint-limit
# prior study:
#   benchmark (MPJPE/PCK) -> pose export over the same test split -> analysis.
#
# Single GPU on purpose: benchmark_model has no DDP path (it pins config.GPU_IDS),
# and the export/analysis are cheap relative to training. --cpus-per-task=24 is
# one quarter of a 96-core c23g node, matching the one-quarter GPU request so the
# job is schedulable alongside others.
#
# Ported from hpc_files/sc_venv_pytorch3d/run_singleview_study_JURECA.sbatch.
# Two fixes carried in besides the RWTH header:
#   * SMAL_FILE now defaults to the AUTHORED .pkl. The JSC version defaulted to
#     3D_model_prep/SMILy_STICK.pkl, which has no 'joint_limits' key — so
#     analyze_baseline_pose.load_limits found nothing and silently skipped the
#     limit-violation table, which is the headline number of this study.
#   * LABEL/CHECKPOINT default to the constrained arm (the unconstrained arm was
#     dropped from the study).
# LIMITS stays unset on purpose: load_limits falls back to the 'joint_limits' key
# inside SMAL_FILE, so pointing at the authored .pkl is sufficient.
#
# Submit FROM THE REPO ROOT, after the training job finishes:
#
#   mkdir -p logs
#   sbatch --account=<your_project> \
#     hpc_files/rwth/run_singleview_study_RWTH.sbatch
#
# Chain it directly after training instead:
#   jid=$(sbatch --parsable --account=<proj> \
#           hpc_files/rwth/run_singleview_training_RWTH.sbatch)
#   sbatch --dependency=afterok:$jid --account=<proj> \
#     hpc_files/rwth/run_singleview_study_RWTH.sbatch
#
# NOTE: intentionally no `set -u` (conda activate hooks reference unset vars).
set -eo pipefail
# ---------------------------------------------------------------------------

if [[ -z "${SLURM_JOB_ACCOUNT:-}" ]]; then
    echo "ERROR: no SLURM account. Submit with:" >&2
    echo "  sbatch --account=<your_rwth_project> $0" >&2
    exit 1
fi

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
ENV_PREFIX="${ENV_PREFIX:-$HPCWORK/conda_envs/pytorch3d}"

# --- activate conda ----------------------------------------------------------
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
echo "  env    : $CONDA_PREFIX"
echo "  python : $(which python)"

# --- edit / override at submit time ------------------------------------------
export LABEL="${LABEL:-constrained}"
export CHECKPOINT="${CHECKPOINT:-runs/singleview_constrained/checkpoints/best_model.pth}"
export DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
export SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK_limits_authored.pkl}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MAX_FRAMES="${MAX_FRAMES:-0}"
export LIMITS="${LIMITS:-}"

# The violation table is the point of this study; warn loudly if the model file
# cannot supply limits rather than discovering an empty section in the report.
python - "$SMAL_FILE" <<'PY'
import pickle, sys
dd = pickle.load(open(sys.argv[1], "rb"), encoding="latin1")
if dd.get("joint_limits") is None:
    print(f"WARNING: {sys.argv[1]} has no 'joint_limits' — the limit-violation "
          f"table will be SKIPPED. Pass SMAL_FILE=<authored .pkl> or LIMITS=<.npy>.")
else:
    print(f"  limits source: {sys.argv[1]}:joint_limits")
PY

bash scripts/prior_study/run_singleview_study.sh

echo "Done — $(date)"
