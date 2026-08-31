#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --job-name=jl_render
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# JOINT-LIMIT PRIOR STUDY — qualitative renders (RWTH CLAIX-2023, c23g)
#
# Renders each arm's ALREADY-EXPORTED poses (prior_study_results/<arm>/clip_*.npz,
# written by export_poses.py during the eval stage) over the SAME frame windows,
# so the change the prior makes can be judged by eye.
#
#   task  arm                    npz
#   ----  ---------------------  --------------------------------------------
#     0   sv_reference           prior_study_results/sv_reference/clip_sv_reference.npz
#     1   sv_constrained @ 1e-4  prior_study_results/lam1e-4/sv_constrained/clip_sv_constrained.npz
#     2   sv_constrained @ 1e-3  prior_study_results/lam1e-3/sv_constrained/clip_sv_constrained.npz
#     3   sv_constrained @ 1e-2  prior_study_results/lam1e-2/sv_constrained/clip_sv_constrained.npz
#     4   sv_constrained @ 1e-1  prior_study_results/lam1e-1/sv_constrained/clip_sv_constrained.npz
#     5   sv_constrained @ 100   prior_study_results/sv_constrained/clip_sv_constrained.npz
#                                (the EARLIER w=100 run — pre-sweep flat layout,
#                                 and NOT epoch-matched to tasks 1-4)
#
# Tasks 0 and 5 are the two arms that exist before the sweep has trained, so
# `--array=0,5` gives a reference-vs-w=100 comparison today.
#
# ---------------------------------------------------------------------------
# WHY THE .npz AND NOT THE RAW CLIPS
# ---------------------------------------------------------------------------
# The .npz files hold the predictions the study actually SCORED — same test
# split, same order, same frames behind every MPJPE / PCK / violation number.
# Rendering them means the pictures and the tables describe one thing. Re-running
# inference on raw video would answer a different question (how the model behaves
# on those clips) and would line up with nothing in the results.
#
# It is also far cheaper: no checkpoint, no dataset, no backbone — just SMAL
# forward plus rasterisation, minutes per arm instead of hours.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING — pick the windows ONCE, on the login node
# ---------------------------------------------------------------------------
# Every arm must render the SAME frames or the grid is not a comparison. The
# windows are chosen once, from the REFERENCE arm, and shared:
#
#   python scripts/prior_study/pick_segments.py \
#       --npz  prior_study_results/sv_reference/clip_sv_reference.npz \
#       --dataset SMILySTICKS_centred_reprojected_FIXED.h5 \
#       --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
#       --segments 3 --length 300 --select worst --one-per-camera \
#       --out prior_study_results/renders/segments.json
#
# --dataset is NOT optional in practice. Consecutive rows of the .npz are one
# instant seen from several cameras, not consecutive time: the single-view items
# are enumerated sample-major over (sample, view) and the camera-centric split
# keeps whole samples together, so raw row order cycles the cameras and the
# animal appears to spin. pick_segments.py needs the HDF5 to undo that
# interleave and emit real per-camera tracks in frame order.
#
# --select worst aims the renders at the windows where the reference violates
# the authored limits most — the frames the prior is supposed to fix. A window
# where the reference was already in range cannot show the prior doing anything.
# Use --select even for an unbiased look alongside it.
#
# CPU only, seconds. This job refuses to start if segments.json is missing,
# rather than inventing its own windows.
#
# ---------------------------------------------------------------------------
# SUBMIT
# ---------------------------------------------------------------------------
#   sbatch --array=0-4 --account=rwth2151 hpc_files/rwth/run_prior_study_render.sbatch
#
# The reference alone, as soon as its eval has run (no training needed):
#   sbatch --array=0 --account=rwth2151 hpc_files/rwth/run_prior_study_render.sbatch
#
# The earlier w=100 arm, which is already on disk:
#   sbatch --array=5 --account=rwth2151 hpc_files/rwth/run_prior_study_render.sbatch
#   bash scripts/prior_study/stack_renders.sh      # -> reference | w=100
#
# An arm in some other location entirely (ARM names the output folder, NPZ the
# input, LABEL the burned-in caption — all three override any task's defaults):
#   ARM=lam100_rerun NPZ=some/other/clip.npz LABEL="lambda = 100 (rerun)" \
#     sbatch --array=5 --account=rwth2151 --export=ALL,ARM,NPZ,LABEL \
#       hpc_files/rwth/run_prior_study_render.sbatch
#
# Chained after the eval array:
#   sbatch --array=0-4 --dependency=afterany:$eid --account=rwth2151 \
#       hpc_files/rwth/run_prior_study_render.sbatch
#
# Then stack them into one labelled video per window (login node, ffmpeg):
#   bash scripts/prior_study/stack_renders.sh
#
# Env knobs:
#   SEGMENTS_FILE  default: prior_study_results/renders/segments.json
#   RENDER_ROOT    output root (default: prior_study_results/renders)
#   CAMERA         orbit | predicted        (default: orbit)
#   GLOBAL_ROT     zero | keep | auto      (default: auto = zero under orbit).
#                  These are camera-centric checkpoints, so the root rotation is
#                  the animal's orientation relative to the camera that saw that
#                  row. Zeroing it puts every frame in a canonical body frame and
#                  costs nothing: the authored .pkl pins the root to [0,0] and
#                  every violation statistic already excludes it. GLOBAL_ROT=keep
#                  shows the predicted turning — only meaningful now that the
#                  segments are single-camera tracks.
#   FPS            output fps (default: the sidecar's, 30). The test split is
#                  ~10% of view-items, so a track is time-sparse and 30 fps plays
#                  ~10x faster than life; pick_segments.py prints the fps that
#                  gives approximately real-time motion.
#   IMAGE_SIZE     square px               (default: 512)
#   ELEV / AZIM    orbit camera angles     (default: 15 / 45)
#   SCALING        auto|mesh_scale|ue|none (default: auto)
#   BATCH_SIZE     frames per render call  (default: 16)
#
# NOTE: intentionally no `set -u` (conda activate hooks reference unset vars).
set -eo pipefail
# ---------------------------------------------------------------------------

if [[ -z "${SLURM_JOB_ACCOUNT:-}" ]]; then
    echo "ERROR: no SLURM account. Submit with:" >&2
    echo "  sbatch --account=rwth2151 $0" >&2
    exit 1
fi

LAMBDAS=(1e-4 1e-3 1e-2 1e-1)
RESULTS_ROOT="${RESULTS_ROOT:-prior_study_results}"
RENDER_ROOT="${RENDER_ROOT:-$RESULTS_ROOT/renders}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK == 0 )); then
    ARM="${ARM:-sv_reference}"
    NPZ="${NPZ:-$RESULTS_ROOT/sv_reference/clip_sv_reference.npz}"
    LABEL="${LABEL:-reference (no limit prior)}"
elif (( TASK >= 1 && TASK <= 4 )); then
    LAMBDA="${LAMBDAS[$((TASK - 1))]}"
    ARM="${ARM:-lam${LAMBDA}}"
    NPZ="${NPZ:-$RESULTS_ROOT/lam${LAMBDA}/sv_constrained/clip_sv_constrained.npz}"
    LABEL="${LABEL:-lambda = ${LAMBDA}}"
elif (( TASK == 5 )); then
    # The EARLIER w=100 run, which predates the sweep and therefore lives in the
    # flat pre-sweep layout (prior_study_results/sv_constrained/) rather than
    # under a lam<L>/ root. Kept as its own task so the one arm that already
    # exists can be rendered and compared against the reference today, without
    # waiting for the sweep to train.
    #
    # It is NOT epoch-matched to the sweep: it continued the reference for the
    # handful of epochs that fit its old 12 h budget, while every sweep arm gets
    # EXTRA_EPOCHS (50). The label carries that so it cannot be forgotten once
    # the panel is sitting in a grid next to four arms that trained 6x longer.
    LAMBDA="100"
    ARM="${ARM:-lam100}"
    NPZ="${NPZ:-$RESULTS_ROOT/sv_constrained/clip_sv_constrained.npz}"
    LABEL="${LABEL:-lambda = 100  [older run, fewer epochs]}"
else
    echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$TASK (expected 0-5)." >&2
    echo "       0 = reference, 1-4 = the lambda sweep, 5 = the earlier w=100 arm." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$HPCWORK/hf_cache}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
# --- resolve the conda env ---------------------------------------------------
# ENV_PREFIX is a full env PREFIX (a directory), not a name, and the layout
# differs per install: a conda rooted at $HPCWORK puts envs in `conda/envs/`,
# while pointing `envs_dirs` at $HPCWORK gives `conda_envs/`. Guessing one and
# letting `conda activate` fail burns a queue slot for a one-line error, so
# probe the known layouts and check bin/python actually exists first.
if [[ -z "${ENV_PREFIX:-}" ]]; then
    for cand in "$HPCWORK/conda/envs/pytorch3d" \
                "$HPCWORK/conda_envs/pytorch3d" \
                "$HPCWORK/miniconda3/envs/pytorch3d" \
                "$HPCWORK/miniforge3/envs/pytorch3d" \
                "$HOME/.conda/envs/pytorch3d"; do
        [[ -x "$cand/bin/python" ]] && { ENV_PREFIX="$cand"; break; }
    done
fi
if [[ ! -x "${ENV_PREFIX:-}/bin/python" ]]; then
    echo "ERROR: no conda env found." >&2
    echo "       ENV_PREFIX='${ENV_PREFIX:-<unset>}' has no bin/python, and none of the" >&2
    echo "       known layouts under HPCWORK=$HPCWORK matched either." >&2
    echo "       Find yours on the login node:  conda env list" >&2
    echo "       Then submit with:" >&2
    echo "         ENV_PREFIX=<full prefix path> sbatch --export=ALL,ENV_PREFIX ... $0" >&2
    exit 1
fi

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

# Run from the repo root: SMAL_FILE and the results tree are addressed relatively.
cd "$REPO_DIR"

SEGMENTS_FILE="${SEGMENTS_FILE:-$RENDER_ROOT/segments.json}"
SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK_limits_authored.pkl}"
DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
CAMERA="${CAMERA:-orbit}"
GLOBAL_ROT="${GLOBAL_ROT:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
ELEV="${ELEV:-15}"
AZIM="${AZIM:-45}"
SCALING="${SCALING:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUT_DIR="$RENDER_ROOT/$ARM"

echo "=================================================================="
echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $TASK on $(hostname) — $(date)"
echo "  arm      : $ARM — $LABEL"
echo "  npz      : $NPZ"
echo "  segments : $SEGMENTS_FILE"
echo "  camera   : $CAMERA   image size ${IMAGE_SIZE}px"
echo "  out dir  : $OUT_DIR"
echo "=================================================================="

# --- the two things that make this job produce a non-comparison --------------
if [[ ! -f "$SEGMENTS_FILE" ]]; then
    echo "ERROR: $SEGMENTS_FILE not found." >&2
    echo "       Every arm must render the SAME windows, so they are chosen once on the" >&2
    echo "       login node before any render job runs:" >&2
    echo >&2
    echo "         python scripts/prior_study/pick_segments.py \\" >&2
    echo "             --npz $RESULTS_ROOT/sv_reference/clip_sv_reference.npz \\" >&2
    echo "             --dataset $DATASET \\" >&2
    echo "             --smal-file $SMAL_FILE \\" >&2
    echo "             --segments 3 --length 300 --select worst --one-per-camera \\" >&2
    echo "             --out $SEGMENTS_FILE" >&2
    echo >&2
    echo "       --dataset matters: without it the segments are raw npz row windows," >&2
    echo "       which interleave the cameras and make the animal appear to spin." >&2
    exit 1
fi

# A segments.json built without --dataset renders a spin, not motion. Refuse by
# default rather than producing a video that misleads at a glance.
if ! python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('time_ordered') else 1)" \
        "$SEGMENTS_FILE"; then
    if [[ "${ALLOW_UNORDERED_SEGMENTS:-0}" == "1" ]]; then
        echo "[render] WARNING: segments are NOT time-ordered (ALLOW_UNORDERED_SEGMENTS=1)." >&2
        echo "         The output shows the camera jumping around the animal, not the animal moving." >&2
    else
        echo "ERROR: $SEGMENTS_FILE was built without --dataset, so its segments are raw npz" >&2
        echo "       row windows. Consecutive rows are one instant seen from several cameras," >&2
        echo "       so the render would spin the animal instead of animating it." >&2
        echo "       Re-run pick_segments.py with --dataset $DATASET, or set" >&2
        echo "       ALLOW_UNORDERED_SEGMENTS=1 if you really want the raw ordering." >&2
        exit 1
    fi
fi

if [[ ! -f "$NPZ" ]]; then
    echo "ERROR: exported poses not found: $NPZ" >&2
    echo "       This arm has not been evaluated yet — run_prior_study.sh writes the" >&2
    echo "       clip_*.npz during the eval stage. Submit the matching eval task first:" >&2
    if (( TASK == 5 )); then
        echo "         This is the PRE-SWEEP w=100 arm, expected in the flat layout" >&2
        echo "         $RESULTS_ROOT/sv_constrained/. If your copy lives elsewhere, point" >&2
        echo "         at it directly rather than moving files:" >&2
        echo "           NPZ=<path to clip_sv_constrained.npz> sbatch --array=5 \\" >&2
        echo "             --account=rwth2151 --export=ALL,NPZ \\" >&2
        echo "             hpc_files/rwth/run_prior_study_render.sbatch" >&2
    else
        echo "         sbatch --array=$TASK --account=rwth2151 --export=ALL,SV_REF \\" >&2
        echo "             hpc_files/rwth/run_prior_study_eval.sbatch" >&2
    fi
    exit 1
fi

python -c "import torch; print(f'  torch: {torch.__version__}, {torch.cuda.device_count()} GPU(s) visible')"

python scripts/prior_study/render_clip_npz.py \
    --npz "$NPZ" \
    --smal-file "$SMAL_FILE" \
    --segments-file "$SEGMENTS_FILE" \
    --out-dir "$OUT_DIR" \
    --label "$LABEL" \
    --camera "$CAMERA" \
    --image-size "$IMAGE_SIZE" \
    --elev "$ELEV" --azim "$AZIM" \
    --scaling "$SCALING" \
    --global-rot "$GLOBAL_ROT" \
    ${FPS:+--fps "$FPS"} \
    --batch-size "$BATCH_SIZE"

echo
echo "=================================================================="
echo " $ARM rendered into $OUT_DIR"
echo
echo " Once every arm has finished, stack them on the login node:"
echo "   bash scripts/prior_study/stack_renders.sh"
echo "=================================================================="
echo "Done ($ARM) — $(date)"
