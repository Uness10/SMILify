#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --job-name=jl_render
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# JOINT-LIMIT PRIOR STUDY — qualitative renders (RWTH CLAIX)
#
# Uses the ESTABLISHED inference renderer,
#   smal_fitter/neuralSMIL/run_singleview_inference.py
# on raw video clips: every consecutive frame, at the clip's own frame rate,
# with the fitted mesh drawn over the actual footage.
#
#   task  arm                    checkpoint
#   ----  ---------------------  --------------------------------------------
#     0   sv_reference           $SV_REF (downloaded, unmodified)
#     1   sv_constrained @ 1e-4  $RUNS_ROOT/singleview_lam1e-4/checkpoints/best_model.pth
#     2   sv_constrained @ 1e-3  $RUNS_ROOT/singleview_lam1e-3/checkpoints/best_model.pth
#     3   sv_constrained @ 1e-2  $RUNS_ROOT/singleview_lam1e-2/checkpoints/best_model.pth
#     4   sv_constrained @ 1e-1  $RUNS_ROOT/singleview_lam1e-1/checkpoints/best_model.pth
#     5   sv_constrained @ 100   $SV_CON100 (the earlier w=100 run, if kept)
#
# RUNS_ROOT must match the value prepare_lambda_sweep.sh was run with, exactly
# as for the eval array.
#
# ---------------------------------------------------------------------------
# WHY RAW CLIPS AND NOT THE EXPORTED POSES
# ---------------------------------------------------------------------------
# An earlier version of this job rendered `prior_study_results/<arm>/clip_*.npz`
# — the pose export the benchmark scores. That was a mistake and the output was
# rightly called out as odd. Three reasons it cannot show locomotion:
#
#   1. Those poses cover the TEST SPLIT ONLY, ~10 % of view-items. Consecutive
#      exported rows are ~7 recording frames apart, so any playback skips 6 of
#      every 7 frames — the "~5 fps with loads of skipped frames" symptom.
#   2. The single-view items are enumerated sample-major over (sample, view) and
#      the camera-centric split keeps whole samples together, so raw export order
#      cycles the cameras around one instant rather than advancing time.
#   3. There is no footage behind the mesh, so there is no reference against
#      which a leg is visibly in the wrong place.
#
# Rendering the clips directly fixes all three: every frame, in order, at native
# frame rate, over the image the model was looking at.
#
# The scored poses remain the right input for the NUMBERS
# (analyze_baseline_pose.py / compare_arms.py). They are simply not a video.
# scripts/prior_study/pick_segments.py and render_clip_npz.py were written for
# that approach and are SUPERSEDED — delete them.
#
# ---------------------------------------------------------------------------
# WHICH CLIPS
# ---------------------------------------------------------------------------
# Give the clips one of three ways (checked in this order):
#
#   CLIP_LIST=clips.txt      a file with one absolute clip path per line
#                            (lines starting with # are ignored)
#   CLIPS="a.mp4,b.mp4"      comma-separated paths
#   CLIPS_DIR=/path/to/clips all *.mp4/*.avi/*.mov/*.mkv directly inside it,
#                            capped at MAX_CLIPS (default 3)
#
# Pick clips with clear walking bouts. A standing animal shows nothing either
# way, and the point is to see whether a leg segment folds somewhere a stick
# insect's leg cannot.
#
# ---------------------------------------------------------------------------
# SUBMIT
# ---------------------------------------------------------------------------
# SMOKE TEST FIRST — one arm, 200 frames, with individual frames written out:
#
#   SV_REF=... CLIPS_DIR=$HPCWORK/clips MAX_FRAMES=200 SAVE_FRAMES=1 \
#     sbatch --array=0 --account=rwth2151 \
#       --export=ALL,SV_REF,CLIPS_DIR,MAX_FRAMES,SAVE_FRAMES,ENV_PREFIX \
#       hpc_files/rwth/run_prior_study_render.sbatch
#
# Then LOOK AT THE FRAMES in <out>/frames/ before spending the rest. Does the
# mesh track the animal, do the legs swing through a stride? That check is the
# one the .npz renders could not answer, and it costs one GPU-minute.
#
# All arms, full clips:
#
#   RUNS_ROOT=$HPCWORK/smilify_runs SV_REF=... CLIPS_DIR=... \
#     sbatch --array=0-4 --account=rwth2151 \
#       --export=ALL,SV_REF,CLIPS_DIR,RUNS_ROOT,ENV_PREFIX \
#       hpc_files/rwth/run_prior_study_render.sbatch
#
# Then stack the arms side by side (login node, ffmpeg):
#   bash scripts/prior_study/stack_renders.sh
#
# Env knobs:
#   MAX_FRAMES      frames per clip, -1 = all           (default: -1)
#   FPS             output fps                          (default: the clip's own)
#   CROP_MODE       centred | default | bbox_crop       (default: centred)
#   EXPORT_MODE     overlay | side_by_side              (default: side_by_side)
#   CAM_SMOOTHING   camera moving-average window        (default: 0 = off)
#   SAVE_FRAMES=1   also write individual frame PNGs
#   SLEAP_PROJECT   required when CROP_MODE=bbox_crop
#   RENDER_ROOT     output root (default: prior_study_results/renders)
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
# Must match the RUNS_ROOT prepare_lambda_sweep.sh used — same contract as the
# eval array.
RUNS_ROOT="${RUNS_ROOT:-runs}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK == 0 )); then
    ARM="${ARM:-sv_reference}"
    CHECKPOINT="${SV_REF:-}"
    WHAT="downloaded single-view checkpoint (set SV_REF=<path>)"
elif (( TASK >= 1 && TASK <= 4 )); then
    LAMBDA="${LAMBDAS[$((TASK - 1))]}"
    ARM="${ARM:-lam${LAMBDA}}"
    CHECKPOINT="${SV_CON:-$RUNS_ROOT/singleview_lam${LAMBDA}/checkpoints/best_model.pth}"
    WHAT="single-view continuation at lambda=$LAMBDA"
elif (( TASK == 5 )); then
    # The earlier w=100 run. NOT epoch-matched to the sweep — it continued the
    # reference for the few epochs its old 12 h budget allowed, while every
    # sweep arm gets EXTRA_EPOCHS. Read its panel against sv_reference, not
    # against the sweep arms.
    LAMBDA="100"
    ARM="${ARM:-lam100}"
    CHECKPOINT="${SV_CON100:-$RUNS_ROOT/singleview_constrained/checkpoints/best_model.pth}"
    WHAT="the earlier w=100 continuation (set SV_CON100=<path>)"
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
# while pointing `envs_dirs` at $HPCWORK gives `conda_envs/`. Probe the known
# layouts and check bin/python exists before `conda activate` fails obscurely.
if [[ -z "${ENV_PREFIX:-}" ]]; then
    for cand in "$HPCWORK/conda_envs/pytorch3d" \
                "$HPCWORK/conda/envs/pytorch3d" \
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

# MUST run from the repo root. run_singleview_inference.py re-derives the SMIL
# model from the path stored inside the checkpoint's own config
# (apply_smal_file_override -> open(smal_file)), and that path is RELATIVE
# ("3D_model_prep/...pkl"). From any other directory it raises FileNotFoundError
# before the first frame is decoded. There is no --smal_file override on this
# entry point, unlike run_multiview_inference.py.
cd "$REPO_DIR"

MAX_FRAMES="${MAX_FRAMES:--1}"
CROP_MODE="${CROP_MODE:-centred}"
EXPORT_MODE="${EXPORT_MODE:-side_by_side}"
CAM_SMOOTHING="${CAM_SMOOTHING:-0}"
OUT_DIR="$RENDER_ROOT/$ARM"

# --- resolve the checkpoint --------------------------------------------------
if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: no checkpoint for arm '$ARM'." >&2
    echo "       This arm needs $WHAT." >&2
    exit 1
fi
# best_model.pth is only written when val_loss beats the value RESTORED from the
# resumed checkpoint, so a short continuation can finish without ever writing
# one. Same fallback run_prior_study.sh applies, so the render and the metrics
# describe the SAME weights.
if [[ ! -e "$CHECKPOINT" && "$(basename "$CHECKPOINT")" == "best_model.pth" ]]; then
    CKPT_DIR="$(dirname "$CHECKPOINT")"
    FALLBACK="$(ls -t "$CKPT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -n1 || true)"
    if [[ -n "$FALLBACK" ]]; then
        echo "  [note] $CHECKPOINT not found (val loss never beat the resumed best)."
        echo "         Falling back to newest periodic checkpoint: $FALLBACK"
        CHECKPOINT="$FALLBACK"
    fi
fi
if [[ ! -e "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    if (( TASK >= 1 && TASK <= 4 )); then
        echo "       RUNS_ROOT is '$RUNS_ROOT'. It must match the value" >&2
        echo "       prepare_lambda_sweep.sh was run with, or this arm has not trained yet." >&2
    fi
    exit 1
fi

# --- resolve the clip list ---------------------------------------------------
CLIP_PATHS=()
if [[ -n "${CLIP_LIST:-}" ]]; then
    [[ -f "$CLIP_LIST" ]] || { echo "ERROR: CLIP_LIST not found: $CLIP_LIST" >&2; exit 1; }
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        CLIP_PATHS+=("$line")
    done < "$CLIP_LIST"
elif [[ -n "${CLIPS:-}" ]]; then
    IFS=',' read -r -a CLIP_PATHS <<< "$CLIPS"
elif [[ -n "${CLIPS_DIR:-}" ]]; then
    [[ -d "$CLIPS_DIR" ]] || { echo "ERROR: CLIPS_DIR not a directory: $CLIPS_DIR" >&2; exit 1; }
    MAX_CLIPS="${MAX_CLIPS:-3}"
    while IFS= read -r f; do CLIP_PATHS+=("$f"); done < <(
        find "$CLIPS_DIR" -maxdepth 1 -type f \
             \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) \
        | sort | head -n "$MAX_CLIPS"
    )
else
    echo "ERROR: no clips given. Set one of CLIP_LIST=<file>, CLIPS=<a.mp4,b.mp4>," >&2
    echo "       or CLIPS_DIR=<folder>. See the header of this file." >&2
    exit 1
fi

(( ${#CLIP_PATHS[@]} > 0 )) || { echo "ERROR: clip list resolved to zero files." >&2; exit 1; }

# Fail before touching the GPU if any clip is missing — a half-rendered arm
# makes an uneven comparison grid that is easy to misread.
missing=0
for c in "${CLIP_PATHS[@]}"; do
    [[ -f "$c" ]] || { echo "  [MISSING] $c" >&2; missing=1; }
done
(( missing == 0 )) || { echo "One or more clips are missing. Aborting." >&2; exit 1; }

if [[ "$CROP_MODE" == "bbox_crop" && -z "${SLEAP_PROJECT:-}" ]]; then
    echo "ERROR: CROP_MODE=bbox_crop needs SLEAP_PROJECT=<sleap project dir>." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=================================================================="
echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $TASK on $(hostname) — $(date)"
echo "  arm        : $ARM — $WHAT"
echo "  renderer   : smal_fitter.neuralSMIL.run_singleview_inference (established)"
echo "  checkpoint : $CHECKPOINT"
echo "  runs root  : $RUNS_ROOT"
echo "  clips      : ${#CLIP_PATHS[@]}"
printf '                 %s\n' "${CLIP_PATHS[@]}"
echo "  out dir    : $OUT_DIR"
echo "  max frames : $MAX_FRAMES per clip (-1 = all)"
echo "  crop mode  : $CROP_MODE   (MUST match the training preprocessing)"
echo "  export mode: $EXPORT_MODE"
echo "  save frames: ${SAVE_FRAMES:-0}"
echo "=================================================================="
python -c "import torch; print(f'  torch: {torch.__version__}, {torch.cuda.device_count()} GPU(s) visible')"

# Provenance. Without it a folder of MP4s is indistinguishable from any other
# folder of MP4s three weeks later.
python - "$OUT_DIR/render.json" "$ARM" "$CHECKPOINT" "${LAMBDA:-0}" "$CROP_MODE" "$EXPORT_MODE" <<'PY'
import json, sys, torch
out, arm, ckpt, lam, crop, mode = sys.argv[1:7]
epoch = "?"
try:
    epoch = torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", "?")
except Exception as e:      # noqa: BLE001 - provenance must never fail the render
    print(f"  [warn] could not read epoch from checkpoint: {e}")
json.dump({"arm": arm, "renderer": "run_singleview_inference.py", "checkpoint": ckpt,
           "checkpoint_epoch": epoch, "joint_limit_weight": float(lam),
           "crop_mode": crop, "video_export_mode": mode},
          open(out, "w"), indent=2)
print(f"  provenance -> {out} (checkpoint epoch {epoch})")
PY

# --- render every clip -------------------------------------------------------
FAILED=()
for CLIP in "${CLIP_PATHS[@]}"; do
    STEM="$(basename "${CLIP%.*}")"
    echo
    echo "------------------------------------------------------------------"
    echo " [$ARM] $STEM"
    echo "------------------------------------------------------------------"

    EXTRA=()
    [[ -n "${FPS:-}" ]]              && EXTRA+=(--fps "$FPS")
    [[ "${SAVE_FRAMES:-0}" == "1" ]] && EXTRA+=(--save_frames)
    [[ -n "${SLEAP_PROJECT:-}" ]]    && EXTRA+=(--sleap_project "$SLEAP_PROJECT")
    [[ -n "${SLEAP_CAMERA:-}" ]]     && EXTRA+=(--sleap_camera "$SLEAP_CAMERA")

    # --export_animation writes <stem>.npz + .json with the RAW (pre-smoothing)
    # parameters, so the same clip can be re-plotted against the authored limits
    # later without re-running the GPU.
    if python -m smal_fitter.neuralSMIL.run_singleview_inference \
        --checkpoint "$CHECKPOINT" \
        --input_video "$CLIP" \
        --output_folder "$OUT_DIR" \
        --crop_mode "$CROP_MODE" \
        --video_export_mode "$EXPORT_MODE" \
        --max_frames "$MAX_FRAMES" \
        --camera_smoothing "$CAM_SMOOTHING" \
        --export_animation "$OUT_DIR/${STEM}_anim" \
        "${EXTRA[@]}" 2>&1 | tee "$OUT_DIR/${STEM}_render.log"
    then
        echo "  -> $OUT_DIR/${STEM}_inference.mp4"
    else
        echo "  [FAILED] $STEM — see $OUT_DIR/${STEM}_render.log" >&2
        FAILED+=("$STEM")
    fi
done

echo
echo "=================================================================="
if (( ${#FAILED[@]} > 0 )); then
    echo " $ARM: ${#FAILED[@]} of ${#CLIP_PATHS[@]} clip(s) FAILED: ${FAILED[*]}" >&2
    exit 1
fi
echo " $ARM: rendered ${#CLIP_PATHS[@]} clip(s) into $OUT_DIR"
if [[ "${SAVE_FRAMES:-0}" == "1" ]]; then
    echo " Individual frames in $OUT_DIR/frames/ — check a few in sequence before"
    echo " trusting the video: the mesh should track the animal and the legs should"
    echo " swing, not jitter in place."
fi
echo
echo " Once every arm has finished, stack them on the login node:"
echo "   bash scripts/prior_study/stack_renders.sh"
echo "=================================================================="
echo "Done ($ARM) — $(date)"
