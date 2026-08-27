#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --job-name=jl_render
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# JOINT-LIMIT PRIOR STUDY — qualitative renders (RWTH CLAIX-2023, c23g)
#
# Renders the SAME video clips through every arm so the change the prior makes
# can be judged by eye, not only through the violation table. One array task per
# arm, each looping over every clip:
#
#   task  arm                    checkpoint
#   ----  ---------------------  --------------------------------------------
#     0   sv_reference           $SV_REF (downloaded, unmodified)
#     1   sv_constrained @ 1e-4  runs/singleview_lam1e-4/checkpoints/best_model.pth
#     2   sv_constrained @ 1e-3  runs/singleview_lam1e-3/checkpoints/best_model.pth
#     3   sv_constrained @ 1e-2  runs/singleview_lam1e-2/checkpoints/best_model.pth
#     4   sv_constrained @ 1e-1  runs/singleview_lam1e-1/checkpoints/best_model.pth
#
# Output (one folder per arm, one MP4 + one animation .npz/.json per clip):
#
#   prior_study_results/renders/
#     sv_reference/<clip>_inference.mp4        <- side-by-side: input | fitted mesh
#     sv_reference/<clip>_anim.npz             <- raw predicted parameters
#     lam1e-4/<clip>_inference.mp4
#     ...
#
# Then stack them into one comparison video per clip on the LOGIN NODE (ffmpeg
# only, no GPU, seconds):
#
#   bash scripts/prior_study/stack_renders.sh
#
# ---------------------------------------------------------------------------
# WHICH CLIPS
# ---------------------------------------------------------------------------
# Give the clips one of three ways (checked in this order):
#
#   CLIP_LIST=clips.txt      a file with one absolute clip path per line
#                            (lines starting with # are ignored)
#   CLIPS="a.mp4,b.mp4"      comma-separated paths
#   CLIPS_DIR=/path/to/clips all *.mp4/*.avi/*.mov directly inside that folder,
#                            capped at MAX_CLIPS (default 4)
#
# Pick clips that show the behaviour the prior is supposed to fix — a walking
# sequence where the reference bends a coxa the wrong way is worth ten clips of
# the animal standing still. The violation table in
# prior_study_results/sv_reference/analysis/limit_violations.csv names the axes
# that violate most; choose clips where those joints are moving.
#
# ---------------------------------------------------------------------------
# SUBMIT
# ---------------------------------------------------------------------------
#   SV_REF=<downloaded singleview .pth> \
#   CLIPS_DIR=$HPCWORK/clips \
#     sbatch --account=rwth2151 --export=ALL,SV_REF,CLIPS_DIR \
#       hpc_files/rwth/run_prior_study_render.sbatch
#
# Reference only, right now (no training needed) — do this FIRST to confirm the
# clips decode and the crop mode is right before spending four more GPU hours:
#
#   SV_REF=... CLIPS_DIR=... MAX_FRAMES=120 \
#     sbatch --array=0 --account=rwth2151 --export=ALL,SV_REF,CLIPS_DIR,MAX_FRAMES \
#       hpc_files/rwth/run_prior_study_render.sbatch
#
# Chained after the sweep training array:
#
#   sbatch --array=0-4 --dependency=afterok:$jid --account=rwth2151 \
#       --export=ALL,SV_REF,CLIPS_DIR hpc_files/rwth/run_prior_study_render.sbatch
#
# Env knobs:
#   MAX_FRAMES      frames per clip, -1 = all          (default: 300)
#   FPS             output fps, empty = same as input  (default: input's)
#   CROP_MODE       centred | default | bbox_crop      (default: centred)
#   EXPORT_MODE     side_by_side | overlay             (default: side_by_side)
#   CAM_SMOOTHING   camera moving-average window       (default: 0 = off)
#   RENDER_ROOT     output root  (default: prior_study_results/renders)
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

TASK="${SLURM_ARRAY_TASK_ID:-0}"
LAMBDA=""
if (( TASK == 0 )); then
    ARM="sv_reference"
    CHECKPOINT="${SV_REF:-}"
    WHAT="downloaded single-view checkpoint (set SV_REF=<path>)"
elif (( TASK >= 1 && TASK <= 4 )); then
    LAMBDA="${LAMBDAS[$((TASK - 1))]}"
    ARM="lam${LAMBDA}"
    CHECKPOINT="${SV_CON:-runs/singleview_lam${LAMBDA}/checkpoints/best_model.pth}"
    WHAT="single-view continuation at lambda=$LAMBDA"
else
    echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$TASK (expected 0-4)." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-$HPCWORK/hf_cache}"
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

# MUST run from the repo root. run_singleview_inference.py re-derives the SMIL
# model from the path stored inside the checkpoint's own config
# (apply_smal_file_override -> open(smal_file)), and that path is RELATIVE
# ("3D_model_prep/SMILy_STICK*.pkl"). From any other directory it raises
# FileNotFoundError before the first frame is decoded. There is no --smal_file
# override on this entry point, unlike run_multiview_inference.py.
cd "$REPO_DIR"

MAX_FRAMES="${MAX_FRAMES:-300}"
CROP_MODE="${CROP_MODE:-centred}"
EXPORT_MODE="${EXPORT_MODE:-side_by_side}"
CAM_SMOOTHING="${CAM_SMOOTHING:-0}"
RENDER_ROOT="${RENDER_ROOT:-prior_study_results/renders}"
OUT_DIR="$RENDER_ROOT/$ARM"

# --- resolve the checkpoint --------------------------------------------------
if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: no checkpoint for arm '$ARM'." >&2
    echo "       This arm needs $WHAT." >&2
    exit 1
fi
# best_model.pth is only written when val_loss beats the value RESTORED from the
# resumed checkpoint, so a continuation can finish without ever writing one.
# Same fallback run_prior_study.sh applies, so the render and the metrics score
# the SAME weights.
if [[ ! -e "$CHECKPOINT" && "$(basename "$CHECKPOINT")" == "best_model.pth" ]]; then
    CKPT_DIR="$(dirname "$CHECKPOINT")"
    FALLBACK="$(ls -t "$CKPT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -n1 || true)"
    if [[ -n "$FALLBACK" ]]; then
        echo "  [note] $CHECKPOINT not found (val loss never beat the resumed best)."
        echo "         Falling back to newest periodic checkpoint: $FALLBACK"
        CHECKPOINT="$FALLBACK"
    fi
fi
[[ -e "$CHECKPOINT" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1; }

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
    MAX_CLIPS="${MAX_CLIPS:-4}"
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

if (( ${#CLIP_PATHS[@]} == 0 )); then
    echo "ERROR: clip list resolved to zero files." >&2
    exit 1
fi

# Fail before touching the GPU if any clip is missing — a half-rendered arm
# makes an uneven comparison grid that is easy to misread.
missing=0
for c in "${CLIP_PATHS[@]}"; do
    [[ -f "$c" ]] || { echo "  [MISSING] $c" >&2; missing=1; }
done
(( missing == 0 )) || { echo "One or more clips are missing. Aborting." >&2; exit 1; }

mkdir -p "$OUT_DIR"

echo "=================================================================="
echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $TASK on $(hostname) — $(date)"
echo "  arm        : $ARM${LAMBDA:+ (lambda=$LAMBDA)} — $WHAT"
echo "  checkpoint : $CHECKPOINT"
echo "  clips      : ${#CLIP_PATHS[@]}"
printf '                 %s\n' "${CLIP_PATHS[@]}"
echo "  out dir    : $OUT_DIR"
echo "  max frames : $MAX_FRAMES per clip"
echo "  crop mode  : $CROP_MODE   (MUST match the training preprocessing)"
echo "  export mode: $EXPORT_MODE"
echo "=================================================================="
python -c "import torch; print(f'  torch: {torch.__version__}, {torch.cuda.device_count()} GPU(s) visible')"

# Record which weights produced these frames. Without it, a folder of MP4s is
# indistinguishable from any other folder of MP4s three weeks later.
python - "$OUT_DIR/render.json" "$ARM" "$CHECKPOINT" "$LAMBDA" "$MAX_FRAMES" "$CROP_MODE" <<'PY'
import json, sys, torch
out, arm, ckpt, lam, max_frames, crop = sys.argv[1:7]
epoch = "?"
try:
    epoch = torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", "?")
except Exception as e:      # noqa: BLE001 - provenance must never fail the render
    print(f"  [warn] could not read epoch from checkpoint: {e}")
json.dump({"arm": arm, "checkpoint": ckpt, "checkpoint_epoch": epoch,
           "joint_limit_weight": float(lam) if lam else 0.0,
           "max_frames": int(max_frames), "crop_mode": crop},
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
    [[ -n "${FPS:-}" ]] && EXTRA+=(--fps "$FPS")

    # --export_animation writes <stem>.npz + .json with the RAW (pre-smoothing)
    # parameters. Keep them: they let the same clip be re-plotted against the
    # authored limits later without re-running the GPU.
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
    echo " The stacked comparison will be missing this arm for those clips." >&2
    exit 1
fi
echo " $ARM: rendered ${#CLIP_PATHS[@]} clip(s) into $OUT_DIR"
echo
echo " Once every arm has finished, build the comparison videos on the login node:"
echo "   bash scripts/prior_study/stack_renders.sh"
echo "=================================================================="
echo "Done ($ARM) — $(date)"
