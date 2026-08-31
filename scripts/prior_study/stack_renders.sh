#!/usr/bin/env bash
#
# Turn the per-arm renders into ONE labelled comparison video per clip, so the
# reference and every swept lambda are watched side by side on the same frames.
#
#   bash scripts/prior_study/stack_renders.sh
#
# Reads   prior_study_results/renders/<arm>/<segment>.mp4
# Writes  prior_study_results/renders/comparison/<segment>_compare.mp4
#
# The segments come from pick_segments.py and are identical across arms, so
# panel k of every frame is the same instant of the same animal.
#
# Login node, ffmpeg only — no GPU, seconds per clip.
#
# Layout: up to 3 arms in a row, more than 3 in a 2-row grid. Panels are scaled
# to a common height first: the arms render at identical settings so they should
# already match, but an --image-size that differed between render runs would
# otherwise make xstack fail with a cryptic filter error.
#
# Env:
#   RENDER_ROOT   default: prior_study_results/renders
#   ARMS          space-separated arm folder names, in display order
#                 default: sv_reference lam1e-4 lam1e-3 lam1e-2 lam1e-1 lam100
#                 (arms with no renders on disk are dropped, so this works
#                 before the sweep has trained)
#   CLIPS         space-separated segment names (default: every segment found
#                 under the FIRST arm folder)
#   PANEL_H       panel height in px (default: 480)
#   CRF           x264 quality, lower = better (default: 20)
#   LABELS_ON=1   also burn an ffmpeg caption per panel. Off by default because
#                 render_clip_npz.py already draws one (arm, frame, violations).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RENDER_ROOT="${RENDER_ROOT:-prior_study_results/renders}"
# lam100 (the earlier w=100 run) is last so the panels read left-to-right in
# increasing prior strength. Arms with no renders are dropped, so before the
# sweep has trained this default already gives you reference | w=100.
read -r -a ARM_ARR <<< "${ARMS:-sv_reference lam1e-4 lam1e-3 lam1e-2 lam1e-1 lam100}"
PANEL_H="${PANEL_H:-480}"
CRF="${CRF:-20}"
OUT_DIR="$RENDER_ROOT/comparison"

command -v ffmpeg >/dev/null 2>&1 || {
    echo "ERROR: ffmpeg not found." >&2
    echo "       On the login node:  module load FFmpeg   (module spider ffmpeg to find it)" >&2
    echo "       Or in the study env: conda install -c conda-forge ffmpeg" >&2
    exit 1
}

# --- which arms actually rendered? -------------------------------------------
PRESENT=()
for arm in "${ARM_ARR[@]}"; do
    if [[ -d "$RENDER_ROOT/$arm" ]] && compgen -G "$RENDER_ROOT/$arm/seg*.mp4" >/dev/null; then
        PRESENT+=("$arm")
    else
        echo "  [skip] $arm — no renders under $RENDER_ROOT/$arm/"
    fi
done
# Only a hard stop here: the real check is against the CURRENT segments, below,
# which can give a much more useful message.
if (( ${#PRESENT[@]} < 1 )); then
    echo "ERROR: no arm under $RENDER_ROOT/ has any renders at all." >&2
    echo "       Run the render job first:" >&2
    echo "         sbatch --array=0,5 --account=rwth2151 hpc_files/rwth/run_prior_study_render.sbatch" >&2
    exit 1
fi

# --- which segments? ---------------------------------------------------------
# Prefer the names in segments.json over globbing the arm folder. Re-running
# pick_segments.py changes the names (they carry the camera and first frame), so
# a glob would pick up MP4s left over from an earlier selection and try to stack
# a segment only some arms have. The JSON is the current truth.
SEGMENTS_FILE="${SEGMENTS_FILE:-$RENDER_ROOT/segments.json}"
CLIP_STEMS=()
if [[ -n "${CLIPS:-}" ]]; then
    read -r -a CLIP_STEMS <<< "$CLIPS"
elif [[ -f "$SEGMENTS_FILE" ]]; then
    while IFS= read -r n; do CLIP_STEMS+=("$n"); done < <(
        python -c "import json,sys; [print(s['name']) for s in json.load(open(sys.argv[1]))['segments']]" \
            "$SEGMENTS_FILE"
    )
    echo "  segments from $SEGMENTS_FILE"
else
    for f in "$RENDER_ROOT/${PRESENT[0]}"/seg*.mp4; do
        b="$(basename "$f")"
        CLIP_STEMS+=("${b%.mp4}")
    done
fi

# Keep only arms that actually hold renders of the CURRENT segments. An arm whose
# whole folder predates the current segments.json has not been re-rendered yet —
# reporting that once here beats one "missing" warning per segment followed by an
# epilogue about videos that were never written.
CURRENT=()
for arm in "${PRESENT[@]}"; do
    have=0 stale=0
    for f in "$RENDER_ROOT/$arm"/seg*.mp4; do
        b="$(basename "$f")"; b="${b%.mp4}"
        if [[ " ${CLIP_STEMS[*]} " == *" $b "* ]]; then have=$((have + 1)); else stale=$((stale + 1)); fi
    done
    (( stale > 0 )) && echo "  [note] $arm has $stale render(s) from an earlier segments.json — ignored."
    if (( have > 0 )); then
        CURRENT+=("$arm")
    else
        echo "  [skip] $arm — nothing rendered for the current segments yet."
    fi
done
PRESENT=("${CURRENT[@]}")

if (( ${#PRESENT[@]} < 2 )); then
    echo >&2
    echo "ERROR: need 2+ arms rendered for the CURRENT segments; found ${#PRESENT[@]}." >&2
    if [[ -f "$SEGMENTS_FILE" ]]; then
        echo "       $SEGMENTS_FILE lists: ${CLIP_STEMS[*]}" >&2
        echo >&2
        echo "       If you just submitted the render job, it has not finished — this script" >&2
        echo "       reads finished MP4s, it does not wait. Check it, then re-run:" >&2
        echo "         squeue --me" >&2
        echo "         tail -n 40 logs/jl_render_<jobid>_0.out" >&2
        echo "       If pick_segments.py was re-run after the last render, the arms need" >&2
        echo "       re-rendering against the new windows:" >&2
        echo "         sbatch --array=0,5 --account=rwth2151 hpc_files/rwth/run_prior_study_render.sbatch" >&2
    fi
    exit 1
fi

mkdir -p "$OUT_DIR"
PRODUCED=0
echo "=================================================================="
echo " Stacking renders"
echo "  arms  : ${PRESENT[*]}"
echo "  clips : ${CLIP_STEMS[*]}"
echo "  out   : $OUT_DIR"
echo "=================================================================="

# lam100 predates the sweep and trained for far fewer epochs than the sweep arms
# will. Side by side in one grid that difference is invisible, so say it here
# rather than letting the grid imply the panels differ only in lambda.
_has_lam100=0
_has_sweep=0
for a in "${PRESENT[@]}"; do
    [[ "$a" == "lam100" ]] && _has_lam100=1
    [[ "$a" == lam1e-* ]] && _has_sweep=1
done
if (( _has_lam100 == 1 && _has_sweep == 1 )); then
    echo
    echo " NOTE: lam100 is the EARLIER run and is NOT epoch-matched to the sweep arms."
    echo "       Reading its panel against theirs mixes the weight with the training"
    echo "       length. Compare it against sv_reference (which it was continued from),"
    echo "       and compare the sweep arms among themselves."
    echo "       Each arm's render.json records the checkpoint it came from."
    echo
fi

# Panel caption: the arm folder name plus what it means.
label_for() {
    case "$1" in
        sv_reference|mv_reference) echo "reference (no limit prior)" ;;
        lam100)                    echo "lambda = 100 (older run)" ;;
        lam*)                      echo "lambda = ${1#lam}" ;;
        *)                         echo "$1" ;;
    esac
}

for STEM in "${CLIP_STEMS[@]}"; do
    INPUTS=()
    LABELS=()
    for arm in "${PRESENT[@]}"; do
        f="$RENDER_ROOT/$arm/${STEM}.mp4"
        if [[ -f "$f" ]]; then
            INPUTS+=("$f")
            LABELS+=("$(label_for "$arm")")
        else
            echo "  [warn] $arm is missing ${STEM}.mp4 — dropping it from this segment's grid" >&2
        fi
    done

    n=${#INPUTS[@]}
    if (( n < 2 )); then
        echo "  [skip] $STEM — only $n arm(s) rendered it"
        continue
    fi

    # Grid shape: a single row up to 3 panels, otherwise two rows.
    if (( n <= 3 )); then
        COLS=$n
    else
        COLS=$(( (n + 1) / 2 ))
    fi
    ROWS=$(( (n + COLS - 1) / COLS ))

    # Build the filtergraph: scale each input to a common height, burn in its
    # label, then xstack into the grid.
    # render_clip_npz.py already burns the arm label, frame index and per-frame
    # violation count into each panel, so drawtext here would double up. Set
    # LABELS=1 to add it anyway (e.g. for panels rendered with --no-hud).
    FILTER=""
    for ((i = 0; i < n; i++)); do
        FILTER+="[${i}:v]scale=-2:${PANEL_H},"
        if [[ "${LABELS_ON:-0}" == "1" ]]; then
            FILTER+="drawtext=text='${LABELS[$i]//\'/}':x=10:y=10:fontsize=22:fontcolor=white:"
            FILTER+="box=1:boxcolor=black@0.6:boxborderw=6,"
        fi
        FILTER+="setsar=1[v${i}];"
    done

    # xstack needs explicit panel origins. Panels share a height; widths can
    # differ, so lay columns out at w0/w1/... offsets rather than assuming a
    # uniform width.
    LAYOUT=""
    for ((i = 0; i < n; i++)); do
        r=$(( i / COLS )); c=$(( i % COLS ))
        if (( c == 0 )); then x="0"; else
            x=""
            for ((k = 0; k < c; k++)); do
                idx=$(( r * COLS + k ))
                x+="${x:+ +}w${idx}"
            done
            x="${x// /}"
        fi
        y=$(( r * PANEL_H ))
        LAYOUT+="${LAYOUT:+|}${x}_${y}"
    done

    for ((i = 0; i < n; i++)); do FILTER+="[v${i}]"; done
    FILTER+="xstack=inputs=${n}:layout=${LAYOUT}:fill=black[out]"

    FF_IN=()
    for f in "${INPUTS[@]}"; do FF_IN+=(-i "$f"); done

    OUT="$OUT_DIR/${STEM}_compare.mp4"
    echo
    echo "  $STEM: ${n} panel(s), ${ROWS}x${COLS} -> $OUT"
    if ffmpeg -y -loglevel error -stats "${FF_IN[@]}" \
        -filter_complex "$FILTER" -map "[out]" \
        -c:v libx264 -crf "$CRF" -pix_fmt yuv420p "$OUT"
    then
        echo "  ok"
        PRODUCED=$((PRODUCED + 1))
    else
        echo "  [FAILED] $STEM" >&2
        echo "           If the error mentions drawtext, this ffmpeg lacks libfreetype —" >&2
        echo "           drop LABELS_ON (the panels already carry their own HUD), or" >&2
        echo "           conda install -c conda-forge ffmpeg, which includes it." >&2
    fi
done

if (( PRODUCED == 0 )); then
    echo >&2
    echo "ERROR: no comparison video was written." >&2
    echo "       Every segment had fewer than 2 arms rendered against the current" >&2
    echo "       segments.json. See the per-segment lines above." >&2
    exit 1
fi

echo
echo "=================================================================="
echo " $PRODUCED comparison video(s) in $OUT_DIR"
echo
echo " Each panel carries its own HUD: arm label, frame index, and how many of the"
echo " 129 authored joint-axes are out of range IN THAT FRAME. Watch that counter"
echo " across panels — it is the same quantity the violation table aggregates."
echo
echo " What to look for, in the order it is worth looking:"
echo "   1. The out-of-range counter falling from left to right while the pose"
echo "      still tracks the animal — that is the prior working."
echo "   2. Legs that stop moving, or a body that drifts off the animal, at the"
echo "      HIGH end of the sweep — that is the prior overriding the data, and it"
echo "      should show up as worse MPJPE/PCK in sweep/sweep.md too."
echo "   3. No visible difference at any lambda, with the violation table also"
echo "      flat: the reference was already inside the authored ranges, so there"
echo "      was nothing for the hinge to fix. Check sv_reference's"
echo "      analysis/limit_violations.csv before concluding anything else."
echo "=================================================================="
