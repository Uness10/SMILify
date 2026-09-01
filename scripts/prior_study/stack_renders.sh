#!/usr/bin/env bash
#
# Turn the per-arm renders into ONE labelled comparison video per clip, so the
# reference and every swept lambda are watched side by side on the same frames.
#
#   bash scripts/prior_study/stack_renders.sh
#
# Reads   prior_study_results/renders/<arm>/<clip>_inference.mp4
# Writes  prior_study_results/renders/comparison/<clip>_compare.mp4
#
# Every arm renders the SAME clips end to end via run_singleview_inference.py,
# so panel k of every frame is the same moment of the same animal.
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
#   CLIPS         space-separated clip stems (default: every clip found under
#                 the FIRST arm folder)
#   PANEL_H       panel height in px (default: 480)
#   CRF           x264 quality, lower = better (default: 20)
#   LABELS_ON=0   drop the per-panel ffmpeg caption (on by default; the
#                 inference renderer draws none of its own; needs libfreetype).

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
    if [[ -d "$RENDER_ROOT/$arm" ]] && compgen -G "$RENDER_ROOT/$arm/*_inference.mp4" >/dev/null; then
        PRESENT+=("$arm")
    else
        echo "  [skip] $arm — no renders under $RENDER_ROOT/$arm/"
    fi
done
# Only a hard stop here; the 2-arm check below gives a more useful message.
if (( ${#PRESENT[@]} < 1 )); then
    echo "ERROR: no arm under $RENDER_ROOT/ has any renders at all." >&2
    echo "       Run the render job first:" >&2
    echo "         sbatch --array=0-4 --account=rwth2151 hpc_files/rwth/run_prior_study_render.sbatch" >&2
    exit 1
fi

# --- which clips? ------------------------------------------------------------
CLIP_STEMS=()
if [[ -n "${CLIPS:-}" ]]; then
    read -r -a CLIP_STEMS <<< "$CLIPS"
else
    for f in "$RENDER_ROOT/${PRESENT[0]}"/*_inference.mp4; do
        b="$(basename "$f")"
        CLIP_STEMS+=("${b%_inference.mp4}")
    done
fi

if (( ${#PRESENT[@]} < 2 )); then
    echo >&2
    echo "ERROR: need 2+ arms with renders to compare; found ${#PRESENT[@]}." >&2
    echo "       If you just submitted the render job it has not finished — this script" >&2
    echo "       reads finished MP4s, it does not wait:" >&2
    echo "         squeue --me" >&2
    echo "         tail -n 40 logs/jl_render_<jobid>_0.out" >&2
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
        f="$RENDER_ROOT/$arm/${STEM}_inference.mp4"
        if [[ -f "$f" ]]; then
            INPUTS+=("$f")
            LABELS+=("$(label_for "$arm")")
        else
            echo "  [warn] $arm is missing ${STEM}_inference.mp4 — dropping it from this clip's grid" >&2
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
    # run_singleview_inference.py draws no caption, so label each panel here.
    # LABELS_ON=0 turns it off (e.g. if this ffmpeg lacks libfreetype).
    FILTER=""
    for ((i = 0; i < n; i++)); do
        FILTER+="[${i}:v]scale=-2:${PANEL_H},"
        if [[ "${LABELS_ON:-1}" == "1" ]]; then
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
        echo "           set LABELS_ON=0 to drop the captions, or" >&2
        echo "           conda install -c conda-forge ffmpeg, which includes it." >&2
    fi
done

if (( PRODUCED == 0 )); then
    echo >&2
    echo "ERROR: no comparison video was written." >&2
    echo "       Every clip had fewer than 2 arms rendered. See the lines above." >&2
    exit 1
fi

echo
echo "=================================================================="
echo " $PRODUCED comparison video(s) in $OUT_DIR"
echo
echo " Panels are labelled by arm."
echo
echo " What to look for, in the order it is worth looking:"
echo "   1. A leg segment that folds the wrong way in the reference panel and"
echo "      stops doing so as lambda rises, while the mesh still sits on the"
echo "      animal — that is the prior working."
echo "   2. Legs that stop moving, or a body that drifts off the animal, at the"
echo "      HIGH end of the sweep — that is the prior overriding the data, and it"
echo "      should show up as worse MPJPE/PCK in sweep/sweep.md too."
echo "   3. No visible difference at any lambda, with the violation table also"
echo "      flat: the reference was already inside the authored ranges, so there"
echo "      was nothing for the hinge to fix. Check sv_reference's"
echo "      analysis/limit_violations.csv before concluding anything else."
echo "=================================================================="
