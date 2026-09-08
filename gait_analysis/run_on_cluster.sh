#!/usr/bin/env bash
# Export the per-frame pose for each checkpoint, then run the gait analysis.
#
# Run from the SMILify repo root, in the environment you normally use for
# run_singleview_inference.py.  The only change to your existing inference
# command is  --export_animation <stem>  , which makes the run write
# <stem>.npz + <stem>.json next to the MP4 (schema in
# smal_fitter/neuralSMIL/animation_export.py).
#
#   bash gait_analysis/run_on_cluster.sh
#
set -euo pipefail

REPO="${REPO:-$(pwd)}"
OUT="${OUT:-$REPO/inference_out}"

# Each run has its own rig: sv_ref was trained against the original stick model,
# the two constrained runs against the authored-limits rig.  The analysis reads
# the rest pose and the joint hierarchy straight out of this file, so it MUST
# match the checkpoint - the wrong one silently shifts every absolute angle.
STICK_PLAIN="${STICK_PLAIN:-$REPO/3D_model_prep/SMILy_STICK.pkl}"
STICK_LIMITS="${STICK_LIMITS:-$REPO/3D_model_prep/SMILy_STICK_limits_authored.pkl}"

# frame windows of clean gait cycles, read off the SMILySTICKS footage
# (see gait_analysis/README.md).  Given in dataset frame numbers.
WINDOWS="32-196 372-495"

DATASET="${DATASET:-/path/to/SMILySTICKS_centred_reprojected_FIXED.h5}"

# name -> "<checkpoint>|<smal file>"
declare -A RUNS=(
  [sv_ref]="/path/to/sv_ref.pth|$STICK_PLAIN"
  [1e-4]="/path/to/1e-4.pth|$STICK_LIMITS"
  [1e-1]="/path/to/1e-1.pth|$STICK_LIMITS"
)

for f in "$STICK_PLAIN" "$STICK_LIMITS"; do
  [[ -f "$f" ]] || { echo "missing SMAL file: $f" >&2; exit 1; }
done

# ---- 1. inference with the animation export -------------------------------
# run_singleview_inference.py applies the checkpoint's own `smal_file`
# automatically, so --smal_file is only an override. It is passed here anyway so
# the run is explicit about which rig it used and fails loudly on a mismatch.
for name in "${!RUNS[@]}"; do
  ckpt="${RUNS[$name]%%|*}"
  smal="${RUNS[$name]##*|}"
  stem="$OUT/SMILySTICKS_centred_reprojected_FIXED_singleview_inference_${name}"
  echo "=== inference: $name   (rig: $(basename "$smal"))"
  python smal_fitter/neuralSMIL/run_singleview_inference.py \
      --checkpoint "$ckpt" \
      --smal_file "$smal" \
      --dataset "$DATASET" \
      --view_indices 0 \
      --max_frames 500 \
      --crop_mode centred \
      --output_folder "$OUT" \
      --export_animation "$stem"
done

# ---- 2. gait analysis ------------------------------------------------------
for name in "${!RUNS[@]}"; do
  smal="${RUNS[$name]##*|}"
  stem="$OUT/SMILySTICKS_centred_reprojected_FIXED_singleview_inference_${name}"
  npz=$(ls "${stem}"*.npz | head -1)          # dataset mode appends a view/frame-range suffix
  echo "=== analysis: $name   ($(basename "$npz"), rig: $(basename "$smal"))"
  python gait_analysis/analyse_gait.py \
      --npz "$npz" \
      --smal-file "$smal" \
      --windows $WINDOWS \
      --label "$name" \
      --out "$REPO/gait_figs"
done

echo
echo "figures + csv in $REPO/gait_figs"
echo "NOTE: sv_ref uses a different rig from the two constrained runs. Compare the"
echo "      'rest pose' block each analysis prints before overlaying their absolute"
echo "      angles - if the rest poses differ, compare the *relative* columns in"
echo "      angles_<run>.csv instead."
