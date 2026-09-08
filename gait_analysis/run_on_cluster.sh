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
SMAL="${SMAL:-$REPO/3D_model_prep/SMILy_STICK.pkl}"

# frame windows of clean gait cycles, read off the SMILySTICKS footage
# (see gait_analysis/README.md).  Given in dataset frame numbers.
WINDOWS="32-196 372-495"

# ---- 1. inference with the animation export -------------------------------
# Fill in the flags you already use for these three runs; everything except
# --export_animation should be identical to the commands that produced the MP4s.
declare -A RUNS=(
  [sv_ref]="--checkpoint /path/to/sv_ref.pth"
  [1e-4]="--checkpoint /path/to/1e-4.pth"
  [1e-1]="--checkpoint /path/to/1e-1.pth"
)

for name in "${!RUNS[@]}"; do
  stem="$OUT/SMILySTICKS_centred_reprojected_FIXED_singleview_inference_${name}"
  echo "=== inference: $name"
  python smal_fitter/neuralSMIL/run_singleview_inference.py \
      ${RUNS[$name]} \
      --dataset /path/to/SMILySTICKS_centred_reprojected_FIXED.h5 \
      --view_indices 0 \
      --max_frames 500 \
      --crop_mode centred \
      --output_folder "$OUT" \
      --export_animation "$stem"
done

# ---- 2. gait analysis ------------------------------------------------------
for name in "${!RUNS[@]}"; do
  stem="$OUT/SMILySTICKS_centred_reprojected_FIXED_singleview_inference_${name}"
  npz=$(ls "${stem}"*.npz | head -1)          # dataset mode appends a view/frame-range suffix
  echo "=== analysis: $name  ($npz)"
  python gait_analysis/analyse_gait.py \
      --npz "$npz" \
      --smal-file "$SMAL" \
      --windows $WINDOWS \
      --label "$name" \
      --out "$REPO/gait_figs"
done

echo
echo "figures + csv in $REPO/gait_figs"
