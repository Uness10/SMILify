#!/usr/bin/env bash
#
# End-to-end baseline study for the UNCONSTRAINED neural stick-insect model.
#
# Runs, in order:
#   1. benchmark_model            -> accuracy (MPJPE mm, PCK) + report + histograms
#   2. run_multiview_inference    -> rendered videos + exported scene params (.npz/.json)
#   3. analyze_baseline_pose.py   -> angle distributions, ROM, limit-violation tables,
#                                    trajectories, comparison-ready summary
#
# This establishes the "before" column. Re-run later with a constrained checkpoint
# (joint_limit_regularization on, or authored joint_limits in the .pkl) and diff the
# two output folders to quantify what the user-defined prior buys.
#
# Paths default to the GETTING_STARTED.md layout (files downloaded to repo root).
# Override any of them via environment variables or edit below.
#
# Usage:
#   bash scripts/prior_study/run_baseline_study.sh
#   CHECKPOINT=my.pth DATASET=my.h5 LABEL=unconstrained bash scripts/prior_study/run_baseline_study.sh
#
# Prereqs: conda env `pytorch3d` active; run from repo root.
set -euo pipefail

# ------------------------------------------------------------------ config ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${CHECKPOINT:-SMILySTICKS_ViT_model.pth}"
DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK.pkl}"
LABEL="${LABEL:-unconstrained}"
ORIG_W="${ORIG_W:-1530}"
ORIG_H="${ORIG_H:-1530}"
MAX_FRAMES="${MAX_FRAMES:-0}"        # 0 = all frames for the export; set e.g. 300 for a quick pass
LIMITS="${LIMITS:-}"                 # optional authored-limits .npy/.json to enable violation table
OUT_DIR="${OUT_DIR:-prior_study_results/${LABEL}}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
EXPORT_STEM="$OUT_DIR/clip_${LABEL}"

echo "=================================================================="
echo " Baseline study | label=$LABEL"
echo " checkpoint : $CHECKPOINT"
echo " dataset    : $DATASET"
echo " smal file  : $SMAL_FILE"
echo " out dir    : $OUT_DIR"
echo "=================================================================="

# -- sanity: required inputs exist -------------------------------------------
missing=0
for f in "$CHECKPOINT" "$DATASET" "$SMAL_FILE"; do
  if [[ ! -e "$f" ]]; then
    echo "  [MISSING] $f" >&2; missing=1
  fi
done
if [[ "$missing" -eq 1 ]]; then
  cat >&2 <<'EOF'

One or more inputs are missing. Download the example checkpoint + dataset per
GETTING_STARTED.md (Google Drive, into the repo root), or point CHECKPOINT= /
DATASET= at your own files. Aborting before touching the GPU.
EOF
  exit 1
fi

# ---------------------------------------------------------- 1. benchmark -----
echo; echo "[1/3] Benchmarking (accuracy: MPJPE mm + PCK)..."
python -m smal_fitter.neuralSMIL.benchmark_model \
    --checkpoint "$CHECKPOINT" \
    --dataset_path "$DATASET" \
    --smal-file "$SMAL_FILE" \
    --orig_width "$ORIG_W" --orig_height "$ORIG_H" \
    2>&1 | tee "$OUT_DIR/benchmark_console_${STAMP}.log"

# benchmark_model writes benchmark_multiview_<ckpt>_on_<dataset>/ next to CWD.
CKPT_STEM="$(basename "${CHECKPOINT%.*}")"
DS_STEM="$(basename "${DATASET%.*}")"
BENCH_DIR="benchmark_multiview_${CKPT_STEM}_on_${DS_STEM}"
BENCH_REPORT="${BENCH_DIR}/benchmark_report.txt"
if [[ -f "$BENCH_REPORT" ]]; then
  cp -r "$BENCH_DIR" "$OUT_DIR/"       # keep the report + histograms with the study
  BENCH_REPORT="$OUT_DIR/${BENCH_DIR}/benchmark_report.txt"
else
  echo "  [warn] benchmark report not found at expected path; accuracy scrape will be empty." >&2
  BENCH_REPORT=""
fi

# ------------------------------------------------ 2. inference + export ------
echo; echo "[2/3] Inference + exporting scene parameters..."
MF_ARG=()
if [[ "$MAX_FRAMES" -gt 0 ]]; then MF_ARG=(--max_frames "$MAX_FRAMES"); fi
python -m smal_fitter.neuralSMIL.run_multiview_inference \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --smal_file "$SMAL_FILE" \
    --export_animation "$EXPORT_STEM" \
    "${MF_ARG[@]}" \
    2>&1 | tee "$OUT_DIR/inference_console_${STAMP}.log"

# run_multiview_inference may suffix the export stem with a frame range; resolve it.
NPZ="$(ls -t "${EXPORT_STEM}"*.npz 2>/dev/null | head -n1 || true)"
if [[ -z "$NPZ" ]]; then
  echo "  [error] no exported .npz found for stem $EXPORT_STEM" >&2; exit 2
fi
JSON="${NPZ%.npz}.json"
echo "  exported: $NPZ"

# --------------------------------------------------------- 3. analysis -------
echo; echo "[3/3] Analysing poses (distributions, ROM, violations, trajectories)..."
ANALYZE_ARGS=(
  --npz "$NPZ"
  --json "$JSON"
  --smal-file "$SMAL_FILE"
  --label "$LABEL"
  --out "$OUT_DIR/analysis"
)
[[ -n "$BENCH_REPORT" ]] && ANALYZE_ARGS+=(--benchmark "$BENCH_REPORT")
[[ -n "$LIMITS" ]] && ANALYZE_ARGS+=(--limits "$LIMITS")

python scripts/prior_study/analyze_baseline_pose.py "${ANALYZE_ARGS[@]}"

echo
echo "=================================================================="
echo " Done. Results in: $OUT_DIR"
echo "   - analysis/baseline_summary.md          (headline tables)"
echo "   - analysis/joint_angle_distributions.png"
echo "   - analysis/range_of_motion.png"
echo "   - analysis/trajectories/*.png"
echo "   - analysis/*.csv                          (comparison-ready)"
echo "   - ${BENCH_DIR}/                           (accuracy report + histograms)"
echo "=================================================================="
