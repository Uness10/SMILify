#!/usr/bin/env bash
#
# Single-view arm of the joint-limit prior study.
#
# Runs, in order:
#   1. benchmark_model              -> accuracy (MPJPE mm + PCK at two resolutions)
#   2. export_singleview_poses.py   -> predicted poses over the SAME test split (.npz/.json)
#   3. analyze_baseline_pose.py     -> angle distributions, ROM, violations, trajectories
#
# This is the single-view counterpart of run_baseline_study.sh (which is multi-view:
# it calls run_multiview_inference, whose export path does not accept an HDF5).
#
# Run it once per arm and diff the two folders:
#   LABEL=unconstrained CHECKPOINT=runs/singleview_unconstrained/checkpoints/best_model.pth \
#     bash scripts/prior_study/run_singleview_study.sh
#   LABEL=constrained   CHECKPOINT=runs/singleview_constrained/checkpoints/best_model.pth \
#     LIMITS=authored_limits.npy bash scripts/prior_study/run_singleview_study.sh
#
# Env knobs (all optional):
#   CHECKPOINT   .pth to evaluate            (default: runs/singleview_unconstrained/checkpoints/best_model.pth)
#   DATASET      HDF5                        (default: SMILySTICKS_centred_reprojected_FIXED.h5)
#   SMAL_FILE    model .pkl                  (default: 3D_model_prep/SMILy_STICK.pkl)
#   LABEL        run label                   (default: unconstrained)
#   OUT_DIR      results root                (default: prior_study_results/singleview_<LABEL>)
#   ORIG_W/H     native image size for PCK   (default: 1530/1530)
#   MAX_FRAMES   cap exported frames, 0=all  (default: 0)
#   BATCH_SIZE / NUM_WORKERS                 (default: checkpoint config / 4)
#   LIMITS       authored limits .npy/.json  (default: none -> violation table skipped)
#   SKIP_BENCH=1 / SKIP_EXPORT=1             resume a partially-completed study
#
# Prereqs: env active (conda `pytorch3d`, or the JURECA sc_venv), run from anywhere.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${CHECKPOINT:-runs/singleview_unconstrained/checkpoints/best_model.pth}"
DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK.pkl}"
LABEL="${LABEL:-unconstrained}"
ORIG_W="${ORIG_W:-1530}"
ORIG_H="${ORIG_H:-1530}"
MAX_FRAMES="${MAX_FRAMES:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
LIMITS="${LIMITS:-}"
OUT_DIR="${OUT_DIR:-prior_study_results/singleview_${LABEL}}"
SKIP_BENCH="${SKIP_BENCH:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
EXPORT_STEM="$OUT_DIR/clip_${LABEL}"

echo "=================================================================="
echo " Single-view prior study | label=$LABEL"
echo " checkpoint : $CHECKPOINT"
echo " dataset    : $DATASET"
echo " smal file  : $SMAL_FILE"
echo " out dir    : $OUT_DIR"
echo " limits     : ${LIMITS:-<none: violation table will be skipped>}"
echo "=================================================================="

# -- sanity: inputs exist -----------------------------------------------------
missing=0
for f in "$CHECKPOINT" "$DATASET" "$SMAL_FILE"; do
  [[ -e "$f" ]] || { echo "  [MISSING] $f" >&2; missing=1; }
done
if [[ "$missing" -eq 1 ]]; then
  echo >&2
  echo "One or more inputs are missing. Aborting before touching the GPU." >&2
  exit 1
fi

# -- sanity: this really is a single-view checkpoint ---------------------------
python - "$CHECKPOINT" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
kind = "multiview" if "view_embeddings.weight" in sd else "singleview"
print(f"  checkpoint kind: {kind}  (epoch {ckpt.get('epoch', '?')})")
if kind != "singleview":
    sys.exit("  ERROR: this is a multi-view checkpoint; use run_baseline_study.sh instead.")
PY

# ---------------------------------------------------------- 1. benchmark -----
CKPT_STEM="$(basename "${CHECKPOINT%.*}")"
DS_STEM="$(basename "${DATASET%.*}")"
BENCH_DIR="benchmark_singleview_${CKPT_STEM}_on_${DS_STEM}"
BENCH_REPORT=""

if [[ "$SKIP_BENCH" != "1" ]]; then
  echo; echo "[1/3] Benchmarking (MPJPE mm + PCK)..."
  BENCH_EXTRA=()
  [[ -n "$BATCH_SIZE" ]]  && BENCH_EXTRA+=(--batch_size "$BATCH_SIZE")
  [[ -n "$NUM_WORKERS" ]] && BENCH_EXTRA+=(--num_workers "$NUM_WORKERS")
  python -m smal_fitter.neuralSMIL.benchmark_model \
      --checkpoint "$CHECKPOINT" \
      --dataset_path "$DATASET" \
      --smal-file "$SMAL_FILE" \
      --orig_width "$ORIG_W" --orig_height "$ORIG_H" \
      "${BENCH_EXTRA[@]}" \
      2>&1 | tee "$OUT_DIR/benchmark_console_${STAMP}.log"
else
  echo; echo "[1/3] Benchmark SKIPPED (SKIP_BENCH=1)"
fi

# benchmark_model names its output dir from the detected model type; resolve
# whichever variant landed on disk rather than assuming the prefix.
for cand in "$BENCH_DIR" "benchmark_multiview_${CKPT_STEM}_on_${DS_STEM}"; do
  if [[ -f "$cand/benchmark_report.txt" ]]; then BENCH_DIR="$cand"; break; fi
done
if [[ -f "$BENCH_DIR/benchmark_report.txt" ]]; then
  cp -r "$BENCH_DIR" "$OUT_DIR/"
  BENCH_REPORT="$OUT_DIR/$(basename "$BENCH_DIR")/benchmark_report.txt"
  echo "  report: $BENCH_REPORT"
else
  echo "  [warn] benchmark report not found; the accuracy columns will be empty." >&2
fi

# ------------------------------------------------- 2. pose export ------------
NPZ="${EXPORT_STEM}.npz"
if [[ "$SKIP_EXPORT" != "1" ]]; then
  echo; echo "[2/3] Exporting predicted poses over the benchmark test split..."
  EXPORT_EXTRA=()
  [[ -n "$BATCH_SIZE" ]]  && EXPORT_EXTRA+=(--batch-size "$BATCH_SIZE")
  [[ -n "$NUM_WORKERS" ]] && EXPORT_EXTRA+=(--num-workers "$NUM_WORKERS")
  python scripts/prior_study/export_singleview_poses.py \
      --checkpoint "$CHECKPOINT" \
      --dataset_path "$DATASET" \
      --smal-file "$SMAL_FILE" \
      --out "$EXPORT_STEM" \
      --max-frames "$MAX_FRAMES" \
      "${EXPORT_EXTRA[@]}" \
      2>&1 | tee "$OUT_DIR/export_console_${STAMP}.log"
else
  echo; echo "[2/3] Export SKIPPED (SKIP_EXPORT=1)"
fi

if [[ ! -f "$NPZ" ]]; then
  echo "  [error] no exported .npz at $NPZ" >&2; exit 2
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
echo "   - analysis/baseline_summary.md      (headline tables)"
echo "   - analysis/joint_angle_distributions.png"
echo "   - analysis/range_of_motion.png"
echo "   - analysis/trajectories/*.png"
echo "   - analysis/*.csv                     (comparison-ready)"
echo "   - $(basename "$BENCH_DIR")/          (accuracy report + histograms)"
echo "=================================================================="
