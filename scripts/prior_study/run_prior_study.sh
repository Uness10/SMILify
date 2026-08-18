#!/usr/bin/env bash
#
# Evaluate ONE arm of the joint-limit prior study. Works for single-view and
# multi-view checkpoints alike — the modality is detected from the checkpoint.
#
# Runs, in order:
#   1. benchmark_model      -> accuracy (MPJPE mm + PCK at two resolutions)
#   2. export_poses.py      -> predicted poses over the SAME test split (.npz/.json)
#   3. analyze_baseline_pose.py -> angle distributions, ROM, limit violations, trajectories
#
# The study has four arms; run this once per arm, then join them with
# compare_arms.py:
#
#   LABEL=sv_reference   CHECKPOINT=<downloaded singleview .pth>                       bash $0
#   LABEL=sv_constrained CHECKPOINT=runs/singleview_constrained/checkpoints/best_model.pth bash $0
#   LABEL=mv_reference   CHECKPOINT=<downloaded multiview .pth>                        bash $0
#   LABEL=mv_constrained CHECKPOINT=runs/multiview_constrained/checkpoints/best_model.pth  bash $0
#
# IMPORTANT: SMAL_FILE defaults to the AUTHORED .pkl for every arm, including the
# reference ones. The reference arms were trained without limits, but they must
# still be *scored* against the authored ranges — that violation rate is the
# "before" number the whole study rests on. Scoring the reference against a .pkl
# with no joint_limits silently drops the headline table.
#
# Env knobs (all optional):
#   CHECKPOINT   .pth to evaluate                (REQUIRED in practice)
#   DATASET      HDF5                            (default: SMILySTICKS_centred_reprojected_FIXED.h5)
#   SMAL_FILE    model .pkl carrying joint_limits (default: 3D_model_prep/SMILy_STICK_limits_authored.pkl)
#   LABEL        arm label                       (default: derived from the checkpoint path)
#   OUT_DIR      results root                    (default: prior_study_results/<LABEL>)
#   ORIG_W/H     native image size for PCK       (default: 1530/1530)
#   MAX_FRAMES   cap exported frames, 0=all      (default: 0)
#   BATCH_SIZE / NUM_WORKERS                     (default: checkpoint config / 4)
#   LIMITS       limits override .npy/.json      (default: none -> read from SMAL_FILE)
#   SKIP_BENCH=1 / SKIP_EXPORT=1                 resume a partially-completed arm
#
# Prereqs: conda env `pytorch3d` active. Runs from anywhere (cds to the repo root).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=<path to .pth>}"
DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK_limits_authored.pkl}"
ORIG_W="${ORIG_W:-1530}"
ORIG_H="${ORIG_H:-1530}"
MAX_FRAMES="${MAX_FRAMES:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
LIMITS="${LIMITS:-}"
SKIP_BENCH="${SKIP_BENCH:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
# analyze_baseline_pose's DEFAULT_IMPORTANT is the six pretarsi (l_*_pt_*) —
# which are exactly the six joints left FREE (+-pi) in the authored .pkl. Featuring
# them would produce a headline distribution plot with no limit lines and nothing
# the prior can act on. Feature the coxae instead: they carry the widest authored
# ranges and are the proximal leg-placement joints the prior actually constrains.
IMPORTANT_JOINTS="${IMPORTANT_JOINTS:-l_1_co_r,l_1_co_l,l_2_co_r,l_2_co_l,l_3_co_r,l_3_co_l}"

STAMP="$(date +%Y%m%d_%H%M%S)"

# -- resolve the checkpoint ---------------------------------------------------
# best_model.pth is only written when val_loss beats the value RESTORED from the
# resumed checkpoint (train_smil_regressor.py:1303 + 2161). Continuing an
# already-converged run for a handful of epochs can therefore finish without ever
# writing one. Fall back to the newest periodic checkpoint so the study still runs.
if [[ ! -e "$CHECKPOINT" && "$(basename "$CHECKPOINT")" == "best_model.pth" ]]; then
  CKPT_DIR="$(dirname "$CHECKPOINT")"
  FALLBACK="$(ls -t "$CKPT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -n1 || true)"
  if [[ -n "$FALLBACK" ]]; then
    echo "  [note] $CHECKPOINT not found (val loss never beat the resumed best)."
    echo "         Falling back to newest periodic checkpoint: $FALLBACK"
    CHECKPOINT="$FALLBACK"
  fi
fi

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

# -- detect the modality from the checkpoint ----------------------------------
MODE="$(python - "$CHECKPOINT" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
print("multiview" if "view_embeddings.weight" in sd else "singleview")
PY
)"
CKPT_EPOCH="$(python - "$CHECKPOINT" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(ckpt.get("epoch", "?"))
PY
)"

LABEL="${LABEL:-$(basename "$(dirname "$(dirname "$CHECKPOINT")")")}"
OUT_DIR="${OUT_DIR:-prior_study_results/${LABEL}}"
mkdir -p "$OUT_DIR"
EXPORT_STEM="$OUT_DIR/clip_${LABEL}"

echo "=================================================================="
echo " Prior study arm | label=$LABEL"
echo " modality   : $MODE  (detected from the checkpoint's weights)"
echo " checkpoint : $CHECKPOINT  (epoch $CKPT_EPOCH)"
echo " dataset    : $DATASET"
echo " smal file  : $SMAL_FILE"
echo " out dir    : $OUT_DIR"
echo " limits     : ${LIMITS:-<from SMAL_FILE joint_limits>}"
echo "=================================================================="

# -- sanity: the .pkl really carries limits to score against -------------------
python - "$SMAL_FILE" "$LIMITS" <<'PY' || exit 1
import pickle, sys
import numpy as np

smal_file, limits_override = sys.argv[1], sys.argv[2]
if limits_override:
    print(f"  limits come from --limits {limits_override}; skipping the .pkl probe")
    raise SystemExit(0)
dd = pickle.load(open(smal_file, "rb"), encoding="latin1")
jl = dd.get("joint_limits")
if jl is None:
    raise SystemExit(
        f"ERROR: {smal_file} has no 'joint_limits' key.\n"
        "       analyze_baseline_pose.load_limits would find nothing and SKIP the\n"
        "       violation table — the headline number of this study. Point SMAL_FILE\n"
        "       at 3D_model_prep/SMILy_STICK_limits_authored.pkl or pass LIMITS=."
    )
jl = np.asarray(jl, float)
free = (jl[..., 0] <= -np.pi + 1e-4) & (jl[..., 1] >= np.pi - 1e-4)
n_con_nonroot = int((~free[1:]).sum())
n_axes_nonroot = free[1:].size
print(f"  authored limits: {n_con_nonroot}/{n_axes_nonroot} non-root axes constrained "
      f"({100.0 * n_con_nonroot / n_axes_nonroot:.0f}%), "
      f"{int((~free[1:]).any(axis=1).sum())} joints")
if n_con_nonroot == 0:
    raise SystemExit("ERROR: every non-root axis is wide open — nothing to score against.")
PY

# ---------------------------------------------------------- 1. benchmark -----
CKPT_STEM="$(basename "${CHECKPOINT%.*}")"
DS_STEM="$(basename "${DATASET%.*}")"
BENCH_DIR="benchmark_${MODE}_${CKPT_STEM}_on_${DS_STEM}"
BENCH_REPORT=""

if [[ "$SKIP_BENCH" != "1" ]]; then
  echo; echo "[1/3] Benchmarking (MPJPE mm + PCK)..."
  BENCH_EXTRA=()
  [[ -n "$BATCH_SIZE" ]]  && BENCH_EXTRA+=(--batch_size "$BATCH_SIZE")
  [[ -n "$NUM_WORKERS" ]] && BENCH_EXTRA+=(--num_workers "$NUM_WORKERS")
  # Multi-view: pin the view set so the two arms are scored on identical inputs.
  [[ "$MODE" == "multiview" ]] && BENCH_EXTRA+=(--no_random_view_sampling)
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
for cand in "$BENCH_DIR" "benchmark_singleview_${CKPT_STEM}_on_${DS_STEM}" \
            "benchmark_multiview_${CKPT_STEM}_on_${DS_STEM}"; do
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
  python scripts/prior_study/export_poses.py \
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
  --important-joints "$IMPORTANT_JOINTS"
  --out "$OUT_DIR/analysis"
)
[[ -n "$BENCH_REPORT" ]] && ANALYZE_ARGS+=(--benchmark "$BENCH_REPORT")
[[ -n "$LIMITS" ]] && ANALYZE_ARGS+=(--limits "$LIMITS")

python scripts/prior_study/analyze_baseline_pose.py "${ANALYZE_ARGS[@]}"

# Provenance sidecar so compare_arms.py can label the table without guessing.
cat > "$OUT_DIR/arm.json" <<JSON
{
  "label": "$LABEL",
  "mode": "$MODE",
  "checkpoint": "$CHECKPOINT",
  "checkpoint_epoch": "$CKPT_EPOCH",
  "dataset": "$DATASET",
  "smal_file": "$SMAL_FILE",
  "benchmark_report": "${BENCH_REPORT}",
  "timestamp": "$STAMP"
}
JSON

echo
echo "=================================================================="
echo " Done. Results in: $OUT_DIR"
echo "   - analysis/baseline_summary.md      (headline tables)"
echo "   - analysis/limit_violations.csv     (the before/after number)"
echo "   - analysis/joint_angle_distributions.png"
echo "   - analysis/range_of_motion.png"
echo "   - $(basename "$BENCH_DIR")/          (accuracy report + histograms)"
echo "=================================================================="
