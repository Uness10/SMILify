#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=03:00:00
#SBATCH --job-name=jl_eval
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# JOINT-LIMIT PRIOR STUDY — evaluation stage (RWTH CLAIX-2023, c23g)
#
# LAMBDA SWEEP. Scores the reference plus one arm per swept lambda, one GPU each:
#
#   task  arm                    checkpoint
#   ----  ---------------------  --------------------------------------------
#     0   sv_reference           $SV_REF (downloaded, unmodified)
#     1   sv_constrained @ 1e-4  runs/singleview_lam1e-4/checkpoints/best_model.pth
#     2   sv_constrained @ 1e-3  runs/singleview_lam1e-3/checkpoints/best_model.pth
#     3   sv_constrained @ 1e-2  runs/singleview_lam1e-2/checkpoints/best_model.pth
#     4   sv_constrained @ 1e-1  runs/singleview_lam1e-1/checkpoints/best_model.pth
#     5   mv_reference           $MV_REF (downloaded, unmodified)
#     6   mv_constrained @ 1e-4  runs/multiview_lam1e-4/checkpoints/best_model.pth
#     7   mv_constrained @ 1e-3  runs/multiview_lam1e-3/checkpoints/best_model.pth
#     8   mv_constrained @ 1e-2  runs/multiview_lam1e-2/checkpoints/best_model.pth
#     9   mv_constrained @ 1e-1  runs/multiview_lam1e-1/checkpoints/best_model.pth
#
# Default --array=0-4 is the single-view sweep. Use --array=5-9 for multi-view.
#
# ---------------------------------------------------------------------------
# RESULT LAYOUT — why each lambda gets its own subfolder
# ---------------------------------------------------------------------------
# compare_arms.py builds its delta table from the fixed pair names
# (sv_reference, sv_constrained). Rather than teach it about lambdas, each arm
# is written as `sv_constrained` inside a PER-LAMBDA root, with the single
# reference symlinked in beside it:
#
#   prior_study_results/
#     sv_reference/                     <- scored once (task 0)
#     lam1e-4/
#       sv_reference -> ../sv_reference (symlink)
#       sv_constrained/                 <- task 1
#       comparison/comparison.md        <- reference vs lambda=1e-4
#     lam1e-3/ ...                      <- task 2
#     ...
#     sweep/arms.csv                    <- one row per lambda, built at the end
#
# So `comparison.md` in each lambda folder is the usual before/after table, and
# sweep/arms.csv is the lambda-vs-metric curve.
#
# ---------------------------------------------------------------------------
# ALL arms are scored against the SAME authored .pkl, including the reference.
# The reference was trained without limits but must still be *measured* against
# those ranges — that violation rate is the "before" number the study rests on.
#
# ---------------------------------------------------------------------------
# CHECKPOINT PATHS — set these before submitting
# ---------------------------------------------------------------------------
#   SV_REF=singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth \
#     sbatch --account=rwth2151 --export=ALL,SV_REF \
#       hpc_files/rwth/run_prior_study_eval.sbatch
#
# Chained after the sweep training array:
#
#   jid=$(sbatch --parsable --array=0-3 --account=rwth2151 \
#           hpc_files/rwth/run_prior_study_train.sbatch)
#   sbatch --array=1-4 --dependency=afterok:$jid --account=rwth2151 \
#       --export=ALL,SV_REF hpc_files/rwth/run_prior_study_eval.sbatch
#
# Score the reference alone (no training needed — it is just the downloaded
# checkpoint, so this can run immediately and in parallel with the sweep):
#
#   sbatch --array=0 --account=rwth2151 --export=ALL,SV_REF \
#       hpc_files/rwth/run_prior_study_eval.sbatch
#
# Do that FIRST. Task 0's violation table is the single most informative number
# in the study: if the reference was already inside the authored ranges, the
# hinge is near-zero by construction at every lambda and the sweep has nothing
# to find.
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

# --- which arm is this array task? -------------------------------------------
# The constrained checkpoints are written by the training job into
# runs/<mode>_lam<lambda>/checkpoints/; run_prior_study.sh falls back to the
# newest periodic checkpoint if best_model.pth was never written (val loss need
# not beat the resumed best in a continuation).
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LAMBDA=""
if (( TASK == 0 )); then
    MODE="singleview"; LABEL="sv_reference"
    CHECKPOINT="${SV_REF:-}"
    WHAT="downloaded single-view checkpoint (set SV_REF=<path>)"
    OUT_DIR="$RESULTS_ROOT/sv_reference"
elif (( TASK >= 1 && TASK <= 4 )); then
    MODE="singleview"; LABEL="sv_constrained"
    LAMBDA="${LAMBDAS[$((TASK - 1))]}"
    CHECKPOINT="${SV_CON:-runs/singleview_lam${LAMBDA}/checkpoints/best_model.pth}"
    WHAT="single-view continuation at lambda=$LAMBDA"
    OUT_DIR="$RESULTS_ROOT/lam${LAMBDA}/sv_constrained"
elif (( TASK == 5 )); then
    MODE="multiview"; LABEL="mv_reference"
    CHECKPOINT="${MV_REF:-}"
    WHAT="downloaded multi-view checkpoint (set MV_REF=<path>)"
    OUT_DIR="$RESULTS_ROOT/mv_reference"
elif (( TASK >= 6 && TASK <= 9 )); then
    MODE="multiview"; LABEL="mv_constrained"
    LAMBDA="${LAMBDAS[$((TASK - 6))]}"
    CHECKPOINT="${MV_CON:-runs/multiview_lam${LAMBDA}/checkpoints/best_model.pth}"
    WHAT="multi-view continuation at lambda=$LAMBDA"
    OUT_DIR="$RESULTS_ROOT/lam${LAMBDA}/mv_constrained"
else
    echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$TASK (expected 0-9)." >&2
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

cd "$REPO_DIR"

# The authored .pkl is the scoring reference for EVERY arm, including the
# reference. Pointing this at SMILy_STICK.pkl (no joint_limits) would make
# analyze_baseline_pose silently skip the violation table.
export SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK_limits_authored.pkl}"
export DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"

echo "=================================================================="
echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $TASK on $(hostname) — $(date)"
echo "  arm        : $LABEL${LAMBDA:+ @ lambda=$LAMBDA} — $WHAT"
echo "  checkpoint : ${CHECKPOINT:-<UNSET>}"
echo "  out dir    : $OUT_DIR"
echo "  dataset    : $DATASET"
echo "  smal file  : $SMAL_FILE"
echo "=================================================================="

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: no checkpoint for arm '$LABEL'." >&2
    echo "       This arm needs $WHAT." >&2
    echo "       Re-submit with --export=ALL,SV_REF (and MV_REF for tasks 5-9)." >&2
    exit 1
fi

# run_prior_study.sh re-checks existence (and applies the best_model.pth
# fallback), but failing here gives a clearer message per arm.
if [[ ! -e "$CHECKPOINT" && "$(basename "$CHECKPOINT")" != "best_model.pth" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

# LABEL stays the canonical pair name ("sv_constrained") so compare_arms.py can
# pair it with the reference; OUT_DIR is what separates the lambdas.
#
# BENCH_TAG must ALSO be unique per arm. benchmark_model names its output
# directory from getcwd() + the checkpoint stem, and every swept arm resumes into
# a file called best_model.pth — so without this, two lambdas benchmarking at the
# same time write into the same benchmark_singleview_best_model_on_*/ directory
# and quietly interleave their reports.
BENCH_TAG="${LABEL}${LAMBDA:+_lam$LAMBDA}"

LABEL="$LABEL" CHECKPOINT="$CHECKPOINT" OUT_DIR="$OUT_DIR" BENCH_TAG="$BENCH_TAG" \
  bash scripts/prior_study/run_prior_study.sh

# --- join reference + this lambda -------------------------------------------
# Symlink (not copy) the reference into this lambda's root so compare_arms.py
# finds the canonical pair. One scored reference, reused by every lambda —
# copying it would invite the two halves of a delta to drift apart.
if [[ -n "$LAMBDA" ]]; then
    REF_LABEL="${LABEL%%_constrained}_reference"   # sv_constrained -> sv_reference
    LAM_ROOT="$RESULTS_ROOT/lam${LAMBDA}"
    if [[ -d "$RESULTS_ROOT/$REF_LABEL" ]]; then
        ln -sfn "../$REF_LABEL" "$LAM_ROOT/$REF_LABEL"
    else
        echo "[eval] NOTE: $RESULTS_ROOT/$REF_LABEL not scored yet — the delta table for" >&2
        echo "       lambda=$LAMBDA will show the constrained row only. Run --array=0 (or 5)" >&2
        echo "       and re-run compare_arms.py; no need to re-run this arm." >&2
    fi

    echo
    echo "[eval] comparison for lambda=$LAMBDA ..."
    python scripts/prior_study/compare_arms.py \
        --results-root "$LAM_ROOT" \
        --out "$LAM_ROOT/comparison" \
        --extra-epochs "${EXTRA_EPOCHS:-50}" \
        --joint-limit-weight "$LAMBDA" || \
      echo "[eval] (comparison incomplete — expected until the reference has run)"
fi

# --- the sweep table ---------------------------------------------------------
# Every task attempts this; whichever finishes last produces the complete curve.
# Pure CSV/text parsing, so racing is cheap enough not to bother avoiding.
echo
echo "[eval] refreshing the sweep table from whatever arms are on disk..."
SWEEP_ARGS=(--results-root "$RESULTS_ROOT" --out "$RESULTS_ROOT/sweep" --extra-epochs "${EXTRA_EPOCHS:-50}")
for L in "${LAMBDAS[@]}"; do
    for A in sv_constrained mv_constrained; do
        [[ -d "$RESULTS_ROOT/lam${L}/$A" ]] && SWEEP_ARGS+=(--arm "$RESULTS_ROOT/lam${L}/$A")
    done
done
for A in sv_reference mv_reference; do
    [[ -d "$RESULTS_ROOT/$A" ]] && SWEEP_ARGS+=(--arm "$RESULTS_ROOT/$A")
done
python scripts/prior_study/compare_arms.py "${SWEEP_ARGS[@]}" || \
  echo "[eval] (sweep table incomplete — expected until every arm has run)"

# compare_arms.py keys arms by the label in arm.json, and every lambda's arm is
# labelled "sv_constrained" — so in the FLAT sweep CSV they would collide. Rebuild
# that one file from the per-lambda dirs, keyed by lambda, which is the actual
# deliverable of a sweep.
python - "$RESULTS_ROOT" "${LAMBDAS[@]}" <<'PY'
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from scripts.prior_study.compare_arms import load_arm, ACC_KEYS  # noqa: E402

root, lambdas = Path(sys.argv[1]), sys.argv[2:]
rows = []
for label in ("sv_reference", "mv_reference"):
    rec = load_arm(root / label)
    if rec:
        rec["lambda"] = 0.0          # the reference IS lambda = 0, un-continued
        rows.append(rec)
for lam in lambdas:
    for arm in ("sv_constrained", "mv_constrained"):
        rec = load_arm(root / f"lam{lam}" / arm)
        if rec:
            rec["lambda"] = float(lam)
            rows.append(rec)

if not rows:
    print("[sweep] nothing scored yet")
    raise SystemExit(0)

rows.sort(key=lambda r: (r.get("mode", "?"), r["lambda"]))
out = root / "sweep"
out.mkdir(parents=True, exist_ok=True)
fields = ["mode", "lambda", "label", "checkpoint_epoch", "n_axes", "violating_axes",
          "mean_viol_rate", "mean_overshoot_deg", "max_overshoot_deg", *ACC_KEYS,
          "checkpoint", "dir"]
with open(out / "sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

def fmt(v, spec=".2f"):
    return "—" if v is None else (v if isinstance(v, str) else format(v, spec))

lines = ["# Joint-limit prior — lambda sweep", "",
         "`lambda = 0` is the reference checkpoint scored as-is: it received **zero**",
         "additional epochs, so every row below differs from it by *the prior plus the",
         "continuation*, not the prior alone. Comparisons BETWEEN lambdas are clean —",
         "they share the epoch count — so read the sweep as a curve first and the",
         "reference row as context.", "",
         "| mode | lambda | violating axes | mean viol. rate % | mean overshoot deg | MPJPE mm | PCK@5px (native) |",
         "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(
        f"| {r.get('mode','?')} | {r['lambda']:g} | "
        f"{fmt(r.get('violating_axes'), 'd') if r.get('violating_axes') is not None else '—'}"
        f"/{fmt(r.get('n_axes'), 'd') if r.get('n_axes') is not None else '—'} | "
        f"{fmt(r.get('mean_viol_rate'))} | {fmt(r.get('mean_overshoot_deg'))} | "
        f"{fmt(r.get('mpjpe_mm'))} | {fmt(r.get('pck_5px_native'), '.4f')} |"
    )
lines += ["", "Lower is better for violations, overshoot and MPJPE; higher for PCK.",
          "A lambda that lowers violations while leaving MPJPE/PCK flat is the win",
          "condition; one that lowers both violations and PCK is the prior overriding",
          "the data, which is what the low end of this sweep is meant to avoid.", ""]
(out / "sweep.md").write_text("\n".join(lines))
print(f"[sweep] wrote {out/'sweep.csv'} and {out/'sweep.md'} ({len(rows)} arm(s))")
print("\n".join(lines))
PY

echo "Done ($LABEL${LAMBDA:+ @ $LAMBDA}) — $(date)"
