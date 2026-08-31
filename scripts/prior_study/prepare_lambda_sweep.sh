#!/usr/bin/env bash
#
# Build every config of the joint-limit LAMBDA SWEEP in one go.
#
#   bash scripts/prior_study/prepare_lambda_sweep.sh [singleview|multiview]
#
# Calls prepare_resume_config.py once per lambda in {1e-4, 1e-3, 1e-2, 1e-1},
# writing configs_runs/<mode>_lam<lambda>.json — the exact names
# run_prior_study_train.sbatch derives from its array task id.
#
# Every arm gets the SAME --extra-epochs. That is the point: a sweep whose arms
# trained for different lengths measures lambda and training length at once.
#
# Env:
#   SV_REF / MV_REF          checkpoint to continue from       (REQUIRED)
#   SV_BASE_CONFIG / MV_BASE_CONFIG
#                            the JSON that PRODUCED that checkpoint. Omit and
#                            prepare_resume_config.py auto-discovers via the
#                            <name>_checkpoints/ -> <name>.json convention;
#                            it will NOT fall back to the checkpoint's embedded
#                            config, which carries stale defaults.
#   EXTRA_EPOCHS             additional epochs per arm         (default: 50)
#   LAMBDAS                  space-separated override          (default: 1e-4 1e-3 1e-2 1e-1)
#   SMAL_FILE                model .pkl carrying joint_limits
#                            (default: 3D_model_prep/SMILy_STICK_limits_authored.pkl)
#   DATASET                  HDF5 (default: SMILySTICKS_centred_reprojected_FIXED.h5)
#   BATCH_SIZE               override training.batch_size (see note below)
#   SAVE_EVERY               output.save_checkpoint_every      (default: 2)
#
# Why SAVE_EVERY defaults to 2: the account's MaxWall is 24 h and 50 epochs may
# not fit. With periodic checkpoints every 2 epochs, a timeout costs at most two
# epochs and the arm can be continued with another prepare_resume_config.py call
# against the newest checkpoint_epoch_*.pth.
#
# Why you may want BATCH_SIZE: batch_size is PER PROCESS under DDP, so on 4 GPUs
# the continuation runs at 4x the effective batch of the run that produced the
# checkpoint while inheriting that run's LR curriculum. Pass BATCH_SIZE=<orig/4>
# to match the original regime, or accept 4x knowingly — but use the SAME value
# for every lambda.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-singleview}"
case "$MODE" in
    singleview|multiview) ;;
    *) echo "usage: $0 [singleview|multiview]" >&2; exit 1 ;;
esac

EXTRA_EPOCHS="${EXTRA_EPOCHS:-50}"
read -r -a LAMBDA_ARR <<< "${LAMBDAS:-1e-4 1e-3 1e-2 1e-1}"
SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK_limits_authored.pkl}"
DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
SAVE_EVERY="${SAVE_EVERY:-2}"
# Where the per-arm run dirs (checkpoints/plots/visualizations) go. The eval
# sbatch reads the SAME variable to find each arm's checkpoints, so these two no
# longer have to be kept in sync by hand: exporting RUNS_ROOT once before both
# steps is enough. Default `runs` keeps the old behaviour.
#
# Set RUNS_ROOT="$HPCWORK/smilify_runs" if you submit from a checkout on $HOME:
# 4 lambdas x ~25 periodic ViT-Large checkpoints is on the order of 100 GB and
# $HOME's quota will stop the run partway through with a write error.
RUNS_ROOT="${RUNS_ROOT:-runs}"

if [[ "$MODE" == "singleview" ]]; then
    CKPT="${SV_REF:?set SV_REF=<path to the single-view .pth>}"
    BASE_CONFIG="${SV_BASE_CONFIG:-}"
else
    CKPT="${MV_REF:?set MV_REF=<path to the multi-view .pth>}"
    BASE_CONFIG="${MV_BASE_CONFIG:-smal_fitter/neuralSMIL/configs/examples/multiview_SMILySTICKS_3D_ViT_Large_AUG_FIXED.json}"
fi

[[ -f "$CKPT" ]]      || { echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$SMAL_FILE" ]] || { echo "ERROR: SMAL file not found: $SMAL_FILE" >&2; exit 1; }

mkdir -p configs_runs logs

echo "=================================================================="
echo " Lambda sweep prep | mode=$MODE"
echo "  checkpoint  : $CKPT"
echo "  base config : ${BASE_CONFIG:-<auto-discover>}"
echo "  lambdas     : ${LAMBDA_ARR[*]}"
echo "  extra epochs: $EXTRA_EPOCHS   (identical for every arm — the sweep invariant)"
echo "  smal file   : $SMAL_FILE"
echo "  dataset     : $DATASET"
echo "  runs root   : $RUNS_ROOT  (export the same RUNS_ROOT for the eval array)"
echo "=================================================================="

WRITTEN=()
for LAMBDA in "${LAMBDA_ARR[@]}"; do
    TAG="lam${LAMBDA}"
    OUT="configs_runs/${MODE}_${TAG}.json"

    echo
    echo "------------------------------------------------------------------"
    echo " lambda = $LAMBDA  ->  $OUT"
    echo "------------------------------------------------------------------"

    ARGS=(
        --checkpoint "$CKPT"
        --extra-epochs "$EXTRA_EPOCHS"
        --label "$TAG"
        --joint-limit-weight "$LAMBDA"
        --smal-file "$SMAL_FILE"
        --data-path "$DATASET"
        --run-dir "$RUNS_ROOT/${MODE}_${TAG}"
        --out "$OUT"
    )
    [[ -n "$BASE_CONFIG" ]]          && ARGS+=(--base-config "$BASE_CONFIG")
    [[ -n "${BATCH_SIZE:-}" ]]       && ARGS+=(--batch-size "$BATCH_SIZE")
    [[ -n "${NUM_WORKERS:-}" ]]      && ARGS+=(--num-workers "$NUM_WORKERS")

    python scripts/prior_study/prepare_resume_config.py "${ARGS[@]}"

    # -- post-patch: checkpoint cadence + curriculum hygiene -------------------
    # prepare_resume_config.py has no flag for save_checkpoint_every, and it does
    # not inspect curriculum_stages for entries that would overwrite the swept
    # weight partway through the run. Both matter more at 50 epochs than at 10.
    python - "$OUT" "$SAVE_EVERY" "$LAMBDA" <<'PY'
import json, sys

path, save_every, lam = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
cfg = json.load(open(path))

out = cfg.setdefault("output", {})
old = out.get("save_checkpoint_every")
out["save_checkpoint_every"] = save_every

# A curriculum stage that sets joint_limit_regularization inside this run's
# epoch range would discard the swept lambda mid-run. Strip those keys (and only
# those) so the sweep measures what it says it measures. The training sbatch
# refuses to launch if any survive.
tr = cfg.get("training", {})
end = tr.get("num_epochs")
stages = cfg.get("loss_curriculum", {}).get("curriculum_stages", {}) or {}
stripped = []
for ep, d in stages.items():
    if isinstance(d, dict) and "joint_limit_regularization" in d:
        d.pop("joint_limit_regularization")
        stripped.append(int(ep))

json.dump(cfg, open(path, "w"), indent=2)
print(f"[patch] save_checkpoint_every: {old} -> {save_every}")
if stripped:
    print(f"[patch] removed joint_limit_regularization overrides from curriculum_stages "
          f"at epoch(s) {sorted(stripped)} — lambda={lam} now holds for the whole run")
print(f"[patch] end epoch (training.num_epochs): {end}")
PY

    WRITTEN+=("$OUT")
done

echo
echo "=================================================================="
echo " Wrote ${#WRITTEN[@]} config(s):"
printf '   %s\n' "${WRITTEN[@]}"
echo
echo " Next — pre-flight ALL of them on this login node before submitting:"
echo
PREFLIGHT="python scripts/prior_study/preflight_study.py"
for f in "${WRITTEN[@]}"; do
    PREFLIGHT="$PREFLIGHT \\
    --config $f --reference $CKPT"
done
echo "$PREFLIGHT"
echo
echo " Then submit all ${#WRITTEN[@]} arms at once — no % throttle, so every arm gets"
echo " its own 4-GPU node concurrently:"
if [[ "$MODE" == "singleview" ]]; then
    ARRAY_SPEC="0-3"
else
    ARRAY_SPEC="4-7"
fi
echo "   jid=\$(sbatch --parsable --array=$ARRAY_SPEC --account=rwth2151 \\"
echo "           --export=ALL,ENV_PREFIX,RUNS_ROOT \\"
echo "           hpc_files/rwth/run_prior_study_train.sbatch)"
echo
echo " Add --nodes=2 to double the ranks per arm (halves the wall clock, doubles"
echo " the effective batch — use the same value for every arm)."
echo
echo " The eval array must see the SAME RUNS_ROOT:"
echo "   RUNS_ROOT=$RUNS_ROOT sbatch --array=1-4 --dependency=afterany:\$jid \\"
echo "           --account=rwth2151 --export=ALL,SV_REF,ENV_PREFIX,RUNS_ROOT,EXTRA_EPOCHS \\"
echo "           hpc_files/rwth/run_prior_study_eval.sbatch"
echo "=================================================================="
