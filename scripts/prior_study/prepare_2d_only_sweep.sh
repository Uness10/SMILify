#!/usr/bin/env bash
#
# Build every config of the 2D-ONLY FROM-SCRATCH study in one go.
#
#   bash scripts/prior_study/prepare_2d_only_sweep.sh
#
# The experiment: take the training setup that produced the sv_reference
# checkpoint, drop the 3D supervision, keep the joint-limit prior, and train
# from random init. Three arms:
#
#   lambda = 0      the CONTROL. 2D-only, no prior. Without it nothing in this
#                   study has a baseline — sv_reference is 3D-supervised, so it
#                   answers a different question.
#   lambda = 1e-4   the low end of the earlier sweep
#   lambda = 1e-1   the high end
#
# Every arm is generated from the SAME base config in ONE call, so the only
# things that can differ between them are lambda and the output paths. That is
# the sweep invariant, and preflight_2d_only.py re-checks it against the written
# JSON rather than trusting this script.
#
# Env:
#   SV_BASE_CONFIG   the JSON that PRODUCED sv_reference          (REQUIRED)
#                    Not the checkpoint's embedded config block: that is written
#                    from the runtime TrainingConfig and carries stale defaults
#                    (job 15521637 got a MOUSE data_path for a STICK model that
#                    way, and trained without error).
#   RUNS_ROOT        where the per-arm run dirs go        (default: runs)
#   LAMBDAS          space-separated override             (default: 0 1e-4 1e-1)
#   SMAL_FILE        model .pkl carrying joint_limits
#                    (default: 3D_model_prep/SMILy_STICK_limits_authored.pkl)
#   DATASET          HDF5 (default: SMILySTICKS_centred_reprojected_FIXED.h5)
#   NUM_EPOCHS       override the epoch count             (default: the base
#                    config's training.num_epochs — "the same training setup")
#   BATCH_SIZE       per-PROCESS batch size (see below)
#   NUM_WORKERS      dataloader workers
#   SAVE_EVERY       output.save_checkpoint_every         (default: 2)
#   VIZ_EVERY        generate_visualizations_every        (default: 10)
#   PLOT_EVERY       plot_history_every                   (default: 10)
#   KP2D_SCALE       multiply every keypoint_2d weight    (default: 1)
#
# On BATCH_SIZE: batch_size is PER PROCESS under DDP. On 4 GPUs a config written
# for a single process runs at 4x the effective batch. Pass BATCH_SIZE=<orig/4>
# to reproduce the reference run's regime — and pass the SAME value to every arm,
# which this script guarantees by construction.
#
# On KP2D_SCALE: the reference curriculum drives keypoint_3d to weight 20 while
# keypoint_2d sits at 0.2, so removing 3D supervision cuts the total loss by
# roughly two orders of magnitude. The gradient on the 2D term is unchanged, so
# the default of 1 is the honest "same setup minus 3D". Raise it only if you
# decide the run needs it, and then raise it identically for every arm.
#
# On SAVE_EVERY: these arms train the FULL schedule from scratch, which will not
# fit in the account's 24 h MaxWall. run_2d_only_train.sbatch chains dependent
# jobs that each resume from the newest checkpoint in the arm's checkpoint_dir,
# so a small save interval is what bounds the work lost at each wall clock.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

read -r -a LAMBDA_ARR <<< "${LAMBDAS:-0 1e-4 1e-1}"
SMAL_FILE="${SMAL_FILE:-3D_model_prep/SMILy_STICK_limits_authored.pkl}"
DATASET="${DATASET:-SMILySTICKS_centred_reprojected_FIXED.h5}"
SAVE_EVERY="${SAVE_EVERY:-2}"
VIZ_EVERY="${VIZ_EVERY:-10}"
PLOT_EVERY="${PLOT_EVERY:-10}"
KP2D_SCALE="${KP2D_SCALE:-1}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
MODE="singleview"

BASE_CONFIG="${SV_BASE_CONFIG:?set SV_BASE_CONFIG=<the JSON that produced sv_reference>}"

# Refuse to write run dirs onto a quota'd home directory unless told to. These
# arms train the full schedule, so they write MORE checkpoints than the earlier
# +50-epoch sweep did, not fewer.
case "$(cd "$(dirname "$RUNS_ROOT")" 2>/dev/null && pwd || echo "$RUNS_ROOT")" in
    "$HOME"/*|"$HOME")
        if [[ "${ALLOW_HOME_OUTPUT:-0}" != "1" ]]; then
            echo "ERROR: RUNS_ROOT='$RUNS_ROOT' resolves under \$HOME ($HOME)." >&2
            echo "       A full-schedule ViT-Large run checkpointing every $SAVE_EVERY epochs is" >&2
            echo "       hundreds of GB per arm and \$HOME is quota'd — the run dies mid-epoch" >&2
            echo "       with 'Disk quota exceeded' hours in, inside imageio/torch.save." >&2
            echo "       Use:  export RUNS_ROOT=\"\$HPCWORK/smilify_runs\"" >&2
            echo "       (or set ALLOW_HOME_OUTPUT=1 if you really mean it)" >&2
            exit 1
        fi
        echo "[prep] WARNING: RUNS_ROOT is under \$HOME and ALLOW_HOME_OUTPUT=1 — watch your quota."
        ;;
esac

[[ -f "$BASE_CONFIG" ]] || { echo "ERROR: base config not found: $BASE_CONFIG" >&2; exit 1; }
[[ -f "$SMAL_FILE" ]]   || { echo "ERROR: SMAL file not found: $SMAL_FILE" >&2; exit 1; }

mkdir -p configs_runs logs

echo "=================================================================="
echo " 2D-only from-scratch study prep"
echo "  base config : $BASE_CONFIG   (the run that produced sv_reference)"
echo "  lambdas     : ${LAMBDA_ARR[*]}"
echo "  smal file   : $SMAL_FILE"
echo "  dataset     : $DATASET"
echo "  runs root   : $RUNS_ROOT"
echo "  epochs      : ${NUM_EPOCHS:-<from the base config>}"
echo "  kp2d scale  : $KP2D_SCALE"
echo "=================================================================="
echo
echo " Dropping: keypoint_3d, global_rot, joint_rot, betas, trans,"
echo "           log_beta_scales, betas_trans"
echo " Keeping : keypoint_2d, fov, cam_rot, cam_trans and every regularizer"
echo "           (the camera terms are what anchor the projection; without them"
echo "            the 2D loss is satisfied by moving the camera, not the animal)"

WRITTEN=()
for LAMBDA in "${LAMBDA_ARR[@]}"; do
    TAG="2d_lam${LAMBDA}"
    OUT="configs_runs/${MODE}_${TAG}.json"

    echo
    echo "------------------------------------------------------------------"
    echo " lambda = $LAMBDA  ->  $OUT"
    echo "------------------------------------------------------------------"

    ARGS=(
        --base-config "$BASE_CONFIG"
        --label "$TAG"
        --joint-limit-weight "$LAMBDA"
        --smal-file "$SMAL_FILE"
        --data-path "$DATASET"
        --run-dir "$RUNS_ROOT/${MODE}_${TAG}"
        --keypoint-2d-scale "$KP2D_SCALE"
        --out "$OUT"
    )
    [[ -n "${BATCH_SIZE:-}" ]]  && ARGS+=(--batch-size "$BATCH_SIZE")
    [[ -n "${NUM_WORKERS:-}" ]] && ARGS+=(--num-workers "$NUM_WORKERS")
    [[ -n "${NUM_EPOCHS:-}" ]]  && ARGS+=(--num-epochs "$NUM_EPOCHS")

    python scripts/prior_study/prepare_scratch_config.py "${ARGS[@]}"

    # -- post-patch: output cadence -------------------------------------------
    # prepare_scratch_config.py has no flags for these and the defaults are set
    # for a short continuation, not a full-schedule run. The visualisation
    # cadence is not cosmetic: the trainer calls visualize_training_progress
    # TWICE per epoch (train and val), writing num_visualization_samples PNGs
    # each time, and on a quota'd filesystem that is what actually kills a long
    # run — the exception lands in imageio.imsave, mid-epoch, hours in.
    python - "$OUT" "$SAVE_EVERY" "$VIZ_EVERY" "$PLOT_EVERY" <<'PY'
import json
import sys

path, save_every, viz_every, plot_every = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cfg = json.load(open(path))
out = cfg.setdefault("output", {})
old = (out.get("save_checkpoint_every"), out.get("generate_visualizations_every"), out.get("plot_history_every"))
out["save_checkpoint_every"] = save_every
out["generate_visualizations_every"] = viz_every
out["plot_history_every"] = plot_every
json.dump(cfg, open(path, "w"), indent=2)
print(f"[patch] save_checkpoint_every: {old[0]} -> {save_every}   "
      f"(bounds the work lost at each 24 h wall clock)")
print(f"[patch] generate_visualizations_every: {old[1]} -> {viz_every}   "
      f"(2 calls/epoch x {out.get('num_visualization_samples', '?')} PNGs when it fires)")
print(f"[patch] plot_history_every: {old[2]} -> {plot_every}")
PY

    WRITTEN+=("$OUT")
done

echo
echo "=================================================================="
echo " Wrote ${#WRITTEN[@]} config(s):"
printf '   %s\n' "${WRITTEN[@]}"
echo
echo " Next — pre-flight ALL of them on this login node before submitting."
echo " It resolves the weights through the real loader, which is the only way to"
echo " catch a curriculum stage or the scale/trans override re-enabling 3D:"
echo
PREFLIGHT="python scripts/prior_study/preflight_2d_only.py"
for f in "${WRITTEN[@]}"; do
    PREFLIGHT="$PREFLIGHT \\
    --config $f"
done
if [[ -n "${SV_REF:-}" ]]; then
    PREFLIGHT="$PREFLIGHT \\
    --reference $SV_REF"
fi
echo "$PREFLIGHT"
echo
echo " Then submit — one whole node per arm, concurrently:"
echo "   jid=\$(sbatch --parsable --array=0-$(( ${#WRITTEN[@]} - 1 )) --account=\$ACCT \\"
echo "           --export=ALL,ENV_PREFIX,RUNS_ROOT \\"
echo "           hpc_files/rwth/run_2d_only_train.sbatch)"
echo
echo " These arms train the FULL schedule from scratch and will NOT fit in 24 h."
echo " Chain the continuations up front — each job resumes from the newest"
echo " checkpoint in its own arm and exits 0 immediately once the arm is done:"
echo "   CHAIN=8 bash hpc_files/rwth/chain_2d_only.sh"
echo "=================================================================="
