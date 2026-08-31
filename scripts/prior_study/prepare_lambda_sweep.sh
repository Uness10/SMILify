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
#   LR_FLAT                  pin optimizer.learning_rate and optimizer.lr_schedule
#                            to this single value for the whole continuation.
#                            Unset = keep the config's schedule (and the script
#                            WARNS if that schedule changes mid-window).
#
# Why LR_FLAT exists: the learning rate is a pure function of the epoch INDEX
# (train_smil_regressor.py:2079) with no early stopping and no plateau logic, so
# a continuation from a late checkpoint inherits whatever stage the absolute
# epoch number lands in. The shipped OptimizerConfig defaults step 350 -> 1e-6
# and 400 -> 1e-5 — a 10x INCREASE — so resuming from epoch 386 for 50 epochs
# crosses that boundary at epoch 400 and trains the last 36 epochs at 10x the LR
# the reference checkpoint ended on. LR_FLAT=1e-6 holds the LR at the value in
# force at the resume epoch, which makes reference-vs-lambda interpretable.
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
# Visualisation / plot cadence. prepare_resume_config.set_output_dirs defaults
# BOTH to 1 (prepare_resume_config.py:253), and the trainer then calls
# visualize_training_progress TWICE per epoch — train and val — writing
# num_visualization_samples PNGs each time, plus a history plot
# (train_smil_regressor.py:2214/2228/2240). Over 50 epochs x 4 arms that is
# thousands of PNGs nobody looks at, and on a quota'd filesystem it is what
# actually kills the run: the exception lands in imageio.imsave, mid-epoch,
# AFTER the GPUs have been busy for hours.
VIZ_EVERY="${VIZ_EVERY:-10}"
PLOT_EVERY="${PLOT_EVERY:-10}"

# Refuse to write run dirs onto a quota'd home directory unless told to.
case "$(cd "$(dirname "$RUNS_ROOT")" 2>/dev/null && pwd || echo "$RUNS_ROOT")" in
    "$HOME"/*|"$HOME")
        if [[ "${ALLOW_HOME_OUTPUT:-0}" != "1" ]]; then
            echo "ERROR: RUNS_ROOT='$RUNS_ROOT' resolves under \$HOME ($HOME)." >&2
            echo "       Four arms x ~25 periodic ViT-Large checkpoints is ~100 GB and \$HOME" >&2
            echo "       is quota'd — the run dies mid-epoch with 'Disk quota exceeded' hours in." >&2
            echo "       Use:  export RUNS_ROOT=\"\$HPCWORK/smilify_runs\"" >&2
            echo "       (or set ALLOW_HOME_OUTPUT=1 if you really mean it)" >&2
            exit 1
        fi
        echo "[prep] WARNING: RUNS_ROOT is under \$HOME and ALLOW_HOME_OUTPUT=1 — watch your quota."
        ;;
esac

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
    python - "$OUT" "$SAVE_EVERY" "$LAMBDA" "$EXTRA_EPOCHS" "${LR_FLAT:-}" "$VIZ_EVERY" "$PLOT_EVERY" <<'PY'
import importlib.util
import json
import pathlib
import sys

path, save_every, lam = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
extra_epochs, lr_flat = int(sys.argv[4]), sys.argv[5]
viz_every, plot_every = int(sys.argv[6]), int(sys.argv[7])
cfg = json.load(open(path))


def effective_lr_schedule(cfg):
    """The LR curriculum the trainer will ACTUALLY use, defaults included.

    The JSON key is `optimizer.lr_schedule` (epoch -> lr), NOT
    `learning_rate_curriculum` — that name only exists in the legacy
    TrainingConfig mirror the loader writes at train_smil_regressor.py:2434.
    A config with no `optimizer` block inherits OptimizerConfig's dataclass
    defaults (base_config.py:160), which is easy to mistake for "no schedule".
    Load that class straight from the file rather than importing the package, so
    this stays cheap and free of import side effects.
    """
    opt = cfg.get("optimizer") or {}
    sched, base = opt.get("lr_schedule"), opt.get("learning_rate")
    source = "config optimizer.lr_schedule"
    if not sched:
        source = "OptimizerConfig defaults (no optimizer.lr_schedule in the config)"
        bc = pathlib.Path("smal_fitter/neuralSMIL/configs/base_config.py")
        spec = importlib.util.spec_from_file_location("_base_config_probe", bc)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        defaults = mod.OptimizerConfig()
        sched = defaults.lr_schedule
        base = defaults.learning_rate if base is None else base
    return {int(k): float(v) for k, v in sched.items()}, float(base), source


def lr_at(epoch, sched, base):
    lr = base
    for threshold in sorted(sched):
        if epoch >= threshold:
            lr = sched[threshold]
    return lr


out = cfg.setdefault("output", {})
old = out.get("save_checkpoint_every")
out["save_checkpoint_every"] = save_every
old_viz, old_plot = out.get("generate_visualizations_every"), out.get("plot_history_every")
out["generate_visualizations_every"] = viz_every
out["plot_history_every"] = plot_every

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

# -- learning rate over the continuation window -------------------------------
# The LR is a pure function of the epoch INDEX (train_smil_regressor.py:2079) —
# there is no early stopping and no plateau scheduler, so whatever the schedule
# says for epochs [start, end) is what this arm gets. Continuing from a late
# checkpoint can walk straight through a stage boundary: the shipped defaults
# step 350 -> 1e-6 and 400 -> 1e-5, a 10x INCREASE, which for a resume from
# epoch 386 lands in the middle of a +50 epoch window. Identical across arms, so
# lambda-vs-lambda stays clean — but reference-vs-lambda then mixes the prior
# with an LR jump, so it must not be discovered after the fact.
start = end - extra_epochs if end is not None else None
if start is not None:
    sched, base, source = effective_lr_schedule(cfg)
    window = [(e, lr_at(e, sched, base)) for e in range(start, end)]
    changes = [(e, lr) for i, (e, lr) in enumerate(window) if i == 0 or lr != window[i - 1][1]]
    print(f"[patch] lr source: {source}")
    if lr_flat:
        flat = float(lr_flat)
        opt = cfg.setdefault("optimizer", {})
        opt["learning_rate"] = flat
        opt["lr_schedule"] = {"0": flat}
        was = " -> ".join(f"{lr:.3g}@e{e}" for e, lr in changes)
        print(f"[patch] LR PINNED to {flat:g} for the whole window (LR_FLAT); was {was}")
    elif len(changes) > 1:
        print(f"[patch] WARNING: the LR CHANGES inside this arm's window [{start}, {end}):")
        for e, lr in changes:
            prev = dict(changes).get(e)
            print(f"[patch]            epoch {e}: lr -> {lr:.3g}")
        print(f"[patch]          Every arm gets the same jump, so lambda-vs-lambda is still")
        print(f"[patch]          clean, but reference-vs-lambda now mixes the prior with an LR")
        print(f"[patch]          change. Re-run with LR_FLAT={changes[0][1]:g} to hold the LR at")
        print(f"[patch]          the value in force at the resume epoch.")
    else:
        print(f"[patch] lr constant at {changes[0][1]:.3g} across [{start}, {end}) — nothing to confound")

json.dump(cfg, open(path, "w"), indent=2)
print(f"[patch] save_checkpoint_every: {old} -> {save_every}")
print(f"[patch] generate_visualizations_every: {old_viz} -> {viz_every}   "
      f"(2 calls/epoch x {out.get('num_visualization_samples', '?')} PNGs when it fires)")
print(f"[patch] plot_history_every: {old_plot} -> {plot_every}")
if stripped:
    print(f"[patch] removed joint_limit_regularization overrides from curriculum_stages "
          f"at epoch(s) {sorted(stripped)} — lambda={lam} now holds for the whole run")
print(f"[patch] epochs (training.num_epochs): [{start}, {end})")
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
