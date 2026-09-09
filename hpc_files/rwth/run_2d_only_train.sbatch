#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=c23g
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --job-name=jl2d_train
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#
# =============================================================================
# 2D-ONLY FROM-SCRATCH STUDY — single-view, joint-limit prior, no 3D supervision
# =============================================================================
#
# Array task -> arm:
#
#     0   lambda = 0      configs_runs/singleview_2d_lam0.json     <- CONTROL
#     1   lambda = 1e-4   configs_runs/singleview_2d_lam1e-4.json
#     2   lambda = 1e-1   configs_runs/singleview_2d_lam1e-1.json
#
# Override the lambda list with LAMBDAS="0 1e-4 1e-1" (same order as the array
# indices) and the config path with CONFIG=<file> for a one-off.
#
# The dataset defaults to $DATA_ROOT/SMILySTICKS_centred_reprojected_FIXED.h5
# with DATA_ROOT=$HPCWORK/SMILify. Override DATA_ROOT (directory) or DATA_PATH
# (full file path) if the .h5 lives somewhere else, and remember to add it to
# --export when submitting: --export=ALL,ENV_PREFIX,RUNS_ROOT,DATA_PATH
#
# HOW THIS DIFFERS FROM run_prior_study_train.sbatch
# --------------------------------------------------
# That script trains CONTINUATIONS and enforces it: it exits if
# resume_checkpoint is null and it exits if the limit weight is 0. Both are
# correct there and both are wrong here.
#
#   * These arms train FROM SCRATCH, so the config carries no resume_checkpoint
#     and the run starts at epoch 0.
#   * lambda = 0 is a legitimate arm — it is the control, and without it the
#     study has no baseline (sv_reference is 3D-supervised, so it answers a
#     different question).
#   * The full schedule will not fit in the account's 24 h MaxWall, so this
#     script AUTO-RESUMES: if the arm's checkpoint_dir already holds
#     checkpoints, it continues from the newest one. That makes a chain of
#     dependent jobs (hpc_files/rwth/chain_2d_only.sh) do the right thing with
#     no config edits, and makes a job that lands after the arm is finished
#     exit 0 immediately instead of re-training it.
#   * The pre-flight asserts, through the REAL loader, that no 3D supervision
#     survives at any epoch. Reading the JSON is not enough: a curriculum stage
#     can re-set a zeroed weight, and the scale/trans mode override is applied
#     after the curriculum (training_config.py:534) and can resurrect betas,
#     log_beta_scales and betas_trans regardless of base_weights.
#
# Submit (from the repo root on $HPCWORK, after prepare + preflight):
#
#   jid=$(sbatch --parsable --array=0-2 --account=$ACCT \
#           --export=ALL,ENV_PREFIX,RUNS_ROOT \
#           hpc_files/rwth/run_2d_only_train.sbatch)
#
# Monitor:  squeue --me   /   tail -f logs/jl2d_train_<jobid>_<taskid>.out
# Cancel :  scancel <jobid>
#
# NOTE: intentionally no `set -u`. Conda's activate hooks
# (libblas_mkl_activate.sh) reference unset vars like MKL_INTERFACE_LAYER and
# abort under set -u.
set -eo pipefail
# ---------------------------------------------------------------------------

# --- account guard -----------------------------------------------------------
if [[ -z "${SLURM_JOB_ACCOUNT:-}" ]]; then
    echo "ERROR: no SLURM account. Submit with:" >&2
    echo "  sbatch --account=rwth2151 $0" >&2
    exit 1
fi

# --- which arm is this array task? -------------------------------------------
read -r -a LAMBDA_ARR <<< "${LAMBDAS:-0 1e-4 1e-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK < 0 || TASK >= ${#LAMBDA_ARR[@]} )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK is outside 0-$(( ${#LAMBDA_ARR[@]} - 1 ))." >&2
    echo "       Arms: ${LAMBDA_ARR[*]}" >&2
    exit 1
fi
LAMBDA="${LAMBDA_ARR[$TASK]}"
MODE="singleview"
TRAIN_MODULE="smal_fitter.neuralSMIL.train_smil_regressor"
TAG="2d_lam${LAMBDA}"
CONFIG="${CONFIG:-configs_runs/${MODE}_${TAG}.json}"

# --- dataset path, resolved dynamically (not trusted from the JSON) --------
# h5py.File() resolves whatever string it gets against the process CWD at the
# moment train_smil_regressor.py opens it. A bare filename baked into the
# config only works by accident if the job happens to `cd` into the one
# directory that has it, and a stale/copied config can silently point at the
# wrong file. Resolve an absolute path here and pass it with --data_path,
# which overrides the config (CLI > JSON per configs/base_config.py's
# precedence), so this arm's dataset never depends on the submit directory.
DATA_ROOT="${DATA_ROOT:-$HPCWORK/SMILify}"
DATA_PATH="${DATA_PATH:-$DATA_ROOT/SMILySTICKS_centred_reprojected_FIXED.h5}"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: dataset not found at $DATA_PATH" >&2
    echo "       Set DATA_ROOT (directory) or DATA_PATH (full file path) explicitly, e.g.:" >&2
    echo "         DATA_PATH=/path/to/file.h5 sbatch --export=ALL,ENV_PREFIX,RUNS_ROOT,DATA_PATH ... $0" >&2
    exit 1
fi

# Backbone weights must already be cached (hpc_files/download_backbone_weights.py
# on the login node); c23g compute nodes have no outbound network. This matters
# MORE here than for the continuation study: these arms build a fresh
# ViT-Large, so the very first thing they do is ask for pretrained weights.
export HF_HOME="${HF_HOME:-$HPCWORK/hf_cache}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- how many GPUs did this task actually get? -------------------------------
# Derived from the allocation, never hard-coded. Slurm already sets
# CUDA_VISIBLE_DEVICES correctly for the allocation, so do not touch it.
# Sources, in order of authority: explicit override, what the scheduler gave us,
# what the node has, this script's own header (which is what makes a login-node
# dry run print the shape you are about to submit rather than 1).
_is_pos_int() { [[ "$1" =~ ^[0-9]+$ ]] && (( $1 > 0 )); }

NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
GPU_SRC="GPUS_PER_NODE env"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"

if ! _is_pos_int "$GPUS_PER_NODE"; then
    GPU_SRC="SLURM_GPUS_ON_NODE"
    GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-}"
fi
if ! _is_pos_int "$GPUS_PER_NODE" && [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    GPU_SRC="SLURM_JOB_GPUS"
    GPUS_PER_NODE="$(tr ',' '\n' <<< "$SLURM_JOB_GPUS" | grep -c .)"
fi
if ! _is_pos_int "$GPUS_PER_NODE" && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_SRC="nvidia-smi"
    GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
fi
if ! _is_pos_int "$GPUS_PER_NODE"; then
    GPU_SRC="#SBATCH --gres header"
    GPUS_PER_NODE="$(grep -m1 -oP '^#SBATCH\s+--gres=gpu:\K[0-9]+' "$0" 2>/dev/null || true)"
    if _is_pos_int "$GPUS_PER_NODE"; then
        if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_PARTITION:-}" != *login* && "${SLURM_JOB_ID}" != "000000" ]]; then
            echo "WARNING: no GPU variables in this allocation; assuming the header's" >&2
            echo "         $GPUS_PER_NODE GPU(s)/node. Re-submit with GPUS_PER_NODE=<n> if wrong." >&2
        else
            echo "[no allocation] login node / dry run; using the header's --gres=gpu:$GPUS_PER_NODE." >&2
        fi
    fi
fi
if ! _is_pos_int "$GPUS_PER_NODE"; then
    echo "WARNING: could not determine the GPU count from any source; falling back to 1." >&2
    echo "         This arm would run at 1/4 the effective batch and would NOT be" >&2
    echo "         comparable with the other arms. Re-submit with GPUS_PER_NODE=4." >&2
    GPU_SRC="fallback"
    GPUS_PER_NODE=1
fi
WORLD=$(( NNODES * GPUS_PER_NODE ))

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"

# --- resolve the conda env ---------------------------------------------------
if [[ -z "${ENV_PREFIX:-}" ]]; then
    for cand in "$HPCWORK/conda/envs/pytorch3d" \
                "$HPCWORK/conda_envs/pytorch3d" \
                "$HPCWORK/miniconda3/envs/pytorch3d" \
                "$HPCWORK/miniforge3/envs/pytorch3d" \
                "$HOME/.conda/envs/pytorch3d"; do
        [[ -x "$cand/bin/python" ]] && { ENV_PREFIX="$cand"; break; }
    done
fi
if [[ ! -x "${ENV_PREFIX:-}/bin/python" ]]; then
    echo "ERROR: no conda env found." >&2
    echo "       ENV_PREFIX='${ENV_PREFIX:-<unset>}' has no bin/python, and none of the" >&2
    echo "       known layouts under HPCWORK=$HPCWORK matched either." >&2
    echo "       Find yours on the login node:  conda env list" >&2
    echo "       Then:  ENV_PREFIX=<prefix> sbatch --export=ALL,ENV_PREFIX ... $0" >&2
    exit 1
fi

# --- activate conda ----------------------------------------------------------
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

echo "=================================================================="
echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $TASK on $(hostname) — $(date)"
echo "  study  : 2D-ONLY, FROM SCRATCH"
echo "  arm    : $MODE / lambda=$LAMBDA$( [[ "$LAMBDA" == "0" ]] && echo '  (CONTROL)' )"
echo "  cluster: RWTH CLAIX-2023 | partition ${SLURM_JOB_PARTITION:-?} | account $SLURM_JOB_ACCOUNT"
echo "  wall   : ${SLURM_JOB_END_TIME:+ends $(date -d @"$SLURM_JOB_END_TIME" 2>/dev/null)}"
echo "  repo   : $REPO_DIR"
echo "  env    : $CONDA_PREFIX"
echo "  config : $CONFIG"
echo "  data   : $DATA_PATH"
echo "  gpus   : $GPUS_PER_NODE per node x $NNODES node(s) = $WORLD rank(s)  [from $GPU_SRC]"
echo "=================================================================="
python -c "import torch; print(f'  torch  : {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s) visible')"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "       Build the study on the login node first:" >&2
    echo "         SV_BASE_CONFIG=<json> RUNS_ROOT=\$HPCWORK/smilify_runs \\" >&2
    echo "           bash scripts/prior_study/prepare_2d_only_sweep.sh" >&2
    exit 1
fi

# --- where does this arm stand? ----------------------------------------------
# The full schedule does not fit in 24 h, so a chain of dependent jobs each pick
# up where the last one stopped. RESUME_FROM is empty for the first job of a
# chain (train from scratch) and the newest checkpoint for every one after.
# `ls -v` sorts checkpoint_epoch_9 before checkpoint_epoch_10, which a plain
# lexicographic sort does not — getting that wrong would silently rewind the run
# by 90 epochs and it would look like it was working.
CKPT_DIR="$(python -c "import json,sys; print((json.load(open(sys.argv[1])).get('output') or {}).get('checkpoint_dir') or '')" "$CONFIG")"
RESUME_FROM=""
if [[ -n "$CKPT_DIR" && -d "$CKPT_DIR" ]]; then
    RESUME_FROM="$(ls -v "$CKPT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | tail -n1 || true)"
    # best_model.pth is NOT used to resume: it is whichever epoch had the lowest
    # val loss, which can be far behind the newest one, and resuming from it
    # would silently redo epochs that are already done.
fi
if [[ -n "$RESUME_FROM" ]]; then
    echo "  resume : $RESUME_FROM  (continuing this arm; chained job)"
else
    echo "  resume : none — training from randomly initialised weights"
fi
export SMILIFY_2D_RESUME="$RESUME_FROM"

# --- pre-flight: everything that can silently no-op --------------------------
# Cheap, runs before the GPUs are touched, and every check below corresponds to
# a way this job can "succeed" while producing a result that answers no question.
python - "$CONFIG" "$LAMBDA" "$WORLD" "$RESUME_FROM" <<'PY' || exit $?
import json, math, os, pickle, sys
import numpy as np
import torch

cfg_path, expected_lambda, world = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
resume_from = sys.argv[4]
cfg = json.load(open(cfg_path))

DROP_3D = ("keypoint_3d", "global_rot", "joint_rot", "betas", "trans",
           "log_beta_scales", "betas_trans")

# 1. mode -------------------------------------------------------------------
mode = cfg.get("mode")
if mode != "singleview":
    sys.exit(f"ERROR: config mode is {mode!r}; this study is single-view only.")

# 2. from scratch, and the epoch arithmetic ---------------------------------
# training.num_epochs is an ABSOLUTE end epoch and the trainer loops
# range(start_epoch, num_epochs). From scratch start_epoch is 0; on a chained
# resume it is checkpoint_epoch + 1.
tr = cfg.get("training", {})
num_epochs = tr.get("num_epochs")
if not num_epochs:
    sys.exit("ERROR: training.num_epochs is missing or 0.")
if tr.get("resume_checkpoint"):
    sys.exit(f"ERROR: training.resume_checkpoint is {tr['resume_checkpoint']!r}, but this study\n"
             f"       trains FROM SCRATCH. Chained continuations are handled by this script\n"
             f"       (it passes --checkpoint), not by editing the config.")

start = 0
if resume_from:
    if not os.path.isfile(resume_from):
        sys.exit(f"ERROR: resume checkpoint vanished between the scan and now: {resume_from}")
    ck = torch.load(resume_from, map_location="cpu", weights_only=False)
    start = int(ck.get("epoch", -1)) + 1
    sd = ck.get("model_state_dict", ck)
    if "view_embeddings.weight" in sd:
        sys.exit(f"ERROR: {resume_from} is a MULTI-VIEW checkpoint. Wrong checkpoint_dir.")
    print(f"  resume epoch      : {start} (checkpoint says {start - 1})")

print(f"  num_epochs (end)  : {num_epochs}")
if num_epochs <= start:
    # Not an error: this is a chained job that landed after its arm finished.
    # Exiting 0 lets the rest of the chain drain harmlessly instead of each job
    # burning a queue slot to re-discover the same thing.
    print(f"  -> this arm already reached epoch {start} >= num_epochs {num_epochs}. Nothing to do.")
    sys.exit(97)
print(f"  -> will train {num_epochs - start} epoch(s) this job")
print(f"  STUDY INVARIANT   : total schedule is {num_epochs} epochs from scratch — every arm MUST match.")

# 3. NO 3D SUPERVISION — resolved the way the trainer resolves it -----------
# This is the check the whole study rests on, and it cannot be done by reading
# the JSON. get_weights_for_epoch does base.copy() then .update(stage), so a
# curriculum stage re-sets a zeroed weight; and the scale/trans mode override is
# applied AFTER the curriculum (training_config.py:534), so with
# entangled_with_betas it sets betas/log_beta_scales/betas_trans regardless of
# base_weights. Go through load_config, which reproduces both.
sys.path.insert(0, os.getcwd())
from smal_fitter.neuralSMIL.configs import load_config

resolved = load_config(cfg_path)

# 3a. the two resolvers must agree BEFORE trusting anything below.
# load_config's object resolves the scale/trans weights from the config. The
# trainer calls TrainingConfig.get_loss_weights_for_epoch, which uses the
# class-level SCALE_TRANS_BETA_CONFIG — and that only receives the config's
# values if to_legacy_dict emits them. Where it does not, the weights printed
# below for betas / log_beta_scales / betas_trans are not the ones the GPUs use.
try:
    _legacy_stb = (resolved.to_legacy_dict().get("scale_trans_beta") or {}).get("loss_weights")
except Exception as exc:
    print(f"  WARNING: could not build the legacy config dict to verify the scale/trans "
          f"plumbing ({exc}). Verify base_config.to_legacy_dict by hand.")
    _legacy_stb = None
else:
    if _legacy_stb is None or dict(_legacy_stb) != dict(resolved.scale_trans_beta.get_mode_loss_weights()):
        sys.exit("ERROR: BaseTrainingConfig.to_legacy_dict() does not propagate\n"
                 "       scale_trans_beta's per-mode loss weights, so the trainer would resolve\n"
                 "       betas / log_beta_scales / betas_trans from TrainingConfig's class-level\n"
                 "       defaults instead of from this config — 3D parameter supervision would\n"
                 "       stay ON while every report says 2D-only.\n"
                 "       Apply the to_legacy_dict patch before submitting.")
    print("  scale/trans plumbing: the config's weights reach the trainer's resolver  OK")

rows, prev = [], None
for epoch in range(0, int(num_epochs)):
    w = resolved.get_loss_weights_for_epoch(epoch)
    if prev is None or w != prev:
        rows.append((epoch, dict(w)))
        prev = w

live = {}
for epoch, w in rows:
    for key in DROP_3D:
        if float(w.get(key, 0.0)) != 0.0:
            live.setdefault(key, (epoch, w[key]))
if live:
    lines = ["ERROR: 3D SUPERVISION IS STILL ACTIVE. This is not a 2D-only run:"]
    for key, (epoch, val) in sorted(live.items()):
        lines.append(f"         {key} = {val} from epoch {epoch}")
    lines += [
        "       Resolved through load_config, so this is what the GPUs would use.",
        "       Usual causes:",
        "         - a curriculum_stages entry re-setting the key after epoch 0;",
        "         - the scale_trans_beta mode override (betas / log_beta_scales /",
        "           betas_trans), which is applied after the curriculum. That one only",
        "           obeys the config if BaseTrainingConfig.to_legacy_dict() emits the",
        "           per-mode 'loss_weights' — check that patch is in this checkout.",
        "       Rebuild with scripts/prior_study/prepare_2d_only_sweep.sh.",
    ]
    sys.exit("\n".join(lines))
print(f"  3D supervision    : all of {', '.join(DROP_3D)} are 0.0 at every epoch  OK")

# 4. something IS being supervised ------------------------------------------
kp2d = [float(w.get("keypoint_2d", 0.0)) for _, w in rows]
if not any(v > 0 for v in kp2d):
    sys.exit("ERROR: keypoint_2d is 0 at every epoch. With 3D dropped too, this run has no\n"
             "       supervision at all and would optimise the regularizers alone.")
print(f"  keypoint_2d       : {min(kp2d):g} .. {max(kp2d):g} across the schedule")
cam = {k: float(rows[0][1].get(k, 0.0)) for k in ("fov", "cam_rot", "cam_trans")}
print(f"  camera terms      : " + ", ".join(f"{k}={v:g}" for k, v in cam.items()))
if all(v == 0.0 for v in cam.values()):
    print("  WARNING: every camera term is 0 as well. With a free camera and no 3D, the 2D")
    print("           reprojection loss can be reduced by moving the CAMERA rather than posing")
    print("           the animal. Expect the pose to be under-determined.")

# 5. this arm carries THIS task's lambda, unchanged for the whole run -------
lams = sorted({float(w.get("joint_limit_regularization", 0.0)) for _, w in rows})
if len(lams) > 1:
    sys.exit(f"ERROR: joint_limit_regularization changes during the run ({lams}). A curriculum\n"
             f"       stage is overriding the swept weight; the arm would measure two lambdas.")
w = lams[0]
print(f"  joint_limit_regularization: {w}  (array task expects {expected_lambda})")
if not math.isclose(w, expected_lambda, rel_tol=1e-9, abs_tol=1e-12):
    sys.exit(f"ERROR: config weight {w} != the lambda this array task is for ({expected_lambda}).\n"
             f"       {cfg_path} was prepared for a different arm. Two arms would train the same\n"
             f"       point and the table would show duplicate rows under different labels.")
# Unlike run_prior_study_train.sbatch, w == 0 is ALLOWED here: it is the control.
if w == 0.0:
    print("  -> CONTROL arm (no joint-limit prior). This is the baseline the other arms are")
    print("     read against; sv_reference is 3D-supervised and is NOT a baseline for them.")

# 6. the model carries limits for the hinge to act on -----------------------
smal_file = (cfg.get("smal_model", {}) or {}).get("smal_file")
print(f"  smal_file         : {smal_file}")
if not smal_file or not os.path.isfile(smal_file):
    sys.exit(f"ERROR: smal_model.smal_file missing on disk: {smal_file}")
dd = pickle.load(open(smal_file, "rb"), encoding="latin1")
jl = dd.get("joint_limits")
if w > 0:
    if jl is None:
        sys.exit(f"ERROR: {smal_file} has no 'joint_limits' key, but the limit weight is {w}.\n"
                 f"       The hinge would be a guaranteed no-op. Use the authored .pkl.")
    jl = np.asarray(jl, dtype=np.float64)
    n_j = len(dd["J_names"])
    if jl.shape != (n_j, 3, 2):
        sys.exit(f"ERROR: joint_limits shape {jl.shape}, expected {(n_j, 3, 2)}.")
    if not np.all(jl[..., 0] <= jl[..., 1]):
        sys.exit("ERROR: joint_limits has min > max somewhere; LimitPrior will raise.")
    if not np.isfinite(jl).all():
        sys.exit("ERROR: joint_limits contains non-finite values.")
    pi = np.pi
    free = (jl[..., 0] <= -pi + 1e-4) & (jl[..., 1] >= pi - 1e-4)
    # The root is pinned to [0,0] by LimitPrior and dropped by the fitter's [3:]
    # slice, so it contributes nothing to the penalty.
    n_con = int((~free[1:]).sum())
    n_axes = free[1:].size
    if n_con == 0:
        sys.exit("ERROR: every non-root joint/axis is wide open ([-pi,pi]) — the hinge can never fire.")
    coverage = 100.0 * n_con / n_axes
    print(f"  authored limits   : {n_con}/{n_axes} non-root axes constrained ({coverage:.0f}%)")
    # _joint_limit_penalty is a torch.mean over the FULL (N, N_POSE, 3) hinge
    # (smil_image_regressor.py:1445): free axes contribute exact zeros, so the
    # authored fraction scales the effective prior strength.
    eff = w * coverage / 100.0
    print(f"  NOTE: the penalty is a mean over ALL {n_axes} non-root axes, so the effective")
    print(f"        prior strength is w x {coverage / 100.0:.2f} = {eff:.4g}.")
    kp_peak = max(kp2d)
    print(f"  -> effective limit weight is {eff / kp_peak:.3g}x the peak keypoint_2d weight.")
    print(f"     CAVEAT: the earlier lambda sweep ran against a loss dominated by keypoint_3d at")
    print(f"     weight 20. With 3D gone the same lambda is a far larger FRACTION of the total")
    print(f"     loss, so these lambdas are NOT on the same scale as that study's.")

# 7. DDP batch arithmetic ---------------------------------------------------
# batch_size is PER PROCESS under DDP, so N ranks multiply the effective batch
# while the LR curriculum is inherited unchanged. Every arm must use the same
# allocation shape and the same batch_size or they are different optimisation
# problems wearing the same labels.
bs = tr.get("batch_size")
print(f"  batch_size {bs} per rank x {world} rank(s) -> effective {bs * world if bs else '?'}")
print(f"  STUDY INVARIANT   : effective batch {bs * world if bs else '?'} — every arm MUST match.")

# 8. output paths and quota -------------------------------------------------
# Every output path is written by rank 0 mid-epoch, hours after the GPUs were
# allocated, and a quota error arrives as an unhandled OSError inside
# imageio.imsave or torch.save — the run is simply lost. These arms train the
# FULL schedule, so they write more than the +50-epoch sweep did, not less.
out_cfg = cfg.get("output", {}) or {}
save_every = out_cfg.get("save_checkpoint_every") or 1
print(f"  save_checkpoint_every: {save_every} (bounds the epochs lost at each 24 h wall)")
print(f"  generate_visualizations_every: {out_cfg.get('generate_visualizations_every')} "
      f"plot_history_every: {out_cfg.get('plot_history_every')} "
      f"samples: {out_cfg.get('num_visualization_samples')}")
print(f"  checkpoints -> {out_cfg.get('checkpoint_dir')}")

home = os.path.realpath(os.environ.get("HOME", "/nonexistent"))
on_home = []
for key in ("checkpoint_dir", "plots_dir", "visualizations_dir", "train_visualizations_dir"):
    d = out_cfg.get(key)
    if not d:
        continue
    real = os.path.realpath(os.path.abspath(d))
    if real == home or real.startswith(home + os.sep):
        on_home.append((key, real))
if on_home and os.environ.get("ALLOW_HOME_OUTPUT") != "1":
    n_ckpt = max(1, (int(num_epochs) - start) // int(save_every))
    lines = [f"ERROR: {len(on_home)} output dir(s) resolve under $HOME ({home}), which is quota'd:"]
    lines += [f"         {k} -> {p}" for k, p in on_home]
    lines += [
        f"       This arm alone will write ~{n_ckpt} ViT-Large checkpoints there, and the arms",
        f"       share the quota. Rebuild with the run dirs on $HPCWORK:",
        f"         export RUNS_ROOT=\"$HPCWORK/smilify_runs\"",
        f"         SV_BASE_CONFIG=<json> bash scripts/prior_study/prepare_2d_only_sweep.sh",
        f"       (or set ALLOW_HOME_OUTPUT=1 to proceed anyway).",
    ]
    sys.exit("\n".join(lines))

ck_dir = out_cfg.get("checkpoint_dir")
if ck_dir:
    probe = os.path.abspath(ck_dir)
    while probe != os.path.dirname(probe) and not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    try:
        import shutil
        print(f"  free space on {probe}: {shutil.disk_usage(probe).free / 2**30:.0f} GB "
              f"(advisory — statvfs shows the FILESYSTEM, not your quota)")
    except Exception as exc:
        print(f"  (free-space probe skipped: {exc})")

# 9. the learning rate over THIS window -------------------------------------
# The LR is a pure function of the absolute epoch index
# (train_smil_regressor.py:2079) — no early stopping, no plateau scheduler. From
# scratch that means the arm walks the WHOLE curriculum, which is what "the same
# training setup" asks for; on a chained resume the restored optimizer LR is
# overwritten from the curriculum at the resume epoch (line 1269), so a chain
# boundary changes nothing.
lr_rows, prev_lr = [], None
for epoch in range(start, int(num_epochs)):
    lr = resolved.get_learning_rate_for_epoch(epoch)
    if lr != prev_lr:
        lr_rows.append((epoch, lr))
        prev_lr = lr
print(f"  lr over [{start}, {num_epochs}): " + " -> ".join(f"{lr:.3g}@e{e}" for e, lr in lr_rows[:12])
      + (" ..." if len(lr_rows) > 12 else ""))

# 10. degeneracy note -------------------------------------------------------
mesh = cfg.get("mesh_scaling") or {}
if mesh.get("allow_mesh_scaling", True):
    print("  WARNING — SCALE/DEPTH DEGENERACY: mesh_scaling.allow_mesh_scaling is on and there is")
    print("            no 3D supervision. TrainingConfig.MESH_SCALING_CONFIG states the global")
    print("            mesh scale is trained implicitly via the 3D keypoint loss; with that gone,")
    print("            a larger animal further away reprojects to the same pixels. Only the")
    print("            limb_scale / limb_trans regularizers and the joint-limit prior hold it.")
    print("            Kept ON deliberately: the study is the reference setup MINUS 3D.")
PY
rc=$?
if (( rc == 97 )); then
    echo "Nothing to do for this arm — exiting cleanly so the rest of the chain drains."
    exit 0
fi
if (( rc != 0 )); then
    echo "ERROR: pre-flight failed — see above. Not launching." >&2
    exit "$rc"
fi

# --- pre-flight: the study as a whole ----------------------------------------
# Per-arm checks cannot see the sweep invariant. If the sibling configs are on
# disk, check all of them together: an arm that differs from its siblings in
# anything but lambda makes the comparison measure two things at once.
SIBLINGS=()
for f in configs_runs/singleview_2d_lam*.json; do
    [[ -f "$f" ]] && SIBLINGS+=(--config "$f")
done
if (( ${#SIBLINGS[@]} > 2 )); then
    python scripts/prior_study/preflight_2d_only.py "${SIBLINGS[@]}" || {
        echo "ERROR: study-level pre-flight failed — see above. Not launching." >&2
        exit 1
    }
fi

# --- launch ------------------------------------------------------------------
# Both trainers detect RANK/LOCAL_RANK/WORLD_SIZE (is_torchrun_launched) and skip
# their own mp.spawn path, so --num_gpus is intentionally not passed.
#
# Distinct rendezvous port per array task: two tasks can land on the same node,
# and two rendezvous on one port there means the second fails to bind.
# Offset by 100 from run_prior_study_train.sbatch so the two studies can run
# concurrently on the same node without colliding.
RDZV_PORT=$((29500 + TASK))
RDZV_ID="jl2d_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${TASK}"

TORCHRUN="torchrun"
command -v torchrun >/dev/null 2>&1 || TORCHRUN="python -m torch.distributed.run"

LAUNCH_PREFIX=()
if (( NNODES > 1 )); then
    MASTER_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-localhost}" 2>/dev/null | head -n1 || true)"
    MASTER_NODE="${MASTER_NODE:-localhost}"
    RDZV_ENDPOINT="${MASTER_NODE}:${RDZV_PORT}"
    LAUNCH_PREFIX=(srun --nodes="$NNODES" --ntasks-per-node=1
                   --cpus-per-task="${SLURM_CPUS_PER_TASK:-96}" --gpus-per-node="$GPUS_PER_NODE")
    echo "  launch : srun x $NNODES node(s), rendezvous on $RDZV_ENDPOINT"
else
    RDZV_ENDPOINT="localhost:${RDZV_PORT}"
    echo "  launch : torchrun on this node, rendezvous on $RDZV_ENDPOINT"
fi

# --checkpoint is how a chained job continues: it overrides the config's
# resume_checkpoint (which is null by design here), so the config on disk stays
# the from-scratch description of the experiment and the chain leaves no trace
# in it.
RESUME_ARGS=()
[[ -n "$RESUME_FROM" ]] && RESUME_ARGS=(--checkpoint "$RESUME_FROM")

"${LAUNCH_PREFIX[@]}" $TORCHRUN \
    --rdzv-backend=c10d \
    --rdzv-id="$RDZV_ID" \
    --rdzv-endpoint="$RDZV_ENDPOINT" \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    -m "$TRAIN_MODULE" --config "$CONFIG" --data_path "$DATA_PATH" "${RESUME_ARGS[@]}"

echo "Done (2D-only, $MODE, lambda=$LAMBDA, $WORLD rank(s)) — $(date)"
