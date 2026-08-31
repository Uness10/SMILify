#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=c23g
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --job-name=jl_train
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# ---------------------------------------------------------------------------
# JOINT-LIMIT PRIOR STUDY — training stage (RWTH CLAIX-2023, c23g)
#
# LAMBDA SWEEP. Each array task fine-tunes ONE checkpoint for
# EXTRA_EPOCHS (default 50) additional epochs at ONE joint-limit weight.
#
#   task  mode        lambda   config
#   ----  ----------  -------  --------------------------------------------
#     0   singleview  1e-4     configs_runs/singleview_lam1e-4.json
#     1   singleview  1e-3     configs_runs/singleview_lam1e-3.json
#     2   singleview  1e-2     configs_runs/singleview_lam1e-2.json
#     3   singleview  1e-1     configs_runs/singleview_lam1e-1.json
#     4   multiview   1e-4     configs_runs/multiview_lam1e-4.json
#     5   multiview   1e-3     configs_runs/multiview_lam1e-3.json
#     6   multiview   1e-2     configs_runs/multiview_lam1e-2.json
#     7   multiview   1e-1     configs_runs/multiview_lam1e-1.json
#
# The DEFAULT array (0-3) is the single-view sweep — the modality where the
# problem is underdetermined and the prior has the most to do. Tasks 4-7 exist
# so the same sweep can be run on multi-view later (`--array=4-7`) without
# editing this file; they are NOT submitted by default.
#
# Each task gets its own node with 4x H100 and runs torchrun over every GPU the
# allocation actually contains. There is still no unconstrained arm: the
# reference is each downloaded checkpoint evaluated as-is (eval sbatch task 0).
#
# ---------------------------------------------------------------------------
# HOW MANY GPUs
# ---------------------------------------------------------------------------
# Nothing here is hard-coded to 4. The script reads SLURM_JOB_NUM_NODES and
# SLURM_GPUS_ON_NODE, so the #SBATCH header is only the default and the sbatch
# command line wins:
#
#   4 arms x 1 node x 4 H100 = 16 GPUs busy   (the default; submit --array=0-3
#                                              with NO % throttle)
#   4 arms x 2 nodes x 4 H100 = 32 GPUs busy  (sbatch --nodes=2 ...)
#
# --nodes=2 halves the wall clock per arm, which is the reason to reach for it:
# 50 epochs may not fit in the 24 h account cap on one node. It is not free —
# batch_size is PER PROCESS under DDP, so doubling the ranks doubles the
# effective batch while the resumed learning-rate curriculum stays put. Pass the
# SAME --nodes to every arm, and pass BATCH_SIZE=<orig/ranks> to
# prepare_lambda_sweep.sh if you want to hold the effective batch at the value
# the reference checkpoint was trained with.
#
# ---------------------------------------------------------------------------
# WALL TIME — read this before submitting
# ---------------------------------------------------------------------------
# The account cap (rwth2151) is MaxWall 1-00:00:00, so 24 h is the most a single
# job can ask for. The previous +10-epoch runs were budgeted at 12 h; if they
# used most of that, 50 epochs will NOT fit and every task will be killed at
# 24 h mid-run.
#
#   grep -h "Epoch .* completed\|epoch time\|Elapsed" logs/jl_train_*_*.out | tail
#
# Multiply the observed per-epoch time by 50 and compare against 24 h BEFORE
# submitting four of these. If it does not fit, the options in order of
# preference are:
#   1. lower EXTRA_EPOCHS (the sweep is about lambda, not about epoch count —
#      as long as every arm gets the SAME number, the comparison holds);
#   2. drop dataset.dataset_fraction for the sweep and restore it for the
#      winning lambda;
#   3. chain a second job with --dependency=afterany and re-run
#      prepare_resume_config.py against the newest periodic checkpoint.
#
# For (3) to lose as little as possible, the sweep prep script sets
# output.save_checkpoint_every = 2, so a timeout costs at most two epochs.
#
# ---------------------------------------------------------------------------
# BEFORE SUBMITTING — prepare all four configs on the LOGIN NODE
# ---------------------------------------------------------------------------
# One command builds the whole sweep:
#
#   cd "$HPCWORK/SMILify"
#   conda activate pytorch3d
#   mkdir -p logs configs_runs
#
#   SV_REF=<path to your singleview .pth> \
#   SV_BASE_CONFIG=<the JSON that produced it> \
#   EXTRA_EPOCHS=50 \
#     bash scripts/prior_study/prepare_lambda_sweep.sh singleview
#
# That is a loop over prepare_resume_config.py, once per lambda:
#
#   python scripts/prior_study/prepare_resume_config.py \
#       --checkpoint  "$SV_REF" \
#       --base-config "$SV_BASE_CONFIG" \
#       --extra-epochs 50 --label lam1e-3 \
#       --joint-limit-weight 1e-3 \
#       --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
#       --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
#       --out configs_runs/singleview_lam1e-3.json
#
# prepare_resume_config.py reads each checkpoint's epoch and turns "50 more
# epochs" into the absolute end epoch the trainers expect (num_epochs is an END
# epoch, not a count — see the pre-flight below).
#
# The script refuses to write if the checkpoint's modality disagrees with the
# base config, or if the model and dataset describe different skeletons.
#
# THEN pre-flight all four configs together — this catches the failures that do
# not crash (architecture drift, mismatched test splits):
#
#   python scripts/prior_study/preflight_study.py \
#       --config configs_runs/singleview_lam1e-4.json --reference "$SV_REF" \
#       --config configs_runs/singleview_lam1e-3.json --reference "$SV_REF" \
#       --config configs_runs/singleview_lam1e-2.json --reference "$SV_REF" \
#       --config configs_runs/singleview_lam1e-1.json --reference "$SV_REF"
#
# Exit code 0 means safe to submit. The same architecture check runs again below
# per array task, but running it on the login node lets you fix a bad
# --base-config in seconds instead of after four queue waits.
#
# ---------------------------------------------------------------------------
# SUBMIT (from the repo root — the #SBATCH --output path resolves at submit
# time, so logs/ must already exist)
# ---------------------------------------------------------------------------
#   jid=$(sbatch --parsable --account=rwth2151 --export=ALL \
#           hpc_files/rwth/run_prior_study_train.sbatch)
#   sbatch --array=1-4 --dependency=afterok:$jid --account=rwth2151 \
#           --export=ALL,SV_REF hpc_files/rwth/run_prior_study_eval.sbatch
#
# Run one lambda only:
#   sbatch --array=1 --account=rwth2151 hpc_files/rwth/run_prior_study_train.sbatch
#
# Run the multi-view sweep later:
#   sbatch --array=4-7 --account=rwth2151 hpc_files/rwth/run_prior_study_train.sbatch
#
# Throttle to two concurrent tasks (each takes a whole 4-GPU node):
#   sbatch --array=0-3%2 --account=rwth2151 hpc_files/rwth/run_prior_study_train.sbatch
#
# Override a config without editing this file:
#   CONFIG=configs_runs/other.json sbatch --array=0 --account=rwth2151 \
#       --export=ALL,CONFIG hpc_files/rwth/run_prior_study_train.sbatch
#
# Monitor:  squeue --me   /   tail -f logs/jl_train_<jobid>_<taskid>.out
# Cancel :  scancel <jobid>
#
# NOTE: intentionally no `set -u`. Conda's activate hooks (libblas_mkl_activate.sh)
# reference unset vars like MKL_INTERFACE_LAYER and abort under set -u.
set -eo pipefail
# ---------------------------------------------------------------------------

# --- account guard -----------------------------------------------------------
# SLURM_JOB_ACCOUNT is set by the scheduler once the job is accepted, so an
# empty value here means neither --account nor a default association applied.
# The `default` association caps you at 100 submitted jobs and 7 days; rwth2151
# gives 1000 submitted / 24 h. A four-task sweep fits either way, but the wall
# clock does not — submit with --account=rwth2151.
if [[ -z "${SLURM_JOB_ACCOUNT:-}" ]]; then
    echo "ERROR: no SLURM account. Submit with:" >&2
    echo "  sbatch --account=rwth2151 $0" >&2
    exit 1
fi

# --- which point of the sweep is this array task? ----------------------------
# Task -> (mode, lambda). Tasks 0-3 are the single-view sweep (the default
# --array), 4-7 the same four lambdas on multi-view.
LAMBDAS=(1e-4 1e-3 1e-2 1e-1)

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK < 0 || TASK > 7 )); then
    echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$TASK (expected 0-7)." >&2
    echo "       0-3 = singleview x {${LAMBDAS[*]}}, 4-7 = multiview x the same." >&2
    exit 1
fi

if (( TASK < 4 )); then
    MODE="singleview"
    TRAIN_MODULE="smal_fitter.neuralSMIL.train_smil_regressor"
    LAMBDA="${LAMBDAS[$TASK]}"
else
    MODE="multiview"
    TRAIN_MODULE="smal_fitter.neuralSMIL.train_multiview_regressor"
    LAMBDA="${LAMBDAS[$((TASK - 4))]}"
fi

# Tag matches prepare_lambda_sweep.sh: "lam1e-3", "lam1e-2", ...
TAG="lam${LAMBDA}"
CONFIG="${CONFIG:-configs_runs/${MODE}_${TAG}.json}"

# Backbone weights must already be cached (hpc_files/download_backbone_weights.py
# on the login node); c23g compute nodes have no outbound network.
export HF_HOME="${HF_HOME:-$HPCWORK/hf_cache}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- how many GPUs did this task actually get? -------------------------------
# Derived from the allocation, never hard-coded. `sbatch --gres=gpu:N --nodes=M`
# on the command line overrides the #SBATCH header above, and the old
# `export CUDA_VISIBLE_DEVICES=0,1,2,3` here silently contradicted that: with
# --gres=gpu:2 it named devices that were not allocated, and with 2 nodes it
# pinned every rank to the first node's ordinals. Slurm already sets
# CUDA_VISIBLE_DEVICES correctly for the allocation, so do not touch it.
NNODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"
if [[ -z "$GPUS_PER_NODE" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
fi
if [[ -z "$GPUS_PER_NODE" || "$GPUS_PER_NODE" -lt 1 ]] 2>/dev/null; then
    # Falling through to 1 would train the arm on a single GPU at 1/4 the speed
    # and 1/4 the effective batch, quietly, and break the sweep invariant against
    # the other three arms. Say so loudly rather than "succeeding".
    echo "WARNING: could not determine the GPU count from the allocation" >&2
    echo "         (SLURM_GPUS_ON_NODE unset and nvidia-smi gave nothing)." >&2
    echo "         Falling back to 1 GPU. If this node really has 4, re-submit with" >&2
    echo "         GPUS_PER_NODE=4 --export=ALL,GPUS_PER_NODE,... — otherwise this arm" >&2
    echo "         will not be comparable with the rest of the sweep." >&2
    GPUS_PER_NODE=1
fi
# Named WORLD (not WORLD_SIZE): is_torchrun_launched() keys on RANK/LOCAL_RANK/
# WORLD_SIZE being present, and setting one of them here would confuse the
# trainer's launch detection. torchrun sets all three for the ranks it spawns.
WORLD=$(( NNODES * GPUS_PER_NODE ))

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
# RWTH: $HPCWORK is the large per-user filesystem; $HOME has a hard quota that a
# conda env plus the 21 GB dataset will blow through.
# --- resolve the conda env ---------------------------------------------------
# ENV_PREFIX is a full env PREFIX (a directory), not a name, and the layout
# differs per install: a conda rooted at $HPCWORK puts envs in `conda/envs/`,
# while pointing `envs_dirs` at $HPCWORK gives `conda_envs/`. Guessing one and
# letting `conda activate` fail burns a queue slot for a one-line error, so
# probe the known layouts and check bin/python actually exists first.
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
    echo "       Then submit with:" >&2
    echo "         ENV_PREFIX=<full prefix path> sbatch --export=ALL,ENV_PREFIX ... $0" >&2
    exit 1
fi

# --- activate conda ----------------------------------------------------------
# Batch jobs don't reliably source ~/.bashrc, so locate conda.sh explicitly.
# Override with CONDA_SH=/path/to/conda.sh if this guess is wrong.
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
echo "  arm    : CONSTRAINED / $MODE / lambda=$LAMBDA"
echo "  cluster: RWTH CLAIX-2023 | partition ${SLURM_JOB_PARTITION:-?} | account $SLURM_JOB_ACCOUNT"
echo "  wall   : ${SLURM_JOB_END_TIME:+ends $(date -d @"$SLURM_JOB_END_TIME" 2>/dev/null)}"
echo "  repo   : $REPO_DIR"
echo "  env    : $CONDA_PREFIX"
echo "  config : $CONFIG"
echo "  gpus   : $GPUS_PER_NODE per node x $NNODES node(s) = $WORLD rank(s)"
echo "=================================================================="
python -c "import torch; print(f'  torch  : {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s) visible')"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "       Build the sweep on the login node first:" >&2
    echo "         SV_REF=<ckpt> SV_BASE_CONFIG=<json> EXTRA_EPOCHS=50 \\" >&2
    echo "           bash scripts/prior_study/prepare_lambda_sweep.sh $MODE" >&2
    exit 1
fi

# --- pre-flight: everything that can silently no-op --------------------------
# Each check below corresponds to a way this job can "succeed" while producing a
# result that does not answer the question. All of them are cheap and run before
# the GPUs are touched.
python - "$CONFIG" "$MODE" "$LAMBDA" "$WORLD" <<'PY' || exit 1
import json, math, os, pickle, sys
import numpy as np
import torch

cfg_path, expected_mode, expected_lambda = sys.argv[1], sys.argv[2], float(sys.argv[3])
world = int(sys.argv[4])
cfg = json.load(open(cfg_path))

# 1. mode -------------------------------------------------------------------
mode = cfg.get("mode")
if mode != expected_mode:
    sys.exit(f"ERROR: config mode is '{mode}' but this array task trains '{expected_mode}'.\n"
             f"       Array tasks 0-3 = singleview, 4-7 = multiview.")

# 2. the num_epochs trap ----------------------------------------------------
# training.num_epochs is an ABSOLUTE end epoch and the trainers resume at
# checkpoint_epoch + 1, looping range(start, num_epochs). If num_epochs <= that,
# the loop body never executes and the job "succeeds" after training nothing.
tr = cfg.get("training", {})
ckpt_path = tr.get("resume_checkpoint")
num_epochs = tr.get("num_epochs")
if not ckpt_path:
    sys.exit("ERROR: training.resume_checkpoint is null — this job is meant to RESUME.")
if not os.path.isfile(ckpt_path):
    sys.exit(f"ERROR: resume_checkpoint does not exist: {ckpt_path}")
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
start = int(ck.get("epoch", -1)) + 1
print(f"  resume checkpoint : {ckpt_path} (epoch {start - 1})")
print(f"  start epoch       : {start}")
print(f"  num_epochs (end)  : {num_epochs}")
if num_epochs is None or num_epochs <= start:
    sys.exit(f"ERROR: num_epochs ({num_epochs}) <= start epoch ({start}) — this would train 0 epochs.\n"
             f"       Re-run prepare_resume_config.py with --extra-epochs N.")
n_train = num_epochs - start
print(f"  -> will train {n_train} epoch(s)")

# 2b. every sweep arm must get the SAME number of epochs --------------------
# A lambda sweep in which the arms trained for different lengths compares
# lambda AND training length at once. Nothing here can enforce it across tasks,
# so record it loudly in the log where compare_arms' provenance can be checked.
print(f"  SWEEP INVARIANT   : this arm trains {n_train} epochs — every other lambda MUST match.")

# 3. the checkpoint really is this modality ---------------------------------
sd = ck.get("model_state_dict", ck)
ckpt_mode = "multiview" if "view_embeddings.weight" in sd else "singleview"
if ckpt_mode != expected_mode:
    sys.exit(f"ERROR: {ckpt_path} is a {ckpt_mode} checkpoint but the config says {expected_mode}.")

# 4. the arm really carries THIS task's lambda ------------------------------
# The sweep's whole failure mode: four configs prepared in a loop, one of them
# built from a stale --joint-limit-weight, and the resulting table shows two
# identical rows at different labels.
w = (cfg.get("loss_curriculum", {}).get("base_weights", {}) or {}).get("joint_limit_regularization", 0.0)
print(f"  joint_limit_regularization: {w}  (array task expects {expected_lambda})")
if not w:
    sys.exit("ERROR: joint_limit_regularization is 0 — this study has only CONSTRAINED arms.\n"
             "       Re-run prepare_resume_config.py with --joint-limit-weight <lambda>.")
if not math.isclose(float(w), expected_lambda, rel_tol=1e-9):
    sys.exit(f"ERROR: config weight {w} != the lambda this array task is for ({expected_lambda}).\n"
             f"       {cfg_path} was prepared for a different sweep point. Two arms of the\n"
             f"       sweep would silently train at the same lambda.")

# 4b. curriculum must not overwrite the swept weight ------------------------
# base_weights is the starting point; loss_curriculum.curriculum_stages replaces
# weights at given epochs. If any stage the continuation will pass through sets
# joint_limit_regularization, the swept lambda is discarded partway through.
stages = cfg.get("loss_curriculum", {}).get("curriculum_stages", {}) or {}
clobber = sorted(
    int(ep) for ep, d in stages.items()
    if isinstance(d, dict) and "joint_limit_regularization" in d and start <= int(ep) < num_epochs
)
if clobber:
    sys.exit(f"ERROR: curriculum_stages at epoch(s) {clobber} override "
             f"joint_limit_regularization inside this run's range [{start}, {num_epochs}).\n"
             f"       The swept lambda would be discarded mid-run. Remove those entries.")

# 5. the model carries limits for the hinge to act on -----------------------
smal_file = (cfg.get("smal_model", {}) or {}).get("smal_file")
print(f"  smal_file         : {smal_file}")
if not smal_file or not os.path.isfile(smal_file):
    sys.exit(f"ERROR: smal_model.smal_file missing on disk: {smal_file}")
dd = pickle.load(open(smal_file, "rb"), encoding="latin1")
jl = dd.get("joint_limits")
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
# Exclude the root: LimitPrior pins J_names[0] to [0,0] and the fitter drops it
# via the [3:] slice, so it contributes nothing to the penalty.
n_con_nonroot = int((~free[1:]).sum())
n_axes_nonroot = free[1:].size
if n_con_nonroot == 0:
    sys.exit("ERROR: every non-root joint/axis is wide open ([-pi,pi] = free) — the hinge can never fire.")
coverage = 100.0 * n_con_nonroot / n_axes_nonroot
print(f"  authored limits   : {n_con_nonroot}/{n_axes_nonroot} non-root axes constrained ({coverage:.0f}%), "
      f"across {int((~free[1:]).any(axis=1).sum())} joints")

# 6. coverage dilution ------------------------------------------------------
# _joint_limit_penalty is a torch.mean over the FULL (N, N_POSE, 3) hinge
# (smil_image_regressor.py:1445). Free axes contribute exact zeros to that mean,
# so the fraction of authored axes scales the effective prior strength.
print(f"  NOTE: the penalty is a mean over ALL {n_axes_nonroot} non-root axes, so the effective")
print(f"        prior strength is w x {coverage / 100.0:.2f}. At w={w} that is an effective {w * coverage / 100.0:.4g}.")

# 6b. is this lambda even on the same scale as the other losses? ------------
# The sweep runs 3-6 orders of magnitude below the w=100 the earlier study used.
# Print the comparison so a log reader can see at a glance whether the hinge can
# possibly win against the keypoint terms, rather than discovering it afterwards.
bw = cfg.get("loss_curriculum", {}).get("base_weights", {}) or {}
peers = {k: bw.get(k) for k in ("keypoint_2d", "keypoint_3d", "joint_angle_regularization") if bw.get(k) is not None}
if peers:
    print(f"  peer weights      : " + ", ".join(f"{k}={v}" for k, v in peers.items()))
    kp = bw.get("keypoint_3d") or bw.get("keypoint_2d")
    if kp:
        print(f"  -> effective limit weight is {w * coverage / 100.0 / float(kp):.3g}x the keypoint weight.")
        print(f"     If the ratio is <<1 the hinge cannot move the pose; that is a RESULT for this")
        print(f"     lambda, not a bug — it is what the sweep is measuring.")

# 7. DDP batch arithmetic ---------------------------------------------------
# batch_size is PER PROCESS under DDP, so N ranks multiply the effective batch
# relative to the run that produced the checkpoint, while inheriting that run's
# learning-rate curriculum unchanged. This scales with the allocation, so an arm
# run on 8 ranks is NOT the same optimisation problem as one run on 4 — every
# arm of the sweep must use the same allocation shape and the same batch_size.
bs = tr.get("batch_size")
print(f"  batch_size {bs} per rank x {world} rank(s) -> effective "
      f"{bs * world if bs else '?'} (batch_size is PER PROCESS under DDP)")
print(f"  SWEEP INVARIANT   : effective batch {bs * world if bs else '?'} — every other lambda MUST match.")
print(f"  save_checkpoint_every: {cfg.get('output', {}).get('save_checkpoint_every')} "
      f"(a wall-clock timeout costs at most this many epochs)")
print(f"  checkpoints -> {cfg.get('output', {}).get('checkpoint_dir')}")
PY

# --- pre-flight: architecture agreement --------------------------------------
# The single check the block above CANNOT do. load_checkpoint drops every tensor
# whose shape disagrees with the freshly-built model and then calls
# load_state_dict(strict=False), returning epoch=0 when anything was dropped
# (train_multiview_regressor.py:2073+2095). So a wrong --base-config silently
# re-initialises layers AND resets the epoch counter — at which point
# "num_epochs: 404" means 404 epochs from scratch, not 50, and the job burns the
# entire wall clock producing a model that answers no question.
#
# The epoch arithmetic above reads the checkpoint's stored epoch, so it is blind
# to this. This is not.
python scripts/prior_study/preflight_study.py --config "$CONFIG" || {
    echo "ERROR: pre-flight failed — see above. Not launching." >&2
    exit 1
}

# --- launch ------------------------------------------------------------------
# Both trainers detect RANK/LOCAL_RANK/WORLD_SIZE (is_torchrun_launched) and skip
# their own mp.spawn path, so --num_gpus is intentionally not passed.
#
# ONE NODE (the default): run torchrun directly. The batch script already runs on
# the allocated node with every allocated GPU visible, and srun+torchrun
# double-launching is a common source of duplicated ranks.
#
# MORE THAN ONE NODE (sbatch --nodes=M): torchrun has to be started once per node,
# which is what srun --ntasks-per-node=1 does, and the rendezvous has to live on a
# real hostname rather than localhost. ddp_main() reads RANK/LOCAL_RANK from the
# environment and calls torch.cuda.set_device(LOCAL_RANK), so the multi-node ranks
# land on the right GPUs (train_smil_regressor.py:194).
#
# Distinct rendezvous port per array task: with --array=0-3%2 two tasks can land
# on the SAME node, and two rendezvous on 29400 there means the second fails to
# bind. --rdzv-id makes the group unique even if a port is reused.
RDZV_PORT=$((29400 + TASK))
RDZV_ID="jl_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${TASK}"

TORCHRUN="torchrun"
command -v torchrun >/dev/null 2>&1 || TORCHRUN="python -m torch.distributed.run"

LAUNCH_PREFIX=()
if (( NNODES > 1 )); then
    MASTER_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    RDZV_ENDPOINT="${MASTER_NODE}:${RDZV_PORT}"
    LAUNCH_PREFIX=(srun --nodes="$NNODES" --ntasks-per-node=1
                   --cpus-per-task="${SLURM_CPUS_PER_TASK:-96}" --gpus-per-node="$GPUS_PER_NODE")
    echo "  launch : srun x $NNODES node(s), rendezvous on $RDZV_ENDPOINT"
else
    RDZV_ENDPOINT="localhost:${RDZV_PORT}"
    echo "  launch : torchrun on this node, rendezvous on $RDZV_ENDPOINT"
fi

"${LAUNCH_PREFIX[@]}" $TORCHRUN \
    --rdzv-backend=c10d \
    --rdzv-id="$RDZV_ID" \
    --rdzv-endpoint="$RDZV_ENDPOINT" \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    -m "$TRAIN_MODULE" --config "$CONFIG"

echo "Done ($MODE, lambda=$LAMBDA, $WORLD rank(s)) — $(date)"
