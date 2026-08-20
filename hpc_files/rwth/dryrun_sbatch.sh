#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# dryrun_sbatch.sh — run an .sbatch script's pre-flight on the LOGIN NODE
# ---------------------------------------------------------------------------
# Executes the batch script exactly as the scheduler would, but with the heavy
# launch commands (torchrun, srun) replaced by no-op stubs. Everything up to the
# launch runs for real: conda activation, the config checks, the checkpoint
# epoch arithmetic, preflight_study.py. Nothing touches a GPU.
#
# Run it from the repo root:
#
#   bash hpc_files/rwth/dryrun_sbatch.sh hpc_files/rwth/run_prior_study_train.sbatch
#
# By default every task in the script's `#SBATCH --array` is checked in turn.
# Pass explicit task IDs to narrow it:
#
#   bash hpc_files/rwth/dryrun_sbatch.sh hpc_files/rwth/run_prior_study_train.sbatch 1
#
# Env passthrough works the same as with sbatch --export, since this is just a
# shell running the script:
#
#   SV_CONFIG=configs_runs/other.json bash hpc_files/rwth/dryrun_sbatch.sh ...
#
# Exit code 0 means every requested task reached the launch step cleanly.
#
# What this does NOT catch: anything that only exists on a compute node —
# GPU availability, CUDA kernel compilation, the c23g network blackout, and
# per-node filesystem visibility. Use `salloc` for those.
# ---------------------------------------------------------------------------
set -uo pipefail

SCRIPT="${1:-}"
if [[ -z "$SCRIPT" ]]; then
    echo "usage: bash $0 <script.sbatch> [array_task_id ...]" >&2
    exit 2
fi
shift
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: no such script: $SCRIPT" >&2
    exit 2
fi

# --- 1. syntax check before executing anything -------------------------------
if ! bash -n "$SCRIPT"; then
    echo "ERROR: $SCRIPT has a bash syntax error (see above)." >&2
    exit 2
fi

# --- 2. which array tasks? ---------------------------------------------------
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    # Expand the script's own `#SBATCH --array=` spec: "0-3", "0,2", "0-3:2".
    spec="$(grep -m1 -oP '^#SBATCH\s+--array=\K[^ %]+' "$SCRIPT" || true)"
    if [[ -n "$spec" ]]; then
        IFS=',' read -ra parts <<< "$spec"
        for p in "${parts[@]}"; do
            step="${p##*:}"; [[ "$step" == "$p" ]] && step=1
            p="${p%%:*}"
            if [[ "$p" == *-* ]]; then
                for ((t = ${p%%-*}; t <= ${p##*-}; t += step)); do TASKS+=("$t"); done
            else
                TASKS+=("$p")
            fi
        done
    fi
    [[ ${#TASKS[@]} -eq 0 ]] && TASKS=(0)
fi

# --- 3. stub out the launch commands -----------------------------------------
STUB_DIR="$(mktemp -d)"
trap 'rm -rf "$STUB_DIR"' EXIT
for cmd in torchrun srun; do
    cat > "$STUB_DIR/$cmd" <<STUB
#!/usr/bin/env bash
echo
echo "  ================================================================"
echo "  DRY RUN — reached the launch step. Would have executed:"
echo "      $cmd \$*"
echo "  ================================================================"
exit 0
STUB
    chmod +x "$STUB_DIR/$cmd"
done
export PATH="$STUB_DIR:$PATH"

# --- 4. fake the scheduler's environment -------------------------------------
# The script's account guard reads SLURM_JOB_ACCOUNT, and REPO_DIR falls back to
# SLURM_SUBMIT_DIR. Real sbatch sets both; here we do.
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-dryrun}"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-login-node-dryrun}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-000000}"
export SLURM_ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-000000}"

mkdir -p logs

failed=()
for task in "${TASKS[@]}"; do
    echo
    echo "##################################################################"
    echo "# DRY RUN  $SCRIPT  (SLURM_ARRAY_TASK_ID=$task)"
    echo "##################################################################"
    SLURM_ARRAY_TASK_ID="$task" bash "$SCRIPT"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "--- task $task FAILED (exit $rc) ---" >&2
        failed+=("$task")
    else
        echo "--- task $task OK ---"
    fi
done

echo
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All dry-run tasks passed (${TASKS[*]}). Safe to sbatch."
    exit 0
fi
echo "FAILED tasks: ${failed[*]} — fix these before submitting." >&2
exit 1
