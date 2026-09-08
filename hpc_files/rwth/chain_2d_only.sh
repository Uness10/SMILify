#!/usr/bin/env bash
#
# Submit the 2D-only study as a CHAIN of dependent array jobs.
#
#   CHAIN=8 bash hpc_files/rwth/chain_2d_only.sh
#
# Why a chain. These arms train the reference schedule FROM SCRATCH — hundreds
# of epochs — and the account's MaxWall is 24 h. One submission cannot finish an
# arm. run_2d_only_train.sbatch resumes from the newest checkpoint in its own
# arm's checkpoint_dir, so N identical jobs submitted back-to-back with
# `--dependency=afterany` walk the schedule forward one wall clock at a time,
# and the first job that finds the arm already at num_epochs exits 0 without
# allocating anything. Over-provisioning the chain is therefore cheap; running
# out of links means coming back and submitting more.
#
# afterany, not afterok: a job killed at the 24 h wall exits non-zero, and that
# is the NORMAL case here — it left checkpoints and the next link should pick
# them up. afterok would stall the chain at the first timeout.
#
# The chain is per-STUDY, not per-arm: each link is an array over all three arms,
# so the arms stay in lockstep and one slow arm does not starve the others of
# queue position. An arm that finishes early no-ops through the remaining links.
#
# Env:
#   CHAIN     number of links to submit          (default: 6)
#   ACCT      SLURM account                      (default: rwth2151)
#   ARRAY     array spec                         (default: 0-2)
#   SBATCH_EXTRA
#             extra args passed to every sbatch   (e.g. "--nodes=2")
#   ENV_PREFIX / RUNS_ROOT
#             forwarded to the jobs; export them before running this.
#
# Estimating CHAIN: time one epoch first (submit a single link, read the log),
# then  CHAIN = ceil(num_epochs * epoch_hours / 23).  Leave an hour of slack —
# the 24 h is wall clock, and it includes queueing inside the job, dataset
# construction and the first checkpoint write.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CHAIN="${CHAIN:-6}"
ACCT="${ACCT:-rwth2151}"
ARRAY="${ARRAY:-0-2}"
SBATCH="hpc_files/rwth/run_2d_only_train.sbatch"

[[ -f "$SBATCH" ]] || { echo "ERROR: $SBATCH not found (run from the repo root)" >&2; exit 1; }
if ! (( CHAIN >= 1 )); then
    echo "ERROR: CHAIN must be >= 1 (got '$CHAIN')" >&2
    exit 1
fi

mkdir -p logs

echo "Submitting a $CHAIN-link chain over array $ARRAY, account $ACCT"
echo "  each link resumes its arm from the newest checkpoint;"
echo "  a link that finds its arm finished exits 0 without training."
echo

PREV=""
LINKS=()
for (( i = 0; i < CHAIN; i++ )); do
    DEP=()
    [[ -n "$PREV" ]] && DEP=(--dependency="afterany:$PREV")
    # ${DEP[@]+...} keeps `set -u` from tripping over the empty array on the
    # first link (bash treats an empty array as unset). SBATCH_EXTRA is
    # deliberately unquoted so "--nodes=2 --time=12:00:00" splits into words.
    # shellcheck disable=SC2086
    jid=$(sbatch --parsable --array="$ARRAY" --account="$ACCT" \
            ${DEP[@]+"${DEP[@]}"} ${SBATCH_EXTRA:-} \
            --export=ALL,ENV_PREFIX,RUNS_ROOT,LAMBDAS \
            "$SBATCH")
    LINKS+=("$jid")
    if [[ -n "$PREV" ]]; then
        echo "  link $((i + 1))/$CHAIN: $jid  (after any of $PREV)"
    else
        echo "  link $((i + 1))/$CHAIN: $jid  (head of chain)"
    fi
    PREV="$jid"
done

echo
echo "Chain: ${LINKS[*]}"
echo
echo "Watch it:      squeue --me"
echo "First link:    tail -f logs/jl2d_train_${LINKS[0]}_0.out"
echo "Cancel all:    scancel ${LINKS[*]}"
echo
echo "When the arms are done, score them:"
echo "  RUNS_ROOT=\$RUNS_ROOT sbatch --array=0-2 --dependency=afterany:${LINKS[-1]} \\"
echo "      --account=$ACCT --export=ALL,ENV_PREFIX,RUNS_ROOT \\"
echo "      hpc_files/rwth/run_prior_study_eval.sbatch"
echo "  (check that script's task->arm mapping first; it was written for the"
echo "   continuation study and may need ARM/CONFIG overrides for these run dirs.)"
