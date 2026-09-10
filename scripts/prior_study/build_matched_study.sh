#!/usr/bin/env bash
#
# Rebuild the joint-limit prior study against the EPOCH-MATCHED reference.
#
#   RESULTS_ROOT=prior_study_results MATCHED_ROOT=prior_study_results_matched \
#   EXTRA_EPOCHS=50 bash scripts/prior_study/build_matched_study.sh
#
# WHAT THIS IS FOR
# ----------------
# In the study as first run, every `*_reference` row is the downloaded checkpoint
# scored as-is: +0 epochs. Every `*_constrained` row is that same checkpoint plus
# EXTRA_EPOCHS with the hinge on. So each delta is *the prior PLUS the
# continuation*, and a reader is entitled to ask which half did the work.
#
# run_prior_study_train.sbatch tasks 8/9 answer that: the same checkpoint, the
# same EXTRA_EPOCHS, the same LR window / batch size / allocation shape, hinge
# OFF. Eval tasks 10/11 score it into $MATCHED_ROOT/{sv,mv}_reference under the
# label "sv_reference" / "mv_reference".
#
# This script does the cheap part: it links the ALREADY-SCORED constrained arms
# into $MATCHED_ROOT beside that new reference and re-runs compare_arms.py there.
# Nothing is re-trained and nothing is re-scored — the constrained arms are
# byte-identical to the ones in the published tables, which is what makes the two
# sets of numbers directly comparable.
#
#   $RESULTS_ROOT/sweep/sweep.md    reference = +0 epochs   (as published)
#   $MATCHED_ROOT/sweep/sweep.md    reference = +N epochs at lambda=0
#
# Read them side by side. The difference between the two delta columns IS the
# fine-tuning confound, quantified.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RESULTS_ROOT="${RESULTS_ROOT:-prior_study_results}"
MATCHED_ROOT="${MATCHED_ROOT:-${RESULTS_ROOT}_matched}"
EXTRA_EPOCHS="${EXTRA_EPOCHS:-50}"
read -r -a LAMBDA_ARR <<< "${LAMBDAS:-1e-4 1e-3 1e-2 1e-1}"

echo "=================================================================="
echo " Epoch-matched study"
echo "  published root : $RESULTS_ROOT   (reference = +0 epochs)"
echo "  matched root   : $MATCHED_ROOT   (reference = +$EXTRA_EPOCHS epochs, lambda=0)"
echo "  lambdas        : ${LAMBDA_ARR[*]}"
echo "=================================================================="

# --- 1. the matched references must already be scored ------------------------
FOUND=0
for M in sv mv; do
    if [[ -f "$MATCHED_ROOT/${M}_reference/arm.json" ]]; then
        FOUND=$((FOUND + 1))
        EP="$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('checkpoint_epoch','?'))" \
              "$MATCHED_ROOT/${M}_reference/arm.json" 2>/dev/null || echo '?')"
        echo "[matched] ${M}_reference present (checkpoint epoch $EP)"
    else
        echo "[matched] ${M}_reference MISSING — train array task $([[ $M == sv ]] && echo 8 || echo 9)," \
             "then eval array task $([[ $M == sv ]] && echo 10 || echo 11)."
    fi
done
if (( FOUND == 0 )); then
    echo "ERROR: no continued reference scored yet — nothing to build." >&2
    exit 1
fi

# --- 2. link the constrained arms in (no copies: they must not drift) --------
LINKED=0
for L in "${LAMBDA_ARR[@]}"; do
    for A in sv_constrained mv_constrained; do
        SRC="$RESULTS_ROOT/lam${L}/$A"
        [[ -d "$SRC" ]] || continue
        REF="${A%%_constrained}_reference"
        [[ -d "$MATCHED_ROOT/$REF" ]] || continue      # no matched ref for this modality yet
        mkdir -p "$MATCHED_ROOT/lam${L}"
        ln -sfn "$(cd "$SRC" && pwd)" "$MATCHED_ROOT/lam${L}/$A"
        ln -sfn "../$REF"             "$MATCHED_ROOT/lam${L}/$REF"
        LINKED=$((LINKED + 1))
    done
done
echo "[matched] linked $LINKED constrained arm(s)"
if (( LINKED == 0 )); then
    echo "ERROR: found no scored constrained arms under $RESULTS_ROOT/lam*/." >&2
    echo "       Is RESULTS_ROOT right? Expected e.g. $RESULTS_ROOT/lam1e-2/sv_constrained/arm.json" >&2
    exit 1
fi

# --- 3. per-lambda comparison tables against the matched reference -----------
for L in "${LAMBDA_ARR[@]}"; do
    LAM_ROOT="$MATCHED_ROOT/lam${L}"
    [[ -d "$LAM_ROOT" ]] || continue
    echo
    echo "[matched] comparison for lambda=$L ..."
    python scripts/prior_study/compare_arms.py \
        --results-root "$LAM_ROOT" \
        --out "$LAM_ROOT/comparison" \
        --extra-epochs "$EXTRA_EPOCHS" \
        --joint-limit-weight "$L" || \
      echo "[matched] (comparison incomplete for lambda=$L)"
done

# --- 4. the matched sweep table ---------------------------------------------
# Same shape as the one the eval sbatch builds, but lambda=0 is now the
# CONTINUED reference, so every row in it shares one epoch budget.
python - "$MATCHED_ROOT" "$EXTRA_EPOCHS" "${LAMBDA_ARR[@]}" <<'PY'
import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from scripts.prior_study.compare_arms import load_arm, ACC_KEYS  # noqa: E402

root, extra, lambdas = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
rows = []
for label in ("sv_reference", "mv_reference"):
    rec = load_arm(root / label)
    if rec:
        rec["lambda"] = 0.0
        rows.append(rec)
for lam in lambdas:
    for arm in ("sv_constrained", "mv_constrained"):
        rec = load_arm(root / f"lam{lam}" / arm)
        if rec:
            rec["lambda"] = float(lam)
            rows.append(rec)

if not rows:
    print("[matched] nothing scored yet")
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

lines = [f"# Joint-limit prior — lambda sweep (EPOCH-MATCHED, +{extra} epochs everywhere)", "",
         f"`lambda = 0` here is **not** the downloaded checkpoint. It is that checkpoint",
         f"continued for the same +{extra} epochs as every other row, with the joint-limit",
         "hinge switched off. Every row therefore shares one training budget, one LR",
         "window and one allocation shape, and the only thing that varies down the table",
         "is lambda — which is what the deltas were always meant to measure.", "",
         "Compare against `../sweep/sweep.md` (reference = +0 epochs): the gap between",
         "the two `lambda = 0` rows is the fine-tuning effect on its own, and the gap",
         "between the two delta columns is how much of the published improvement was",
         "the extra epochs rather than the prior.", "",
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
lines += ["", "Lower is better for violations, overshoot and MPJPE; higher for PCK.", ""]
(out / "sweep.md").write_text("\n".join(lines))
print(f"[matched] wrote {out/'sweep.csv'} and {out/'sweep.md'} ({len(rows)} arm(s))")
print("\n".join(lines))
PY

# --- 5. the confound table: +0 vs +N at lambda=0 ------------------------------
# One small table that is the actual answer to "was it the prior or the epochs?".
python - "$RESULTS_ROOT" "$MATCHED_ROOT" "$EXTRA_EPOCHS" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from scripts.prior_study.compare_arms import load_arm, ACC_KEYS  # noqa: E402

pub, matched, extra = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
KEYS = [("violating_axes", "violating axes", "d"),
        ("mean_viol_rate", "mean viol. rate %", ".2f"),
        ("mean_overshoot_deg", "mean overshoot deg", ".2f"),
        ("max_overshoot_deg", "max overshoot deg", ".2f")] + [(k, k, ".4f") for k in ACC_KEYS]

lines = [f"# Is it the prior, or is it the extra {extra} epochs?", "",
         "Both columns are the SAME checkpoint. The only difference is that the right",
         f"column trained {extra} more epochs with `joint_limit_regularization = 0`.",
         "Whatever moves here moved without any prior at all, and must be subtracted",
         "from the published reference-vs-lambda deltas before they are read as the",
         "effect of the prior.", ""]
any_rows = False
for m, label in (("singleview", "sv_reference"), ("multiview", "mv_reference")):
    a, b = load_arm(pub / label), load_arm(matched / label)
    if not a or not b:
        continue
    any_rows = True
    lines += [f"## {m}", "",
              f"| metric | reference (+0 ep) | reference (+{extra} ep, lambda=0) | delta |",
              "|---|---|---|---|"]
    for key, name, spec in KEYS:
        x, y = a.get(key), b.get(key)
        if x is None and y is None:
            continue
        d = "—" if (x is None or y is None) else format(y - x, spec if spec != "d" else "+d")
        fx = "—" if x is None else format(x, spec)
        fy = "—" if y is None else format(y, spec)
        lines.append(f"| {name} | {fx} | {fy} | {d} |")
    lines.append("")

if not any_rows:
    print("[matched] no reference pair to compare yet")
    raise SystemExit(0)

out = matched / "confound"
out.mkdir(parents=True, exist_ok=True)
(out / "fine_tuning_confound.md").write_text("\n".join(lines))
print(f"[matched] wrote {out/'fine_tuning_confound.md'}")
print("\n".join(lines))
PY

echo
echo "=================================================================="
echo " Read, in this order:"
echo "   $MATCHED_ROOT/confound/fine_tuning_confound.md   <- the epochs on their own"
echo "   $MATCHED_ROOT/sweep/sweep.md                     <- the sweep, all rows +$EXTRA_EPOCHS"
echo "   $RESULTS_ROOT/sweep/sweep.md                     <- the published table, for contrast"
echo "=================================================================="
