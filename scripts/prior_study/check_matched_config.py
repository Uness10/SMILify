#!/usr/bin/env python3
"""Assert that the continued-reference config is the sweep config minus the prior.

    python scripts/prior_study/check_matched_config.py \
        --refcont configs_runs/singleview_refcont.json \
        --arm     configs_runs/singleview_lam1e-2.json

The epoch-matched baseline is only worth anything if it is identical to the
constrained arms in EVERY respect except ``joint_limit_regularization``. One
stale ``--extra-epochs``, one forgotten ``BATCH_SIZE``, one different
``lr_schedule`` and the arm no longer isolates the prior — it just adds a second
confound in the opposite direction, which is worse than having no baseline at
all because the table now looks rigorous.

This is a pure JSON diff, so it runs on a login node in under a second. Exit 0
means the only differences are the ones on the allow-list below.

Allow-listed differences (everything else is an ERROR):
  * loss_curriculum.base_weights.joint_limit_regularization   0 vs lambda
  * output.* / checkpoint_dir / run_dir / label paths         per-arm by design
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keys whose values are allowed to differ, matched on the flattened dotted path.
ALLOWED_EXACT = {
    "loss_curriculum.base_weights.joint_limit_regularization",
}
# Any flattened path containing one of these substrings is a per-arm output path.
ALLOWED_SUBSTRINGS = (
    "checkpoint_dir",
    "output_dir",
    "log_dir",
    "run_dir",
    "visualization_dir",
    "plot_dir",
    "results_dir",
    "experiment_name",
    "run_name",
    "label",
)


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, sort_keys=True)
    else:
        out[prefix] = obj
    return out


def allowed(path: str) -> bool:
    return path in ALLOWED_EXACT or any(s in path for s in ALLOWED_SUBSTRINGS)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refcont", required=True, type=Path,
                   help="the lambda=0 continued-reference config")
    p.add_argument("--arm", required=True, type=Path,
                   help="any ONE of the constrained sweep configs for the same mode")
    args = p.parse_args()

    for f in (args.refcont, args.arm):
        if not f.is_file():
            print(f"ERROR: no such config: {f}", file=sys.stderr)
            return 2

    ref, arm = json.load(open(args.refcont)), json.load(open(args.arm))
    fr, fa = flatten(ref), flatten(arm)

    print(f"refcont : {args.refcont}")
    print(f"arm     : {args.arm}")
    print()

    problems, benign = [], []
    for key in sorted(set(fr) | set(fa)):
        a, b = fr.get(key, "<absent>"), fa.get(key, "<absent>")
        if a == b:
            continue
        (benign if allowed(key) else problems).append((key, a, b))

    # The two things that must hold positively, not just "not differ".
    w_ref = (ref.get("loss_curriculum", {}).get("base_weights", {}) or {}).get(
        "joint_limit_regularization", 0.0)
    w_arm = (arm.get("loss_curriculum", {}).get("base_weights", {}) or {}).get(
        "joint_limit_regularization", 0.0)
    if float(w_ref) != 0.0:
        problems.append(("loss_curriculum.base_weights.joint_limit_regularization",
                         w_ref, "must be exactly 0 in the refcont arm"))
    if float(w_arm) == 0.0:
        problems.append(("--arm", w_arm,
                         "the comparison arm has weight 0 — that is not a constrained arm"))

    n_ref = ref.get("training", {}).get("num_epochs")
    n_arm = arm.get("training", {}).get("num_epochs")
    if n_ref != n_arm:
        problems.append(("training.num_epochs", n_ref,
                         f"{n_arm} — the arms end at different epochs, so the baseline "
                         f"is not epoch-matched"))

    if benign:
        print(f"{len(benign)} allow-listed difference(s):")
        for key, a, b in benign:
            print(f"  ok   {key}: {a!r} (refcont) vs {b!r} (arm)")
        print()

    if problems:
        print(f"{len(problems)} DISQUALIFYING difference(s):", file=sys.stderr)
        for key, a, b in problems:
            print(f"  FAIL {key}: {a!r} (refcont) vs {b!r} (arm)", file=sys.stderr)
        print(file=sys.stderr)
        print("The continued reference must differ from the sweep arms ONLY in the",
              file=sys.stderr)
        print("joint-limit weight. Rebuild it with the SAME EXTRA_EPOCHS / BATCH_SIZE /",
              file=sys.stderr)
        print("LR_FLAT / SAVE_EVERY you used for the sweep:", file=sys.stderr)
        mode = ref.get("mode", "singleview")
        print(f"    LAMBDAS=0 EXTRA_EPOCHS=<same> [BATCH_SIZE=<same>] \\", file=sys.stderr)
        print(f"        bash scripts/prior_study/prepare_lambda_sweep.sh {mode}",
              file=sys.stderr)
        return 1

    print(f"OK — epoch-matched. num_epochs={n_ref} in both; "
          f"joint_limit_regularization {w_ref} vs {w_arm}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
