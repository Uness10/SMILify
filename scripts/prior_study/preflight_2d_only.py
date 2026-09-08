#!/usr/bin/env python
"""Pre-flight for the 2D-only from-scratch study.

``preflight_study.py`` checks a CONTINUATION: it loads the checkpoint each arm
resumes from and looks for architecture drift and split disagreement. These arms
resume from nothing, so that script would reject all of them on its first check.
What has to be verified here instead is different, and it is verified through the
real loader rather than by reading JSON, because the two failures that matter are
both invisible in the JSON:

  1. a curriculum stage re-enabling a supervision term the config zeroed
     (``get_weights_for_epoch`` does ``base.copy()`` then ``.update(stage)``), and
  2. the scale/trans mode override, which is applied AFTER the curriculum
     (``training_config.py:534``) and, with ``entangled_with_betas``, sets
     ``betas``/``log_beta_scales``/``betas_trans`` regardless of ``base_weights``.

Checks
------
  1. every arm is from scratch (``resume_checkpoint`` null)
  2. every dropped 3D term resolves to 0.0 at EVERY epoch in [0, num_epochs)
  3. ``keypoint_2d`` is non-zero somewhere — otherwise the run has no supervision
  4. ``joint_limit_regularization`` is constant across the run and equals the
     lambda this arm claims
  5. the arms differ ONLY in lambda and their output paths: the sweep invariant.
     Anything else that differs (seed, split, batch size, epochs, augmentation,
     backbone) makes the comparison measure two things at once
  6. optional: architecture and split match the reference checkpoint, so the
     2D-only arms can be benchmarked on the same test frames as ``sv_reference``

Usage
-----
    python scripts/prior_study/preflight_2d_only.py \
        --config configs_runs/singleview_2d_lam0.json \
        --config configs_runs/singleview_2d_lam1e-4.json \
        --config configs_runs/singleview_2d_lam1e-1.json \
        --reference singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DROP_3D_WEIGHTS = (
    "keypoint_3d",
    "global_rot",
    "joint_rot",
    "betas",
    "trans",
    "log_beta_scales",
    "betas_trans",
)

# Fields that must be identical across arms for the sweep to measure lambda and
# nothing else. Paths and the swept weight are excluded by construction.
INVARIANT_SECTIONS = (
    "mode",
    "model",
    "dataset",
    "training",
    "optimizer",
    "augmentation",
    "smal_model",
    "mesh_scaling",
    "joint_importance",
    "ignored_joints",
    "ignored_joint_locations",
    "scale_trans_beta",
)

SPLIT_FIELDS = ("seed", "train_ratio", "val_ratio", "test_ratio", "dataset_fraction")


def resolve(config_path: Path):
    from smal_fitter.neuralSMIL.configs import load_config

    return load_config(str(config_path))


def stb_plumbing_ok(resolved) -> bool | None:
    """Do the dataclass resolver and the resolver the TRAINER uses agree?

    ``BaseTrainingConfig.get_loss_weights_for_epoch`` reads
    ``scale_trans_beta.*_loss_weights`` off the config. The trainer instead calls
    ``TrainingConfig.get_loss_weights_for_epoch``, which applies the class-level
    ``SCALE_TRANS_BETA_CONFIG[mode]["loss_weights"]`` — and that dict only
    receives the config's values if ``to_legacy_dict`` emits them. Where it does
    not, every weight this script prints for betas / log_beta_scales /
    betas_trans is a value the GPUs will never see.

    Returns True/False, or None if it could not be determined here (constructing
    the legacy dict can need a backbone probe).
    """
    try:
        legacy = resolved.to_legacy_dict()
    except Exception:
        return None
    emitted = (legacy.get("scale_trans_beta") or {}).get("loss_weights")
    if emitted is None:
        return False
    return dict(emitted) == dict(resolved.scale_trans_beta.get_mode_loss_weights())


def weight_rows(cfg, num_epochs: int):
    rows, prev = [], None
    for epoch in range(0, max(int(num_epochs), 1)):
        w = cfg.get_loss_weights_for_epoch(epoch)
        if prev is None or w != prev:
            rows.append((epoch, dict(w)))
            prev = w
    return rows


def strip_variable(cfg: dict) -> dict:
    """The parts of a config that are ALLOWED to differ between arms."""
    out = json.loads(json.dumps(cfg))  # deep copy through JSON
    out.pop("output", None)
    out.pop("_study", None)
    lc = out.get("loss_curriculum") or {}
    bw = lc.get("base_weights") or {}
    bw.pop("joint_limit_regularization", None)
    return out


def check_one(path: Path, problems: list, warnings: list) -> dict | None:
    print(f"\n=== {path} ===")
    if not path.is_file():
        problems.append(f"{path}: missing")
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))

    # --- 1. from scratch ----------------------------------------------------
    resume = (raw.get("training") or {}).get("resume_checkpoint")
    if resume:
        problems.append(
            f"{path}: training.resume_checkpoint is {resume!r}, but this study trains FROM SCRATCH.\n"
            f"    A resumed arm would start at the checkpoint's epoch and train almost nothing."
        )
    else:
        print("  from scratch: resume_checkpoint is null  OK")

    num_epochs = (raw.get("training") or {}).get("num_epochs")
    if not num_epochs:
        problems.append(f"{path}: training.num_epochs is missing or 0")
        return None
    print(f"  num_epochs: {num_epochs}")

    try:
        cfg = resolve(path)
    except Exception as exc:
        problems.append(f"{path}: does not load through load_config: {exc}")
        return None

    rows = weight_rows(cfg, int(num_epochs))

    # --- 1b. the two resolvers must agree -----------------------------------
    # Everything below reads the dataclass resolver. If to_legacy_dict does not
    # carry the scale/trans loss weights, the trainer's resolver disagrees with
    # it on exactly the three keys this study zeroes, and every "OK" printed
    # after this point would be about a config the GPUs do not run.
    plumbing = stb_plumbing_ok(cfg)
    if plumbing is False:
        problems.append(
            f"{path}: BaseTrainingConfig.to_legacy_dict() does not emit "
            f"scale_trans_beta['loss_weights'].\n"
            f"    The trainer resolves betas / log_beta_scales / betas_trans from the class-level\n"
            f"    TrainingConfig.SCALE_TRANS_BETA_CONFIG instead of from this config, so the zeros\n"
            f"    below are not what would run. Apply the to_legacy_dict patch."
        )
    elif plumbing is None:
        warnings.append(
            f"{path}: could not confirm that to_legacy_dict propagates the scale/trans loss "
            f"weights (building the legacy dict failed here). Verify by hand that base_config.py's "
            f"to_legacy_dict emits 'loss_weights' under scale_trans_beta."
        )
    else:
        print("  scale/trans loss weights reach the trainer's resolver  OK")

    # --- 2. no 3D supervision, at any epoch ---------------------------------
    live = {}
    for epoch, w in rows:
        for key in DROP_3D_WEIGHTS:
            if float(w.get(key, 0.0)) != 0.0:
                live.setdefault(key, []).append((epoch, w[key]))
    if live:
        for key, hits in sorted(live.items()):
            first_ep, first_val = hits[0]
            problems.append(
                f"{path}: 3D SUPERVISION STILL ACTIVE — '{key}' resolves to {first_val} from epoch "
                f"{first_ep} ({len(hits)} weight change(s) affected).\n"
                f"    Resolved through load_config, so this is what the trainer would use. Likely a\n"
                f"    curriculum stage re-setting the key, or the scale_trans_beta mode override."
            )
    else:
        print(f"  3D supervision: all of {', '.join(DROP_3D_WEIGHTS)} are 0.0 at every epoch  OK")

    # --- 3. something is still being supervised -----------------------------
    kp2d = [float(w.get("keypoint_2d", 0.0)) for _, w in rows]
    if not any(v > 0 for v in kp2d):
        problems.append(f"{path}: keypoint_2d is 0 at every epoch — this run has NO supervision at all.")
    else:
        print(f"  keypoint_2d: {min(kp2d):g} .. {max(kp2d):g} across the run  OK")

    # --- 4. lambda constant and as declared ---------------------------------
    lams = sorted({float(w.get("joint_limit_regularization", 0.0)) for _, w in rows})
    if len(lams) > 1:
        problems.append(
            f"{path}: joint_limit_regularization CHANGES during the run ({lams}). "
            f"A curriculum stage is overriding the swept weight."
        )
    lam = lams[0] if lams else 0.0
    declared = (raw.get("_study") or {}).get("joint_limit_regularization")
    if declared is not None and float(declared) != lam:
        problems.append(
            f"{path}: config claims lambda={declared} but resolves to {lam}. "
            f"The written config and the effective weights disagree."
        )
    print(f"  joint_limit_regularization: {lam:g} (constant)  OK")

    if lam > 0:
        smal = (raw.get("smal_model") or {}).get("smal_file")
        if smal and "limits" not in Path(smal).name.lower():
            warnings.append(
                f"{path}: lambda > 0 but smal_file is {Path(smal).name}, which does not look like the "
                f"authored-limits model. The hinge penalty is only meaningful against real limits."
            )

    return {"path": path, "raw": raw, "lam": lam, "num_epochs": int(num_epochs), "rows": rows}


def check_invariants(arms: list[dict], problems: list) -> None:
    """Every arm identical except lambda and paths."""
    print("\n=== sweep invariant ===")
    if len(arms) < 2:
        print("  only one arm — nothing to compare")
        return

    lams = [a["lam"] for a in arms]
    if len(set(lams)) != len(lams):
        problems.append(f"DUPLICATE LAMBDAS across arms: {lams}. Two arms would measure the same point.")
    else:
        print(f"  lambdas distinct: {[f'{v:g}' for v in lams]}  OK")

    ref = arms[0]
    ref_stripped = strip_variable(ref["raw"])
    for arm in arms[1:]:
        stripped = strip_variable(arm["raw"])
        diffs = []
        for section in INVARIANT_SECTIONS:
            if ref_stripped.get(section) != stripped.get(section):
                diffs.append(section)
        # loss curriculum minus the swept weight
        if (ref_stripped.get("loss_curriculum") or {}) != (stripped.get("loss_curriculum") or {}):
            diffs.append("loss_curriculum (beyond joint_limit_regularization)")
        if diffs:
            problems.append(
                f"SWEEP INVARIANT BROKEN: {arm['path'].name} differs from {ref['path'].name} in "
                f"{diffs}.\n"
                f"    The arms would then differ by lambda AND by that, and the comparison measures\n"
                f"    both at once. Regenerate every arm from the same base config in one call."
            )
    if not problems:
        print(f"  all {len(arms)} arms agree on {', '.join(INVARIANT_SECTIONS)}  OK")

    epochs = {a["num_epochs"] for a in arms}
    if len(epochs) > 1:
        problems.append(f"EPOCH COUNTS DIFFER across arms: {sorted(epochs)}. Same schedule or no comparison.")

    dirs = [(a["raw"].get("output") or {}).get("checkpoint_dir") for a in arms]
    if len(set(dirs)) != len(dirs):
        problems.append(f"OUTPUT COLLISION: arms share a checkpoint_dir: {dirs}. They would overwrite each other.")
    else:
        print("  checkpoint dirs distinct  OK")


def check_reference(arms: list[dict], reference: Path, problems: list, warnings: list) -> None:
    """The 2D-only arms should be scorable on the reference's test split."""
    print(f"\n=== vs reference {reference} ===")
    if not reference.is_file():
        warnings.append(f"reference checkpoint not found: {reference} — split agreement unverified")
        return
    import torch

    ref = torch.load(reference, map_location="cpu", weights_only=False)
    ref_cfg = dict(ref.get("config") or {})
    if not ref_cfg:
        warnings.append(
            f"{reference}: no embedded config block, so the test split cannot be verified. The "
            f"2D-only arms may be benchmarked on different frames than sv_reference."
        )
        return
    print(f"  reference epoch: {ref.get('epoch', '?')}")
    for arm in arms:
        flat = {}
        for section in ("training", "dataset", "model", "smal_model"):
            block = arm["raw"].get(section)
            if isinstance(block, dict):
                for k, v in block.items():
                    flat.setdefault(k, v)
        for field in SPLIT_FIELDS:
            rv, cv = ref_cfg.get(field), flat.get(field)
            if rv is None or cv is None:
                continue
            if rv != cv:
                problems.append(
                    f"SPLIT MISMATCH on '{field}': sv_reference says {rv!r}, {arm['path'].name} says "
                    f"{cv!r}. The arms would be scored on different test frames than the reference."
                )
    print("  split fields checked against the reference")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", required=True, help="Prepared 2D-only config (repeatable)")
    p.add_argument("--reference", default=None, help="sv_reference checkpoint, for split agreement")
    args = p.parse_args()

    problems: list[str] = []
    warnings: list[str] = []
    arms = [a for a in (check_one(Path(c), problems, warnings) for c in args.config) if a]

    if arms:
        check_invariants(arms, problems)
        if args.reference:
            check_reference(arms, Path(args.reference), problems, warnings)

    print("\n==================================================================")
    for w in warnings:
        print(f"WARNING: {w}")
    if problems:
        for prob in problems:
            print(f"PROBLEM: {prob}")
        print(f"\n{len(problems)} blocking problem(s). Do not submit.")
        return 1
    print("Pre-flight clean. Safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
