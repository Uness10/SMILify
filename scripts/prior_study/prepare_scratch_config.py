#!/usr/bin/env python
"""Derive a *from-scratch* training config from the JSON that produced a reference run.

Why this exists (and why it is not ``prepare_resume_config.py``)
----------------------------------------------------------------
``prepare_resume_config.py`` builds a CONTINUATION: it reads the epoch out of a
checkpoint, sets ``training.resume_checkpoint`` and turns "N more epochs" into an
absolute end epoch. The 2D-only study is the opposite experiment — every arm
starts from randomly initialised weights and trains the SAME schedule the
reference config declares — so almost none of that logic applies, and the pieces
that do (isolating output dirs, setting the joint-limit weight, stripping
curriculum overrides of the swept key) are re-implemented here rather than bent
into a script whose whole contract is "resume".

What it does
------------
  * copies the reference config verbatim, then
  * MATERIALISES the loss curriculum. This is the important part. In the loader,
    ``loss_curriculum.base_weights`` and ``loss_curriculum.curriculum_stages``
    are plain ``dict`` fields, so a JSON block REPLACES the dataclass default
    wholesale (configs/config_utils.py:_deep_merge_into_dataclass) — but a
    config that omits the block inherits ``LossCurriculumConfig``'s defaults,
    including a curriculum that drives ``keypoint_3d`` up to 20. Editing a
    config that never mentioned ``keypoint_3d`` would therefore change nothing.
    So both blocks are written out in full, resolved from the defaults when the
    base config is silent, and only then edited.
  * zeroes every 3D-supervision weight in ``base_weights`` and DELETES those
    keys from every curriculum stage, so no stage can resurrect them mid-run,
  * zeroes the ``scale_trans_beta`` per-mode overrides for the dropped keys.
    ``TrainingConfig.get_loss_weights_for_epoch`` applies those AFTER the
    curriculum (training_config.py:534), so with
    ``mode = entangled_with_betas`` the hardcoded ``betas``/``log_beta_scales``/
    ``betas_trans`` weights win over anything in ``base_weights``. Zeroing them
    here only takes effect together with the ``to_legacy_dict`` change that
    actually propagates these values — see ``--require-stb-plumbing``.
  * sets ``loss_curriculum.base_weights.joint_limit_regularization`` to the
    swept lambda and strips that key from every stage,
  * sets ``training.resume_checkpoint = null`` and leaves ``training.num_epochs``
    exactly as the reference config declares it,
  * isolates ``output.*`` per label,
  * validates through the real loader and prints the EFFECTIVE weights, resolved
    the way the trainer resolves them, at every epoch where they change.

Usage
-----
    python scripts/prior_study/prepare_scratch_config.py \
        --base-config singleview_SMILySTICKS_3D_ViT.json \
        --label 2d_lam1e-4 \
        --joint-limit-weight 1e-4 \
        --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \
        --data-path SMILySTICKS_centred_reprojected_FIXED.h5 \
        --run-dir "$RUNS_ROOT/singleview_2d_lam1e-4" \
        --out configs_runs/singleview_2d_lam1e-4.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------- the sets

# Supervision that comes from 3D ground truth. `keypoint_3d` is the explicit 3D
# keypoint loss; the rest are the ground-truth SMAL parameters, which encode the
# same 3D information in parameter space — dropping only `keypoint_3d` would
# leave the model supervised on the exact joint rotations and translation and
# the run would not be "2D only" in any meaningful sense.
DROP_3D_WEIGHTS = (
    "keypoint_3d",
    "global_rot",
    "joint_rot",
    "betas",
    "trans",
    "log_beta_scales",
    "betas_trans",
)

# Kept on purpose. The camera terms are what anchor the projection: without them
# a single-view 2D-only run has no handle on where the camera is, and the 2D
# reprojection loss is satisfied by moving the camera instead of posing the
# animal. `silhouette` is 0 in every shipped config but is listed so the printed
# table is complete.
KEEP_WEIGHTS = ("keypoint_2d", "fov", "cam_rot", "cam_trans", "silhouette")

# The subset of DROP_3D_WEIGHTS that the scale/trans mode override can resurrect.
STB_OVERRIDABLE = ("betas", "log_beta_scales", "betas_trans")

MODE_TO_STB_FIELD = {
    "ignore": "ignore_loss_weights",
    "separate": "separate_loss_weights",
    "entangled_with_betas": "entangled_loss_weights",
}


# --------------------------------------------------------------------- helpers


def load_base_config_module():
    """Load ``configs/base_config.py`` directly, without importing the package.

    The package pulls in torch and the SMAL model at import time; this file is
    pure dataclasses. Same trick prepare_lambda_sweep.sh uses to read the LR
    defaults on a login node.
    """
    path = REPO_ROOT / "smal_fitter" / "neuralSMIL" / "configs" / "base_config.py"
    if not path.is_file():
        raise SystemExit(f"ERROR: cannot find {path}")
    spec = importlib.util.spec_from_file_location("_base_config_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def materialise_loss_curriculum(cfg: dict, bc) -> tuple[dict, dict, list[str]]:
    """Return (base_weights, curriculum_stages) fully resolved, and what was filled in.

    A missing block means "use the dataclass defaults", not "no curriculum".
    Writing the resolved values back makes the config self-describing, which is
    what lets the preflight assert on it and what lets a reader six months from
    now see the experiment without cross-referencing base_config.py.
    """
    defaults = bc.LossCurriculumConfig()
    lc = cfg.get("loss_curriculum") or {}
    filled = []

    base_weights = lc.get("base_weights")
    if not base_weights:
        base_weights = dict(defaults.base_weights)
        filled.append("base_weights")
    else:
        # A partial block still replaces the default wholesale in the loader, so
        # what is written here must be exactly what the loader would end up with.
        base_weights = dict(base_weights)

    stages = lc.get("curriculum_stages")
    if not stages:
        stages = {str(k): dict(v) for k, v in defaults.curriculum_stages.items()}
        filled.append("curriculum_stages")
    else:
        stages = {str(k): dict(v) for k, v in stages.items()}

    return base_weights, stages, filled


def set_output_dirs(cfg: dict, run_dir: str) -> None:
    """Point every output directory inside *run_dir* so arms never collide."""
    out = cfg.setdefault("output", {})
    out["checkpoint_dir"] = f"{run_dir}/checkpoints"
    out["plots_dir"] = f"{run_dir}/plots"
    out["visualizations_dir"] = f"{run_dir}/visualizations"
    out["train_visualizations_dir"] = f"{run_dir}/visualizations_train"
    out.setdefault("save_checkpoint_every", 2)
    out.setdefault("generate_visualizations_every", 10)
    out.setdefault("plot_history_every", 10)
    out.setdefault("num_visualization_samples", 5)


def stb_plumbing_present(resolved=None) -> bool:
    """Does ``to_legacy_dict`` propagate the per-mode scale/trans loss weights?

    This matters because there are TWO resolvers and only one of them runs on the
    GPU. ``BaseTrainingConfig.get_loss_weights_for_epoch`` (the dataclass) reads
    ``scale_trans_beta.*_loss_weights`` straight off the config, so it reports
    whatever the JSON says. The trainer calls
    ``TrainingConfig.get_loss_weights_for_epoch`` (the legacy class), which
    applies ``SCALE_TRANS_BETA_CONFIG[mode]["loss_weights"]`` — a class-level
    dict that only ever receives the config's values if ``to_legacy_dict``
    emits them.

    So without the patch the two disagree, the dataclass view looks clean, and
    the run trains with ``betas=0.0005`` while every report says 2D-only. Prefer
    asking a real config object what it emits; fall back to reading the source
    if constructing one is not possible here.
    """
    if resolved is not None:
        try:
            legacy = resolved.to_legacy_dict()
            emitted = (legacy.get("scale_trans_beta") or {}).get("loss_weights")
            if emitted is None:
                return False
            return dict(emitted) == dict(resolved.scale_trans_beta.get_mode_loss_weights())
        except Exception:
            pass  # e.g. to_legacy_dict needs a backbone probe; fall through
    path = REPO_ROOT / "smal_fitter" / "neuralSMIL" / "configs" / "base_config.py"
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return False
    tail = src.split("def to_legacy_dict", 1)
    if len(tail) < 2:
        return False
    return '"loss_weights": self.scale_trans_beta.get_mode_loss_weights()' in tail[1]


def effective_weights_table(config_path: Path, num_epochs: int) -> list[tuple[int, dict]]:
    """Resolve the weights the way the TRAINER resolves them, epoch by epoch.

    Goes through ``load_config`` and ``BaseTrainingConfig.get_loss_weights_for_epoch``,
    which applies curriculum stages and then the scale/trans mode override — the
    exact order of ``TrainingConfig.get_loss_weights_for_epoch``. Anything that
    only reads the JSON would miss the mode override, which is the one that bites.
    """
    from smal_fitter.neuralSMIL.configs import load_config

    cfg = load_config(str(config_path))
    rows, prev = [], None
    for epoch in range(0, max(int(num_epochs), 1)):
        w = cfg.get_loss_weights_for_epoch(epoch)
        if prev is None or w != prev:
            rows.append((epoch, dict(w)))
            prev = w
    return rows


# ------------------------------------------------------------------------ main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-config", required=True, help="The JSON that produced the reference run")
    p.add_argument("--label", required=True, help="Run label; drives the output dir names")
    p.add_argument(
        "--joint-limit-weight",
        type=float,
        required=True,
        help="loss_curriculum.base_weights.joint_limit_regularization. 0.0 is the CONTROL arm.",
    )
    p.add_argument("--run-dir", default=None, help="Output root (default: runs/singleview_<label>)")
    p.add_argument("--out", required=True, help="Where to write the derived config JSON")
    p.add_argument("--data-path", default=None, help="Override dataset.data_path")
    p.add_argument("--smal-file", default=None, help="Override smal_model.smal_file")
    p.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    p.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers")
    p.add_argument("--num-epochs", type=int, default=None, help="Override training.num_epochs (default: keep the base config's)")
    p.add_argument(
        "--keypoint-2d-scale",
        type=float,
        default=1.0,
        help="Multiply every keypoint_2d weight by this. Default 1.0 = untouched. "
        "See the warning this script prints about the total-loss scale collapsing "
        "once keypoint_3d (weight up to 20 in the reference curriculum) is removed.",
    )
    p.add_argument(
        "--keep-3d",
        action="append",
        default=[],
        choices=list(DROP_3D_WEIGHTS),
        help="Keep this 3D supervision term at its reference weight (repeatable). "
        "Escape hatch for a narrower ablation; the default drops all of them.",
    )
    p.add_argument("--no-validate", action="store_true", help="Skip the load_config round-trip check")
    p.add_argument(
        "--allow-missing-stb-plumbing",
        action="store_true",
        help="Write the config even though to_legacy_dict does not propagate the per-mode "
        "scale/trans loss weights. The run would then keep the hardcoded betas / "
        "log_beta_scales / betas_trans supervision. Only for inspecting the output.",
    )
    args = p.parse_args()

    base_path = Path(args.base_config)
    if not base_path.is_file():
        raise SystemExit(f"ERROR: --base-config not found: {base_path}")

    bc = load_base_config_module()
    cfg = copy.deepcopy(json.loads(base_path.read_text(encoding="utf-8")))

    mode = cfg.get("mode", "singleview")
    if mode != "singleview":
        print(f"[prepare] NOTE: base config declares mode={mode!r}; this study is single-view.")

    label = args.label
    run_dir = args.run_dir or f"runs/{mode}_{label}"

    print("==================================================================")
    print(f" 2D-only from-scratch config | label={label}")
    print(f"  base config : {base_path}")
    print(f"  lambda      : {args.joint_limit_weight:g}")
    print(f"  run dir     : {run_dir}")
    print("==================================================================")

    # ---- from scratch ------------------------------------------------------
    training = cfg.setdefault("training", {})
    prev_resume = training.get("resume_checkpoint")
    training["resume_checkpoint"] = None
    if prev_resume:
        print(f"[prepare] training.resume_checkpoint: {prev_resume} -> null (FROM SCRATCH)")
    else:
        print("[prepare] training.resume_checkpoint: null (FROM SCRATCH)")

    num_epochs = args.num_epochs if args.num_epochs is not None else training.get("num_epochs")
    if num_epochs is None:
        raise SystemExit(
            "ERROR: the base config declares no training.num_epochs and none was passed.\n"
            "       This study trains the reference schedule from scratch, so the epoch count\n"
            "       has to come from the reference config. Pass --num-epochs to override."
        )
    training["num_epochs"] = int(num_epochs)
    print(f"[prepare] training.num_epochs: {num_epochs} (absolute end epoch, from epoch 0)")

    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.num_workers is not None:
        training["num_workers"] = args.num_workers

    # ---- data / model overrides -------------------------------------------
    if args.data_path:
        cfg.setdefault("dataset", {})["data_path"] = args.data_path
    if args.smal_file:
        cfg.setdefault("smal_model", {})["smal_file"] = args.smal_file

    # ---- loss curriculum: materialise, then edit ---------------------------
    base_weights, stages, filled = materialise_loss_curriculum(cfg, bc)
    if filled:
        print(
            f"[prepare] loss_curriculum: {', '.join(filled)} was absent from the base config, so the\n"
            f"          run would have inherited LossCurriculumConfig's defaults. Materialised them\n"
            f"          into this config before editing — otherwise zeroing a key the config never\n"
            f"          mentioned would have changed nothing."
        )

    dropped = [k for k in DROP_3D_WEIGHTS if k not in args.keep_3d]
    kept_3d = [k for k in DROP_3D_WEIGHTS if k in args.keep_3d]
    if kept_3d:
        print(f"[prepare] --keep-3d: KEEPING {kept_3d} at their reference weights")

    for key in dropped:
        base_weights[key] = 0.0
    # Deleting from the stages rather than zeroing them keeps the base weight
    # authoritative: get_weights_for_epoch does base.copy() then .update(stage),
    # so a stage that still carries the key would overwrite the 0 the moment its
    # epoch threshold is crossed.
    stage_hits = {}
    for ep, overrides in stages.items():
        for key in dropped:
            if key in overrides:
                stage_hits.setdefault(key, []).append(int(ep))
                del overrides[key]
    for key, eps in sorted(stage_hits.items()):
        print(f"[prepare] stripped '{key}' from curriculum stage(s) {sorted(eps)}")

    # keypoint_2d rescale (opt-in)
    if args.keypoint_2d_scale != 1.0:
        s = float(args.keypoint_2d_scale)
        if "keypoint_2d" in base_weights:
            base_weights["keypoint_2d"] = float(base_weights["keypoint_2d"]) * s
        for overrides in stages.values():
            if "keypoint_2d" in overrides:
                overrides["keypoint_2d"] = float(overrides["keypoint_2d"]) * s
        print(f"[prepare] keypoint_2d weights multiplied by {s:g} (--keypoint-2d-scale)")

    # the swept knob
    base_weights["joint_limit_regularization"] = float(args.joint_limit_weight)
    jl_stripped = []
    for ep, overrides in stages.items():
        if "joint_limit_regularization" in overrides:
            del overrides["joint_limit_regularization"]
            jl_stripped.append(int(ep))
    if jl_stripped:
        print(
            f"[prepare] stripped 'joint_limit_regularization' from curriculum stage(s) "
            f"{sorted(jl_stripped)} — lambda={args.joint_limit_weight:g} now holds for the whole run"
        )

    cfg["loss_curriculum"] = {"base_weights": base_weights, "curriculum_stages": stages}

    # ---- the scale/trans override ------------------------------------------
    # This block is applied AFTER the curriculum, so it is the last word on
    # betas / log_beta_scales / betas_trans.
    stb = cfg.setdefault("scale_trans_beta", {})
    stb_mode = stb.get("mode") or bc.ScaleTransBetaConfig().mode
    stb["mode"] = stb_mode
    stb_field = MODE_TO_STB_FIELD.get(stb_mode)
    if stb_field is None:
        raise SystemExit(f"ERROR: unknown scale_trans_beta.mode {stb_mode!r}")
    defaults_stb = getattr(bc.ScaleTransBetaConfig(), stb_field)
    overrides_stb = dict(stb.get(stb_field) or defaults_stb)
    touched = []
    for key in STB_OVERRIDABLE:
        if key in dropped and overrides_stb.get(key, 0.0) != 0.0:
            overrides_stb[key] = 0.0
            touched.append(key)
    stb[stb_field] = overrides_stb
    if touched:
        print(
            f"[prepare] scale_trans_beta.{stb_field}: zeroed {touched}.\n"
            f"          TrainingConfig.get_loss_weights_for_epoch applies this block AFTER the\n"
            f"          curriculum (training_config.py:534), so without this the mode's hardcoded\n"
            f"          weights would have overridden the zeros in base_weights."
        )

    if touched and not stb_plumbing_present():
        msg = (
            "ERROR: this checkout's BaseTrainingConfig.to_legacy_dict() does not propagate\n"
            "       scale_trans_beta's per-mode loss weights (it emits only 'mode'), so the\n"
            "       zeros written above would be DISCARDED at train time and the run would keep\n"
            f"       supervising {touched} from ground truth while claiming to be 2D-only.\n"
            "       Apply the accompanying to_legacy_dict patch, or re-run with\n"
            "       --allow-missing-stb-plumbing if you only want to look at the JSON."
        )
        if args.allow_missing_stb_plumbing:
            print(msg.replace("ERROR:", "WARNING:"))
        else:
            raise SystemExit(msg)

    # ---- output dirs -------------------------------------------------------
    set_output_dirs(cfg, run_dir)

    # ---- provenance --------------------------------------------------------
    cfg["_study"] = {
        "name": "2d_only_from_scratch",
        "label": label,
        "base_config": str(base_path).replace("\\", "/"),
        "joint_limit_regularization": float(args.joint_limit_weight),
        "dropped_3d_supervision": dropped,
        "kept_3d_supervision": kept_3d,
        "keypoint_2d_scale": float(args.keypoint_2d_scale),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[prepare] wrote {out_path}")

    # ---- validate + show the weights the trainer will actually use ---------
    if args.no_validate:
        print("[prepare] skipping load_config validation (--no-validate)")
        return 0

    try:
        rows = effective_weights_table(out_path, int(num_epochs))
    except Exception as exc:
        print(f"[prepare] ERROR: the written config does not load: {exc}")
        return 1

    print("\n[prepare] EFFECTIVE loss weights (curriculum + scale/trans override, as the trainer resolves them)")
    cols = list(dropped) + list(KEEP_WEIGHTS) + ["joint_limit_regularization"]
    cols = [c for c in cols if any(c in w for _, w in rows)]
    print("  epoch | " + " | ".join(cols))
    for epoch, w in rows:
        print(f"  {epoch:>5} | " + " | ".join(f"{w.get(c, 0.0):g}" for c in cols))

    bad = []
    for epoch, w in rows:
        for key in dropped:
            if float(w.get(key, 0.0)) != 0.0:
                bad.append((epoch, key, w[key]))
    if bad:
        print("\n[prepare] FAILED: 3D supervision is still active:")
        for epoch, key, val in bad:
            print(f"          epoch {epoch}: {key} = {val}")
        return 1
    print(f"\n[prepare] OK: every dropped term is 0.0 across all {num_epochs} epochs")

    if not any(float(w.get("keypoint_2d", 0.0)) > 0 for _, w in rows):
        print("[prepare] FAILED: keypoint_2d is 0 everywhere — this run has no supervision at all.")
        return 1

    # ---- the degeneracy warning -------------------------------------------
    mesh = cfg.get("mesh_scaling") or {}
    allow_scaling = mesh.get("allow_mesh_scaling", bc.MeshScalingConfig().allow_mesh_scaling)
    if allow_scaling and "trans" in dropped and "keypoint_3d" in dropped:
        print(
            "\n[prepare] WARNING — SCALE/DEPTH DEGENERACY.\n"
            "          mesh_scaling.allow_mesh_scaling is true and there is no 3D supervision left.\n"
            "          TrainingConfig.MESH_SCALING_CONFIG says in as many words that the global mesh\n"
            "          scale 'is trained implicitly via 3D keypoint losses'. With keypoint_3d and the\n"
            "          GT translation both at weight 0, a uniformly larger animal further from the\n"
            "          camera reprojects to the same pixels, so nothing in the loss distinguishes\n"
            "          scale from depth. What is left holding it are the limb_scale / limb_trans\n"
            "          regularizers and the joint-limit prior.\n"
            "          This is kept ON because the study is specified as the reference setup minus\n"
            "          3D supervision. Set mesh_scaling.allow_mesh_scaling=false in the base config,\n"
            "          for EVERY arm, if you decide to remove the degeneracy instead."
        )

    if float(args.joint_limit_weight) > 0:
        print(
            "\n[prepare] NOTE — lambda is not comparable across studies. The sweep that produced\n"
            "          lam1e-4 .. lam1e-1 ran against a total loss dominated by keypoint_3d at\n"
            "          weight 20. With that term gone the same lambda is a much larger FRACTION of\n"
            "          the loss, so 1e-1 here is a stronger prior than 1e-1 there."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
