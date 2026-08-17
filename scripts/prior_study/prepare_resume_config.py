#!/usr/bin/env python
"""Derive a *continuation* training config from a downloaded checkpoint folder.

Why this exists
---------------
``training.num_epochs`` in this codebase is an **absolute end epoch**, not a
count of additional epochs: on resume the trainer sets

    start_epoch = checkpoint["epoch"] + 1          (train_smil_regressor.py:1185)

and then loops ``for epoch in range(start_epoch, num_epochs)``. So a checkpoint
saved at epoch 240 with ``num_epochs: 250`` in its config trains 9 more epochs;
with ``num_epochs: 10`` it trains **zero** and exits silently. This script reads
the actual epoch out of the checkpoint and writes ``num_epochs = epoch + 1 + N``
so "N more epochs" means N more epochs.

It also:
  * points ``training.resume_checkpoint`` at the checkpoint,
  * sets ``loss_curriculum.base_weights.joint_limit_regularization`` (0.0 for the
    unconstrained arm, >0 for the constrained arm),
  * isolates ``output.*`` dirs per label so the two arms never overwrite each
    other,
  * optionally repoints ``dataset.data_path`` / ``smal_model.smal_file``,
  * validates the result through the real loader (``load_config``) unless
    ``--no-validate``.

Usage
-----
    python scripts/prior_study/prepare_resume_config.py \
        --checkpoint singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth \
        --base-config singleview_SMILySTICKS_3D_ViT_checkpoints/config.json \
        --extra-epochs 10 \
        --label unconstrained \
        --out configs_runs/singleview_unconstrained.json

If ``--base-config`` is omitted it defaults to ``config.json`` next to the
checkpoint, and failing that to the ``config`` block embedded in the checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------- helpers


def read_checkpoint_epoch(ckpt_path: Path) -> tuple[int, dict | None]:
    """Return ``(epoch, embedded_config_or_None)`` from a checkpoint.

    Loaded on CPU with ``weights_only=False`` because the checkpoint carries a
    plain-dict ``config`` block alongside the tensors.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"ERROR: {ckpt_path} is not a checkpoint dict (got {type(ckpt).__name__}).")
    if "epoch" not in ckpt:
        raise SystemExit(
            f"ERROR: {ckpt_path} has no 'epoch' key, so the absolute end epoch cannot be computed.\n"
            f"       Keys present: {sorted(k for k in ckpt if not k.endswith('state_dict'))}\n"
            f"       Pass --assume-epoch N to override."
        )
    return int(ckpt["epoch"]), ckpt.get("config")


def embedded_config_to_json_schema(embedded: dict) -> dict:
    """Best-effort conversion of a checkpoint's flat ``config`` block to the
    JSON config schema.

    Only used as a fallback when no ``config.json`` sits next to the checkpoint.
    The trainer's own ``save_config_json`` output is always preferred because it
    round-trips exactly; this path fills in the sections the loader requires and
    warns about anything it had to guess.
    """
    model_config = dict(embedded.get("model_config") or {})
    training_params = dict(embedded.get("training_params") or {})

    out = {
        "mode": "singleview",
        "smal_model": {
            "smal_file": embedded.get("smal_file"),
            "shape_family": embedded.get("shape_family", -1),
        },
        "dataset": {
            "data_path": embedded.get("data_path") or training_params.get("data_path"),
            "train_ratio": embedded.get("train_ratio", training_params.get("train_ratio", 0.85)),
            "val_ratio": embedded.get("val_ratio", training_params.get("val_ratio", 0.05)),
            "test_ratio": embedded.get("test_ratio", training_params.get("test_ratio", 0.10)),
            "from_multiview": bool(embedded.get("from_multiview", False)),
            "frame_convention": embedded.get("frame_convention", "model_centric"),
            "expand_all_views": bool(embedded.get("expand_all_views", False)),
            "use_ue_scaling": bool(embedded.get("use_ue_scaling", False)),
        },
        "model": model_config,
        "training": training_params,
    }
    if embedded.get("scale_trans_mode"):
        out["scale_trans_beta"] = {"mode": embedded["scale_trans_mode"]}
    if "allow_mesh_scaling" in embedded:
        out["mesh_scaling"] = {
            "allow_mesh_scaling": bool(embedded["allow_mesh_scaling"]),
            "init_mesh_scale": float(embedded.get("init_mesh_scale", 1.0)),
        }
    return out


def discover_base_config(ckpt_path: Path) -> Path | None:
    """Find the JSON config that produced *ckpt_path*.

    The trainer writes ``config.json`` into ``output.checkpoint_dir``, but that
    file is often not what gets shipped alongside a checkpoint. The naming
    convention is the reliable hook: configs set
    ``checkpoint_dir: "<name>_checkpoints"``, so a checkpoint living in
    ``singleview_SMILySTICKS_3D_ViT_checkpoints/`` came from
    ``singleview_SMILySTICKS_3D_ViT.json``.

    Search order:
      1. ``config.json`` inside the checkpoint dir (the trainer's own dump)
      2. any single ``*.json`` inside the checkpoint dir
      3. ``<checkpoint_dir_without_suffix>.json`` in the repo root, the CWD and
         ``smal_fitter/neuralSMIL/configs/examples/``
    """
    ckpt_dir = ckpt_path.parent

    direct = ckpt_dir / "config.json"
    if direct.is_file():
        return direct

    jsons = sorted(ckpt_dir.glob("*.json"))
    if len(jsons) == 1:
        return jsons[0]

    stem = ckpt_dir.name
    for suffix in ("_checkpoints", "_ckpt", "_checkpoint"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for parent in (Path.cwd(), REPO_ROOT, REPO_ROOT / "smal_fitter/neuralSMIL/configs/examples"):
        cand = parent / f"{stem}.json"
        if cand.is_file():
            return cand
    return None


def cross_check_model_and_data(cfg: dict) -> list[str]:
    """Return a list of blocking problems with the model/dataset pairing.

    Job 15521637 trained the 55-joint stick model against a mouse HDF5 without
    crashing, so a silent mismatch here is a real failure mode, not a
    hypothetical. Compare the dataset's stored joint count against the model's
    and refuse to proceed when they disagree.
    """
    problems: list[str] = []
    data_path = (cfg.get("dataset") or {}).get("data_path")
    smal_file = (cfg.get("smal_model") or {}).get("smal_file")

    if not data_path:
        problems.append("dataset.data_path is empty")
    elif not (Path(data_path).is_file() or (REPO_ROOT / data_path).is_file()):
        problems.append(
            f"dataset.data_path does not exist: {data_path}\n"
            f"         (pass --data-path to point at the dataset you actually mean)"
        )

    if not smal_file:
        problems.append("smal_model.smal_file is empty")
        return problems

    pkl = Path(smal_file) if Path(smal_file).is_file() else REPO_ROOT / smal_file
    if not pkl.is_file():
        problems.append(f"smal_model.smal_file does not exist: {smal_file}")
        return problems

    h5 = None
    if data_path:
        cand = Path(data_path) if Path(data_path).is_file() else REPO_ROOT / data_path
        h5 = cand if cand.is_file() else None
    if h5 is None or h5.suffix not in (".h5", ".hdf5"):
        return problems  # nothing to cross-check against

    try:
        import pickle

        import h5py

        with open(pkl, "rb") as f:
            dd = pickle.load(f, encoding="latin1")
        names = list(dd.get("J_names", []))
        model_joints = len(names) if names else int(dd["kintree_table"].shape[1])

        with h5py.File(h5, "r") as f:
            if "metadata" not in f:
                return problems
            data_joints = f["metadata"].attrs.get("n_joints")

        if data_joints is not None and int(data_joints) != int(model_joints):
            problems.append(
                f"MODEL/DATA MISMATCH: {Path(smal_file).name} has {model_joints} joints but "
                f"{Path(data_path).name} stores n_joints={int(data_joints)}.\n"
                f"         These describe different skeletons; training will not crash but the\n"
                f"         keypoint losses would be meaningless."
            )
        else:
            print(f"[prepare] model/data cross-check: {model_joints} joints on both sides — OK")
    except Exception as exc:  # advisory only; never block on a probe failure
        print(f"[prepare] (model/data cross-check skipped: {exc})")

    return problems


def set_output_dirs(cfg: dict, run_dir: str) -> None:
    out = cfg.setdefault("output", {})
    out["checkpoint_dir"] = f"{run_dir}/checkpoints"
    out["plots_dir"] = f"{run_dir}/plots"
    out["visualizations_dir"] = f"{run_dir}/visualizations"
    out["train_visualizations_dir"] = f"{run_dir}/visualizations_train"
    out.setdefault("save_checkpoint_every", 1)
    out.setdefault("generate_visualizations_every", 1)
    out.setdefault("plot_history_every", 1)
    out.setdefault("num_visualization_samples", 5)


# ------------------------------------------------------------------------ main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Checkpoint to resume from (.pth)")
    p.add_argument(
        "--base-config",
        default=None,
        help="JSON config to start from (default: config.json next to the checkpoint, "
        "else the config block embedded in the checkpoint)",
    )
    p.add_argument("--extra-epochs", type=int, default=10, help="Number of ADDITIONAL epochs to train (default: 10)")
    p.add_argument(
        "--assume-epoch",
        type=int,
        default=None,
        help="Skip loading the checkpoint and assume it was saved at this epoch (for dry runs)",
    )
    p.add_argument("--label", default="unconstrained", help="Run label; drives the output dir names")
    p.add_argument("--run-dir", default=None, help="Output root (default: runs/singleview_<label>)")
    p.add_argument("--out", default=None, help="Where to write the derived config JSON")
    p.add_argument(
        "--joint-limit-weight",
        type=float,
        default=0.0,
        help="loss_curriculum.base_weights.joint_limit_regularization. "
        "0.0 = UNCONSTRAINED arm (default); >0 = constrained arm.",
    )
    p.add_argument("--data-path", default=None, help="Override dataset.data_path")
    p.add_argument("--smal-file", default=None, help="Override smal_model.smal_file")
    p.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    p.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers")
    p.add_argument("--no-validate", action="store_true", help="Skip the load_config round-trip check")
    p.add_argument(
        "--allow-embedded-config",
        action="store_true",
        help="Permit reconstructing the config from the checkpoint's embedded block when no JSON "
        "config can be found. Carries stale defaults — see the error message. Off by default.",
    )
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if args.assume_epoch is None and not ckpt_path.is_file():
        raise SystemExit(f"ERROR: checkpoint not found: {ckpt_path}")

    # ---- base config -------------------------------------------------------
    embedded = None
    if args.base_config:
        base_path = Path(args.base_config)
        if not base_path.is_file():
            raise SystemExit(f"ERROR: --base-config not found: {base_path}")
    else:
        base_path = discover_base_config(ckpt_path)
        if base_path is not None:
            print(f"[prepare] auto-discovered base config: {base_path}")

    # ---- checkpoint epoch --------------------------------------------------
    if args.assume_epoch is not None:
        ckpt_epoch = int(args.assume_epoch)
        print(f"[prepare] assuming checkpoint epoch {ckpt_epoch} (--assume-epoch, checkpoint not read)")
    else:
        ckpt_epoch, embedded = read_checkpoint_epoch(ckpt_path)
        print(f"[prepare] checkpoint epoch: {ckpt_epoch}")

    if base_path is not None:
        with open(base_path) as f:
            cfg = json.load(f)
        print(f"[prepare] base config: {base_path}")
    elif embedded and args.allow_embedded_config:
        cfg = embedded_config_to_json_schema(embedded)
        print(
            "[prepare] WARNING: reconstructing the config from the checkpoint's embedded 'config' block\n"
            "          (--allow-embedded-config). That block is written from the RUNTIME TrainingConfig,\n"
            "          so its data_path and model dims can be stale defaults rather than what the run\n"
            "          actually used. Loss curriculum, augmentation, joint-importance and ignored-joint\n"
            "          settings are absent entirely. Verify every value below before submitting."
        )
    else:
        raise SystemExit(
            "ERROR: could not find the JSON config that produced this checkpoint.\n"
            f"       Searched: {ckpt_path.parent}/config.json, {ckpt_path.parent}/*.json,\n"
            f"                 ./<name>.json for checkpoint dir '{ckpt_path.parent.name}'\n"
            "       Pass it explicitly with --base-config <file>.json\n"
            "\n"
            "       The checkpoint DOES embed a config block, but it is written from the runtime\n"
            "       TrainingConfig and carries stale defaults: on job 15521637 it produced a MOUSE\n"
            "       data_path for a STICK model, which trained without error and would have\n"
            "       silently invalidated the comparison. Use --allow-embedded-config only if you\n"
            "       have no config file and intend to check every field by hand."
        )

    cfg = copy.deepcopy(cfg)
    cfg["mode"] = "singleview"

    # ---- resume + absolute end epoch ---------------------------------------
    start_epoch = ckpt_epoch + 1  # trainer resumes here
    end_epoch = start_epoch + int(args.extra_epochs)

    training = cfg.setdefault("training", {})
    training["resume_checkpoint"] = str(ckpt_path).replace("\\", "/")
    prev_num_epochs = training.get("num_epochs")
    training["num_epochs"] = end_epoch
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.num_workers is not None:
        training["num_workers"] = args.num_workers

    print(
        f"[prepare] resume from epoch {start_epoch}, train {args.extra_epochs} epoch(s) "
        f"-> training.num_epochs = {end_epoch} (was {prev_num_epochs})"
    )
    if prev_num_epochs is not None and prev_num_epochs <= start_epoch:
        print(
            f"[prepare] note: the base config's num_epochs ({prev_num_epochs}) was already <= the resume "
            f"epoch ({start_epoch}); running it unmodified would have trained 0 epochs."
        )

    # ---- constrained / unconstrained switch --------------------------------
    lc = cfg.setdefault("loss_curriculum", {})
    bw = lc.setdefault("base_weights", {})
    bw["joint_limit_regularization"] = float(args.joint_limit_weight)
    # A curriculum stage firing after the resume epoch would silently override
    # the base weight, so strip the key from any stage inside the run window.
    stages = lc.get("curriculum_stages") or {}
    for stage_key, overrides in list(stages.items()):
        try:
            stage_epoch = int(stage_key)
        except (TypeError, ValueError):
            continue
        if start_epoch <= stage_epoch < end_epoch and isinstance(overrides, dict):
            if overrides.pop("joint_limit_regularization", None) is not None:
                print(
                    f"[prepare] removed joint_limit_regularization from curriculum stage {stage_epoch} "
                    f"(inside the run window; would have overridden the base weight)"
                )
    arm = "UNCONSTRAINED" if args.joint_limit_weight == 0.0 else f"CONSTRAINED (w={args.joint_limit_weight})"
    print(f"[prepare] arm: {arm}")

    # ---- overrides + isolated outputs --------------------------------------
    if args.data_path:
        cfg.setdefault("dataset", {})["data_path"] = args.data_path
    if args.smal_file:
        cfg.setdefault("smal_model", {})["smal_file"] = args.smal_file

    run_dir = args.run_dir or f"runs/singleview_{args.label}"
    set_output_dirs(cfg, run_dir)
    print(f"[prepare] outputs -> {run_dir}/")

    # ---- model/dataset pairing (blocking) ----------------------------------
    print(f"[prepare] dataset   : {(cfg.get('dataset') or {}).get('data_path')}")
    print(f"[prepare] smal file : {(cfg.get('smal_model') or {}).get('smal_file')}")
    problems = cross_check_model_and_data(cfg)
    if problems:
        print("\n[prepare] BLOCKING problems with the model/dataset pairing:", file=sys.stderr)
        for problem in problems:
            print(f"       - {problem}", file=sys.stderr)
        print("\n       Nothing written. Fix the config or pass --data-path/--smal-file.", file=sys.stderr)
        return 4

    # ---- constrained-arm pre-flight ----------------------------------------
    if args.joint_limit_weight > 0.0:
        smal_file = (cfg.get("smal_model") or {}).get("smal_file")
        if smal_file and Path(REPO_ROOT / smal_file).is_file():
            try:
                import pickle

                with open(REPO_ROOT / smal_file, "rb") as f:
                    dd = pickle.load(f, encoding="latin1")
                if "joint_limits" not in dd:
                    print(
                        f"[prepare] ERROR: joint-limit weight > 0 but {smal_file} has NO 'joint_limits' key.\n"
                        f"          The constrained arm needs authored limits; the trainer will abort.\n"
                        f"          Author them in Blender (docs/joint_limits_user_guide.md) or point\n"
                        f"          --smal-file at a model that has them "
                        f"(e.g. 3D_model_prep/OmniAnt_25PCs_joint_limited.pkl).",
                        file=sys.stderr,
                    )
                    return 2
            except Exception as exc:  # pragma: no cover - advisory only
                print(f"[prepare] (could not probe {smal_file} for joint_limits: {exc})")

    # ---- write -------------------------------------------------------------
    out_path = Path(args.out or f"configs_runs/singleview_{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[prepare] wrote {out_path}")

    # ---- validate through the real loader ----------------------------------
    if not args.no_validate:
        try:
            os.chdir(REPO_ROOT)
            from smal_fitter.neuralSMIL.configs.config_utils import load_config

            load_config(config_file=str(out_path), expected_mode="singleview")
            print("[prepare] load_config round-trip: OK")
        except Exception as exc:
            print(f"[prepare] ERROR: derived config failed to load: {exc}", file=sys.stderr)
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
