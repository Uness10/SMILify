#!/usr/bin/env python3
"""Pre-flight the prior study: catch the failures that do NOT crash.

Run this on the login node after preparing the configs and before submitting.
Everything here is cheap (loads checkpoints on CPU, reads JSON) and targets ways
the study can complete successfully while producing a number that does not
answer the question.

Checks
------
1. **Architecture agreement.** ``load_checkpoint`` calls
   ``load_state_dict(..., strict=False)`` after dropping every tensor whose shape
   disagrees with the freshly-built model
   (``train_multiview_regressor.py:2073``). If the base config declares a
   different ``hidden_dim`` / ``max_views`` / backbone than the checkpoint was
   trained with, the mismatched layers are silently **re-initialised** — and the
   function then returns ``epoch = 0``. Two consequences, neither of which
   raises:

     * the "continuation" is partly a fresh model;
     * ``start_epoch`` becomes 0, so ``num_epochs = 251`` trains **251** epochs
       rather than 10, and the job runs until the wall clock kills it.

   The training sbatch computes its epoch arithmetic from the checkpoint's
   stored epoch, so it cannot see this coming. This check can.

2. **Split agreement.** Both the benchmark and the exporter derive the test split
   from ``seed`` + ``train_ratio`` + ``val_ratio``. The reference arm reads those
   from the *downloaded checkpoint's* embedded config; the constrained arm reads
   them from the config we prepared. If the two disagree, the arms are scored on
   **different frames** and the comparison is meaningless — while every
   individual number still looks plausible.

3. **Same dataset and same model file** across the arms being compared.

Usage
-----
    python scripts/prior_study/preflight_study.py \
        --reference "$SV_REF" --config configs_runs/singleview_constrained.json \
        --reference "$MV_REF" --config configs_runs/multiview_constrained.json

Pass ``--config`` alone to check architecture only (no reference to compare
splits against). Exit code 0 = safe to submit, 1 = blocking problem found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Architecture fields that must match between the checkpoint and the config the
# trainer will build from. The single-view trainer builds purely from config; the
# multi-view trainer infers max_views (and hidden_dim, via the benchmark path)
# from the state dict, so a mismatch there is less fatal — but still reported.
ARCH_FIELDS = [
    "backbone_name",
    "hidden_dim",
    "head_type",
    "rotation_representation",
    "scale_trans_mode",
    "cross_attention_layers",
    "cross_attention_heads",
    "max_views",
    "freeze_backbone",
]

SPLIT_FIELDS = ["seed", "train_ratio", "val_ratio"]


def load_ckpt(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def infer_from_state_dict(sd) -> dict:
    """Architecture facts read from the tensors themselves — the ground truth."""
    out = {}
    if "view_embeddings.weight" in sd:
        out["max_views"] = int(sd["view_embeddings.weight"].shape[0])
    if "transformer_head.pos_embedding" in sd:
        out["hidden_dim"] = int(sd["transformer_head.pos_embedding"].shape[-1])
    layers = {
        k.split(".")[1]
        for k in sd
        if k.startswith("cross_attention.") and len(k.split(".")) > 1 and k.split(".")[1].isdigit()
    }
    if layers:
        out["cross_attention_layers"] = len(layers)
    return out


def adjusted_hidden_dim(model_block: dict) -> int | None:
    """Resolve the hidden_dim the model is ACTUALLY built with.

    Mirrors ``ModelConfig.get_adjusted_hidden_dim`` (``base_config.py:122``),
    which overrides the config's ``hidden_dim`` from the backbone name so the
    decoder width matches the backbone feature dim without a projection. For any
    ViT/ResNet backbone the JSON's ``hidden_dim`` is **dead** — comparing against
    it produces a false architecture mismatch on every correctly-paired ViT
    checkpoint, which is exactly what this function exists to prevent.
    """
    backbone = (model_block or {}).get("backbone_name") or ""
    if backbone.startswith("vit"):
        if "base" in backbone:
            return 768
        if "large" in backbone:
            return 1024
    elif backbone.startswith("resnet"):
        return 2048
    elif backbone.startswith("unet_"):
        return {
            "unet_efficientnet_b0": 512,
            "unet_efficientnet_b3": 512,
            "unet_resnet34": 512,
            "unet_mobilenet_v3": 256,
        }.get(backbone, 512)
    return (model_block or {}).get("hidden_dim")


def flatten_config(cfg: dict) -> dict:
    """Pull the architecture/split fields out of the nested JSON schema."""
    flat = {}
    for section in ("model", "training", "dataset", "multiview", "smal_model"):
        block = cfg.get(section)
        if isinstance(block, dict):
            for k, v in block.items():
                flat.setdefault(k, v)
    tc = (cfg.get("model") or {}).get("transformer_config")
    if isinstance(tc, dict):
        for k, v in tc.items():
            flat.setdefault(k, v)
    # Override with the value the model is really constructed at (see above).
    resolved = adjusted_hidden_dim(cfg.get("model") or {})
    if resolved is not None:
        flat["hidden_dim"] = resolved
    flat.setdefault("mode", cfg.get("mode"))
    return flat


def check_one(config_path: Path, reference: Path | None, problems: list, warnings: list) -> None:
    print(f"\n=== {config_path} ===")
    cfg = json.loads(config_path.read_text())
    flat = flatten_config(cfg)
    mode = cfg.get("mode")
    print(f"  mode: {mode}")

    resume = (cfg.get("training") or {}).get("resume_checkpoint")
    if not resume:
        problems.append(f"{config_path}: training.resume_checkpoint is null")
        return
    resume_path = Path(resume)
    if not resume_path.is_file():
        problems.append(f"{config_path}: resume_checkpoint missing on disk: {resume}")
        return

    ck = load_ckpt(resume_path)
    sd = ck.get("model_state_dict", ck)
    embedded = dict(ck.get("config") or {})
    inferred = infer_from_state_dict(sd)
    print(f"  resume checkpoint: {resume} (epoch {ck.get('epoch', '?')})")
    if inferred:
        print(f"  inferred from tensors: {inferred}")
    declared = (cfg.get("model") or {}).get("hidden_dim")
    if declared is not None and flat.get("hidden_dim") != declared:
        print(
            f"  note: config declares hidden_dim={declared} but backbone "
            f"'{(cfg.get('model') or {}).get('backbone_name')}' forces "
            f"{flat['hidden_dim']} via get_adjusted_hidden_dim() — the declared value is unused"
        )

    # ---- 1. architecture ---------------------------------------------------
    for field, ckpt_value in inferred.items():
        cfg_value = flat.get(field)
        if cfg_value is not None and int(cfg_value) != int(ckpt_value):
            problems.append(
                f"{config_path}: ARCHITECTURE MISMATCH on '{field}': the checkpoint's tensors say "
                f"{ckpt_value}, the config says {cfg_value}.\n"
                f"    load_state_dict(strict=False) would drop those layers, re-init them, and "
                f"reset the epoch counter to 0 — turning '10 more epochs' into "
                f"{(cfg.get('training') or {}).get('num_epochs')} epochs from scratch."
            )

    if embedded:
        for field in ARCH_FIELDS:
            if field in inferred:
                continue  # already checked against the tensors, which outrank the config block
            ckpt_value, cfg_value = embedded.get(field), flat.get(field)
            if ckpt_value is None or cfg_value is None:
                continue
            if ckpt_value != cfg_value:
                problems.append(
                    f"{config_path}: ARCHITECTURE MISMATCH on '{field}': checkpoint config says "
                    f"{ckpt_value!r}, prepared config says {cfg_value!r}. "
                    f"--base-config is probably not the JSON that produced this checkpoint."
                )
    else:
        warnings.append(
            f"{config_path}: the checkpoint carries no embedded config block, so only the "
            f"tensor-inferred fields {sorted(inferred) or '(none)'} could be verified."
        )

    if not problems:
        print("  architecture: consistent with the checkpoint")

    # ---- 2. split agreement ------------------------------------------------
    if reference is None:
        return
    if not reference.is_file():
        problems.append(f"reference checkpoint missing on disk: {reference}")
        return
    ref = load_ckpt(reference)
    ref_cfg = dict(ref.get("config") or {})
    if not ref_cfg:
        warnings.append(
            f"{reference}: no embedded config, so the reference arm's test split cannot be "
            f"verified against the constrained arm's. The benchmark will fall back to "
            f"library defaults — confirm they match {config_path} before trusting the deltas."
        )
        return

    print(f"  reference checkpoint: {reference} (epoch {ref.get('epoch', '?')})")
    split_ok = True
    for field in SPLIT_FIELDS:
        ref_value, cfg_value = ref_cfg.get(field), flat.get(field)
        if ref_value is None or cfg_value is None:
            continue
        if ref_value != cfg_value:
            split_ok = False
            problems.append(
                f"SPLIT MISMATCH on '{field}': the reference checkpoint says {ref_value!r} but "
                f"{config_path} says {cfg_value!r}.\n"
                f"    The two arms would be scored on DIFFERENT test frames and the delta would "
                f"be meaningless. Align the prepared config with the reference run."
            )
    if split_ok:
        print(f"  split params agree: {', '.join(f'{f}={flat.get(f)}' for f in SPLIT_FIELDS)}")

    # ---- 3. same data and same model ---------------------------------------
    ref_data = ref_cfg.get("data_path") or ref_cfg.get("dataset_path")
    cfg_data = (cfg.get("dataset") or {}).get("data_path")
    if ref_data and cfg_data and Path(ref_data).name != Path(cfg_data).name:
        warnings.append(
            f"dataset differs: reference trained on {Path(ref_data).name}, this config uses "
            f"{Path(cfg_data).name}. Fine if you repointed deliberately; otherwise the arms are "
            f"not comparable."
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", action="append", required=True, help="Prepared continuation config (repeatable)")
    p.add_argument(
        "--reference",
        action="append",
        default=[],
        help="Reference checkpoint for the matching --config, in the same order (repeatable)",
    )
    args = p.parse_args()

    configs = [Path(c) if Path(c).is_absolute() else REPO_ROOT / c for c in args.config]
    refs = [Path(r) if Path(r).is_absolute() else REPO_ROOT / r for r in args.reference]
    refs += [None] * (len(configs) - len(refs))

    problems: list[str] = []
    warnings: list[str] = []
    for cfg_path, ref in zip(configs, refs):
        if not cfg_path.is_file():
            problems.append(f"config not found: {cfg_path}")
            continue
        check_one(cfg_path, ref, problems, warnings)

    print()
    if warnings:
        print("WARNINGS (not blocking):")
        for w in warnings:
            print(f"  - {w}")
        print()
    if problems:
        print("BLOCKING PROBLEMS:", file=sys.stderr)
        for prob in problems:
            print(f"  - {prob}", file=sys.stderr)
        print("\nDo not submit until these are resolved.", file=sys.stderr)
        return 1
    print("Pre-flight OK — safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
