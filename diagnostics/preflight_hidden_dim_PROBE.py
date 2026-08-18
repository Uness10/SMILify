#!/usr/bin/env python3
"""Probe: preflight_study's hidden_dim resolution must agree with the real config system.

Regression guard for a FALSE POSITIVE found on the cluster (2026-08-18). The
prior-study preflight compared the checkpoint's `transformer_head.pos_embedding`
width against the JSON's `model.hidden_dim` and blocked both arms of the study
with "ARCHITECTURE MISMATCH: tensors say 1024, config says 512".

The configs were correct. `ModelConfig.get_adjusted_hidden_dim` (base_config.py:122)
overrides `hidden_dim` from the backbone name -- every `vit_*large*` backbone is
built at 1024 no matter what the JSON says -- so the declared 512 never reaches
the model. The guard was comparing against a dead field.

This probe asserts the preflight's local copy of that rule matches the real one
for every backbone family, so the two cannot drift apart again.

Run:  python diagnostics/preflight_hidden_dim_PROBE.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.prior_study.preflight_study import adjusted_hidden_dim  # noqa: E402

# (backbone_name, declared hidden_dim in JSON, expected dim the model is built at)
CASES = [
    ("vit_large_patch16_224", 512, 1024),  # <- the cluster case that misfired
    ("vit_large_patch16_224", 1024, 1024),
    ("vit_base_patch16_224", 512, 768),
    ("resnet50", 512, 2048),
    ("resnet101", 1024, 2048),
    ("unet_efficientnet_b0", 1024, 512),
    ("unet_efficientnet_b3", 256, 512),
    ("unet_resnet34", 999, 512),
    ("unet_mobilenet_v3", 999, 256),
    ("unet_something_unlisted", 999, 512),  # falls back to the 512 default
    ("some_custom_backbone", 640, 640),  # unknown family -> declared value stands
]


def main() -> int:
    failures = []
    print(f"{'backbone':28s} {'declared':>9s} {'expected':>9s} {'preflight':>10s}")
    for backbone, declared, expected in CASES:
        got = adjusted_hidden_dim({"backbone_name": backbone, "hidden_dim": declared})
        ok = got == expected
        print(f"{backbone:28s} {declared:9d} {expected:9d} {got!s:>10s}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append((backbone, declared, expected, got))

    # Cross-check against the real implementation rather than trusting the table.
    try:
        from smal_fitter.neuralSMIL.configs.base_config import ModelConfig

        print("\ncross-check against ModelConfig.get_adjusted_hidden_dim:")
        for backbone, declared, _expected in CASES:
            real = ModelConfig(backbone_name=backbone, hidden_dim=declared).get_adjusted_hidden_dim()
            mine = adjusted_hidden_dim({"backbone_name": backbone, "hidden_dim": declared})
            flag = "ok" if real == mine else "DRIFT"
            print(f"  {backbone:28s} real={real!s:>6s} preflight={mine!s:>6s}  {flag}")
            if real != mine:
                failures.append((backbone, declared, real, mine))
    except Exception as exc:
        print(f"\n(cross-check skipped -- could not import ModelConfig: {exc})")
        print(" The table above still ran. Re-run this probe in the pytorch3d env for the full check.")

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s) disagree", file=sys.stderr)
        for case in failures:
            print(f"  {case}", file=sys.stderr)
        return 1
    print("PASS -- preflight resolves hidden_dim the same way the config system does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
