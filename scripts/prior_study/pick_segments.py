#!/usr/bin/env python3
"""Choose which frame windows of the exported test split to render, ONCE, for all arms.

Why this exists
---------------
The renders only mean something if every arm shows the **same frames**. The
exported ``clip_<arm>.npz`` files are all in test-split order (``export_poses.py``
walks the same ``Subset`` for every arm, and ``preflight_study.py`` verifies the
seed/ratio agreement that makes that true), so frame *i* is the same animal at
the same instant in every arm. This script picks the windows once and writes them
to a JSON that every render job reads — rather than letting each job choose,
which would silently produce an un-comparable grid.

It also picks *informatively*. A window where the reference model never leaves
the authored ranges cannot show the prior doing anything, no matter how good the
prior is. ``--select worst`` ranks windows by how much the REFERENCE violates and
takes the top ones, so the renders are aimed at the frames the study is about.

Usage
-----
    python scripts/prior_study/pick_segments.py \\
        --npz  prior_study_results/sv_reference/clip_sv_reference.npz \\
        --json prior_study_results/sv_reference/clip_sv_reference.json \\
        --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \\
        --segments 3 --length 300 --select worst \\
        --out prior_study_results/renders/segments.json

CPU only, seconds. Run it on the login node.

The ``--npz`` MUST be the reference arm: ranking by a constrained arm would pick
the windows where the *already-corrected* model still violates, which is a
different and much less interesting question.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAD2DEG = 180.0 / np.pi


def load_limits(smal_file: Path, n_joints: int) -> np.ndarray:
    with open(smal_file, "rb") as f:
        dd = pickle.load(f, encoding="latin1")
    jl = dd.get("joint_limits")
    if jl is None:
        raise SystemExit(
            f"ERROR: {smal_file} has no 'joint_limits' key.\n"
            "       Point --smal-file at 3D_model_prep/SMILy_STICK_limits_authored.pkl."
        )
    jl = np.asarray(jl, dtype=np.float64)
    if jl.shape != (n_joints, 3, 2):
        raise SystemExit(f"ERROR: joint_limits shape {jl.shape}, expected {(n_joints, 3, 2)}.")
    return jl


def per_frame_overshoot(poses: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """(F,) mean overshoot in degrees over the AUTHORED non-root axes.

    Free axes (wide-open ±π) are excluded rather than averaged in as zeros: they
    can never contribute, and including them would just scale every window by the
    same constant while making the numbers unreadable.
    """
    lo, hi = limits[..., 0], limits[..., 1]
    free = (lo <= -np.pi + 1e-4) & (hi >= np.pi - 1e-4)
    mask = ~free
    mask[0] = False  # root is pinned and dropped by the fitter's [3:] slice

    if not mask.any():
        raise SystemExit("ERROR: every axis is wide open — nothing to rank windows by.")

    p = poses[:, mask]  # (F, n_authored)
    l_, h_ = lo[mask], hi[mask]
    overshoot = np.maximum(np.maximum(l_ - p, 0.0), np.maximum(p - h_, 0.0))
    return overshoot.mean(axis=1) * RAD2DEG


def pick_windows(
    score: np.ndarray,
    n_frames: int,
    n_segments: int,
    length: int,
    select: str,
    min_gap: int,
) -> List[dict]:
    if length > n_frames:
        raise SystemExit(f"ERROR: --length {length} exceeds the clip's {n_frames} frames.")

    if select == "even":
        starts = [i * (n_frames - length) // max(n_segments - 1, 1) for i in range(n_segments)]
        return [{"name": f"seg{i:02d}_f{s:06d}", "start": int(s), "length": int(length)} for i, s in enumerate(starts)]

    # "worst": rank non-overlapping windows by mean overshoot.
    # Uniform-filter the per-frame score via a cumulative sum — F is ~50k and
    # n windows is F-length, so an explicit loop would be needlessly slow.
    csum = np.concatenate([[0.0], np.cumsum(score)])
    win = (csum[length:] - csum[:-length]) / length  # (F - length + 1,)

    chosen: List[int] = []
    order = np.argsort(-win)
    for s in order:
        s = int(s)
        if all(abs(s - c) >= max(length, min_gap) for c in chosen):
            chosen.append(s)
        if len(chosen) == n_segments:
            break
    if len(chosen) < n_segments:
        print(
            f"[pick] WARNING: only {len(chosen)} non-overlapping window(s) fit; "
            f"asked for {n_segments}. Reduce --length or --segments.",
            file=sys.stderr,
        )
    chosen.sort()
    return [
        {"name": f"seg{i:02d}_f{s:06d}", "start": int(s), "length": int(length), "mean_overshoot_deg": float(win[s])}
        for i, s in enumerate(chosen)
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, type=Path, help="REFERENCE arm's clip_*.npz")
    p.add_argument("--json", default=None, type=Path, help="Its sidecar (default: same stem, .json)")
    p.add_argument("--smal-file", default="3D_model_prep/SMILy_STICK_limits_authored.pkl", type=Path)
    p.add_argument("--segments", type=int, default=3, help="How many windows to render (default: 3)")
    p.add_argument("--length", type=int, default=300, help="Frames per window (default: 300 = 10 s at 30 fps)")
    p.add_argument(
        "--select",
        choices=["worst", "even"],
        default="worst",
        help="worst = windows where the reference violates the authored limits most "
        "(default; these are the frames the prior is supposed to fix). "
        "even = evenly spaced across the clip, for an unbiased look.",
    )
    p.add_argument("--min-gap", type=int, default=0, help="Extra minimum spacing between window starts")
    p.add_argument("--out", required=True, type=Path, help="Where to write segments.json")
    args = p.parse_args()

    sidecar_path: Optional[Path] = args.json or args.npz.with_suffix(".json")
    if not args.npz.is_file():
        raise SystemExit(f"ERROR: npz not found: {args.npz}")
    if not sidecar_path.is_file():
        raise SystemExit(f"ERROR: sidecar not found: {sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text())
    with np.load(args.npz) as z:
        poses = z["poses"].astype(np.float64)  # (F, J, 3) axis-angle
    n_frames, n_joints = poses.shape[0], poses.shape[1]

    print(f"[pick] {args.npz}")
    print(f"[pick]   frames {n_frames}, joints {n_joints}, fps {sidecar.get('fps')}")
    print(f"[pick]   source: {sidecar.get('source_checkpoint')}")

    limits = load_limits(args.smal_file, n_joints)
    score = per_frame_overshoot(poses, limits)
    print(
        f"[pick]   overshoot over authored axes: mean {score.mean():.3f} deg, "
        f"p95 {np.percentile(score, 95):.3f} deg, max {score.max():.3f} deg"
    )
    if score.max() <= 1e-9:
        print(
            "[pick]   NOTE: the reference never leaves the authored ranges anywhere in this\n"
            "         clip. No window can show the prior correcting anything, and --select\n"
            "         worst degenerates to an arbitrary choice. That is a RESULT — check\n"
            "         analysis/limit_violations.csv — not a reason to render more frames.",
            file=sys.stderr,
        )

    segs = pick_windows(score, n_frames, args.segments, args.length, args.select, args.min_gap)

    payload = {
        "source_npz": str(args.npz),
        "source_checkpoint": sidecar.get("source_checkpoint"),
        "source_input": sidecar.get("source_input"),
        "n_frames": n_frames,
        "fps": sidecar.get("fps", 30.0),
        "select": args.select,
        "smal_file": str(args.smal_file),
        "segments": segs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"\n[pick] wrote {args.out}")
    for s in segs:
        extra = f", mean overshoot {s['mean_overshoot_deg']:.3f} deg" if "mean_overshoot_deg" in s else ""
        print(f"        {s['name']}: frames {s['start']}..{s['start'] + s['length']}{extra}")
    print(
        "\n[pick] Every arm must render THESE windows. Pass the same file to\n"
        "       render_clip_npz.py --segments-file for each arm."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
