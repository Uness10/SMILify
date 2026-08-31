#!/usr/bin/env python3
"""Choose which frames of the exported test split to render — as REAL camera tracks.

The problem this solves
-----------------------
Consecutive rows of ``clip_<arm>.npz`` are **not** consecutive moments in time.
The single-view dataset is a multi-view HDF5 expanded one item per (sample,
view) — ``_build_single_view_index`` enumerates them sample-major::

    _sv_items = [(s0,v0), (s0,v1), ... (s0,vN), (s1,v0), (s1,v1), ...]

and the camera-centric split keeps whole SAMPLES together
(``export_poses.build_singleview_test_split``), preserving that ascending item
order. So N consecutive exported rows are *one instant seen from N cameras*.
Playing them back as video spins the animal like a top: it is not moving, the
camera is jumping around it, and because these are camera-centric checkpoints
each view carries its own ``global_rot`` relative to that camera.

This script undoes the interleave. It reconstructs the exact item list the
export walked, maps every npz row back to its ``(sample, view)``, and groups
rows into **per-camera tracks ordered by frame index**. A segment is then a run
of consecutive frames from ONE camera in ONE session — an actual animation.

Because the test split is only ~10 % of samples, a track is time-*sparse*: the
frames are in order but with gaps. The reported ``median_frame_gap`` says how
sparse, so the playback speed can be read honestly (a gap of 10 at 30 fps plays
~10x faster than life).

Usage
-----
    python scripts/prior_study/pick_segments.py \\
        --npz     prior_study_results/sv_reference/clip_sv_reference.npz \\
        --dataset SMILySTICKS_centred_reprojected_FIXED.h5 \\
        --smal-file 3D_model_prep/SMILy_STICK_limits_authored.pkl \\
        --segments 3 --length 300 --select worst \\
        --out prior_study_results/renders/segments.json

The reconstruction is **verified**: if the rebuilt item list does not have
exactly as many entries as the npz has frames, the split parameters are wrong
and the script refuses rather than emitting mislabelled windows. Take ``--seed``
/ ``--train-ratio`` / ``--val-ratio`` from the config that produced the
reference checkpoint if they differ from the defaults here.

Without ``--dataset`` it falls back to raw row windows and says loudly that they
are not time-ordered — kept only so an npz can be inspected with no HDF5 around.

CPU only, seconds (it reads ``view_mask`` and ``auxiliary/*``, never an image).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAD2DEG = 180.0 / np.pi


# ------------------------------------------------------------------- limits


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

    Free axes (wide-open +-pi) are excluded rather than averaged in as zeros:
    they can never contribute, and including them would scale every window by
    the same constant while making the numbers unreadable. The root is excluded
    too — LimitPrior pins it and the fitter drops it via the [3:] slice.
    """
    lo, hi = limits[..., 0], limits[..., 1]
    free = (lo <= -np.pi + 1e-4) & (hi >= np.pi - 1e-4)
    mask = ~free
    mask[0] = False
    if not mask.any():
        raise SystemExit("ERROR: every axis is wide open — nothing to rank windows by.")
    p = poses[:, mask]
    l_, h_ = lo[mask], hi[mask]
    over = np.maximum(np.maximum(l_ - p, 0.0), np.maximum(p - h_, 0.0))
    return over.mean(axis=1) * RAD2DEG


# ------------------------------------------------- reconstruct the item list


def rebuild_test_items(
    dataset_path: Path, seed: int, train_ratio: float, val_ratio: float, expect_rows: int
) -> Tuple[List[Tuple[int, int]], Dict[int, str], np.ndarray, np.ndarray, int]:
    """Return (items_per_npz_row, view_slot->camera, session[], frame_idx[], n_items_total).

    Mirrors ``_build_single_view_index`` + ``build_singleview_test_split``'s
    camera-centric branch. Everything here is metadata: ``view_mask`` and the
    ``auxiliary/*`` tables, never an image.
    """
    import h5py
    import torch

    with h5py.File(dataset_path, "r") as f:
        view_mask = f["multiview_images/view_mask"][:]  # (N, max_views) bool
        session = np.array([s.decode("utf-8") for s in f["auxiliary/session_name"][:]])
        frame_idx = f["auxiliary/frame_idx"][:].astype(np.int64)
        cam_order: List[str] = []
        meta = f.get("metadata")
        if meta is not None:
            raw = meta.attrs.get("canonical_camera_order", "[]")
            try:
                cam_order = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            except Exception:  # noqa: BLE001 - a missing camera name is cosmetic
                cam_order = []

    n_samples = int(view_mask.shape[0])

    # Same enumeration order as _build_single_view_index: sample-major.
    items: List[Tuple[int, int]] = [
        (s, int(v)) for s in range(n_samples) for v in np.where(view_mask[s])[0]
    ]

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    _, _, sample_test = torch.utils.data.random_split(
        range(n_samples), [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )
    test_samples = {int(s) for s in sample_test}

    # Ascending item order, exactly as the exporter walked it.
    test_items = [it for it in items if it[0] in test_samples]

    print(f"[pick]   dataset: {n_samples} samples, {len(items)} view-items, "
          f"max_views {view_mask.shape[1]}")
    print(f"[pick]   split  : seed={seed} train={train_ratio} val={val_ratio} "
          f"-> {n_train}/{n_val}/{n_test} samples -> {len(test_items)} test view-items")

    if len(test_items) != expect_rows:
        raise SystemExit(
            f"ERROR: rebuilt test split has {len(test_items)} items but the npz has {expect_rows}\n"
            f"       frames. The split parameters do not match the ones the export used, so every\n"
            f"       row would be attributed to the wrong camera and frame.\n"
            f"       Take seed / train_ratio / val_ratio from the config that produced the\n"
            f"       reference checkpoint and pass them as --seed / --train-ratio / --val-ratio.\n"
            f"       (The export console log prints the split it chose.)"
        )

    cam_names = {v: (cam_order[v] if v < len(cam_order) else f"view{v}") for v in range(view_mask.shape[1])}
    return test_items, cam_names, session, frame_idx, len(items)


# ------------------------------------------------------------------ picking


def build_tracks(
    test_items: List[Tuple[int, int]],
    cam_names: Dict[int, str],
    session: np.ndarray,
    frame_idx: np.ndarray,
) -> Dict[Tuple[str, str], List[int]]:
    """Group npz row indices into (session, camera) tracks, ordered in time."""
    tracks: Dict[Tuple[str, str], List[int]] = {}
    for row, (s, v) in enumerate(test_items):
        tracks.setdefault((str(session[s]), cam_names[v]), []).append(row)
    # test_items is ascending in sample, so each track is already ascending in
    # frame index — sort anyway rather than rely on it.
    for key, rows in tracks.items():
        rows.sort(key=lambda r: frame_idx[test_items[r][0]])
    return tracks


def pick_from_tracks(
    tracks: Dict[Tuple[str, str], List[int]],
    test_items: List[Tuple[int, int]],
    frame_idx: np.ndarray,
    score: np.ndarray,
    n_segments: int,
    length: int,
    select: str,
    one_per_camera: bool,
) -> List[dict]:
    """Slide a window of `length` consecutive in-track rows; rank by overshoot."""
    cands = []
    for (sess, cam), rows in sorted(tracks.items()):
        if len(rows) < length:
            continue
        arr = np.asarray(rows)
        s = score[arr]
        csum = np.concatenate([[0.0], np.cumsum(s)])
        win = (csum[length:] - csum[:-length]) / length
        for start in range(len(arr) - length + 1):
            cands.append({"session": sess, "camera": cam, "pos": start,
                          "rows": arr[start : start + length], "score": float(win[start])})

    if not cands:
        longest = max((len(r) for r in tracks.values()), default=0)
        raise SystemExit(
            f"ERROR: no (session, camera) track has {length} frames in the test split.\n"
            f"       The longest track is {longest}. Lower --length to at most that."
        )

    order = sorted(range(len(cands)), key=(lambda i: -cands[i]["score"]) if select == "worst" else (lambda i: i))
    chosen: List[dict] = []
    for i in order:
        c = cands[i]
        if one_per_camera and any(x["camera"] == c["camera"] and x["session"] == c["session"] for x in chosen):
            continue
        # No overlap within a track.
        if any(
            x["session"] == c["session"] and x["camera"] == c["camera"] and abs(x["pos"] - c["pos"]) < length
            for x in chosen
        ):
            continue
        chosen.append(c)
        if len(chosen) == n_segments:
            break

    if select == "even" and len(chosen) < n_segments:
        chosen = cands[:: max(len(cands) // max(n_segments, 1), 1)][:n_segments]

    if len(chosen) < n_segments:
        print(
            f"[pick] WARNING: only {len(chosen)} segment(s) fit; asked for {n_segments}. "
            f"Lower --length, or drop --one-per-camera.",
            file=sys.stderr,
        )

    segs = []
    for i, c in enumerate(sorted(chosen, key=lambda x: (x["session"], x["camera"], x["pos"]))):
        rows = c["rows"]
        frames = np.array([frame_idx[test_items[r][0]] for r in rows], dtype=np.int64)
        gaps = np.diff(frames)
        segs.append(
            {
                "name": f"seg{i:02d}_{c['camera']}_f{int(frames[0]):06d}",
                "indices": [int(r) for r in rows],
                # The real recording frame number per row, so the render's HUD
                # can show when this is rather than an export-order row index.
                "frames": [int(x) for x in frames],
                "session": c["session"],
                "camera": c["camera"],
                "frame_first": int(frames[0]),
                "frame_last": int(frames[-1]),
                "n_frames": int(len(rows)),
                "median_frame_gap": float(np.median(gaps)) if gaps.size else 1.0,
                "mean_overshoot_deg": c["score"],
            }
        )
    return segs


def pick_raw_windows(score: np.ndarray, n_frames: int, n_segments: int, length: int, select: str) -> List[dict]:
    """Fallback with no HDF5: raw row windows. NOT time-ordered — see the warning."""
    if length > n_frames:
        raise SystemExit(f"ERROR: --length {length} exceeds the clip's {n_frames} frames.")
    if select == "even":
        starts = [i * (n_frames - length) // max(n_segments - 1, 1) for i in range(n_segments)]
    else:
        csum = np.concatenate([[0.0], np.cumsum(score)])
        win = (csum[length:] - csum[:-length]) / length
        starts, order = [], np.argsort(-win)
        for s in order:
            s = int(s)
            if all(abs(s - c) >= length for c in starts):
                starts.append(s)
            if len(starts) == n_segments:
                break
        starts.sort()
    return [
        {
            "name": f"seg{i:02d}_rows{s:06d}",
            "indices": list(range(s, s + length)),
            "start": int(s),
            "length": int(length),
            "time_ordered": False,
        }
        for i, s in enumerate(starts)
    ]


# --------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, type=Path, help="REFERENCE arm's clip_*.npz")
    p.add_argument("--json", default=None, type=Path, help="Its sidecar (default: same stem, .json)")
    p.add_argument("--smal-file", default="3D_model_prep/SMILy_STICK_limits_authored.pkl", type=Path)
    p.add_argument(
        "--dataset",
        default=None,
        type=Path,
        help="The HDF5 the poses were exported from. WITHOUT it the segments are raw row "
        "windows, which are NOT time-ordered — see this file's docstring.",
    )
    p.add_argument("--seed", type=int, default=42, help="Split seed used by the export (default: 42)")
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--segments", type=int, default=3, help="How many segments to render (default: 3)")
    p.add_argument("--length", type=int, default=300, help="Frames per segment (default: 300)")
    p.add_argument(
        "--select",
        choices=["worst", "even"],
        default="worst",
        help="worst = the windows where the reference violates the authored limits most "
        "(default; the frames the prior is supposed to fix). even = spread across the clip.",
    )
    p.add_argument(
        "--one-per-camera",
        action="store_true",
        help="At most one segment per (session, camera), so the set spans viewpoints "
        "instead of stacking on the single worst track.",
    )
    p.add_argument("--out", required=True, type=Path, help="Where to write segments.json")
    args = p.parse_args()

    sidecar_path: Optional[Path] = args.json or args.npz.with_suffix(".json")
    if not args.npz.is_file():
        raise SystemExit(f"ERROR: npz not found: {args.npz}")
    if not sidecar_path.is_file():
        raise SystemExit(f"ERROR: sidecar not found: {sidecar_path}")

    sidecar = json.loads(sidecar_path.read_text())
    with np.load(args.npz) as z:
        poses = z["poses"].astype(np.float64)
    n_frames, n_joints = poses.shape[0], poses.shape[1]

    print(f"[pick] {args.npz}")
    print(f"[pick]   rows {n_frames}, joints {n_joints}, fps {sidecar.get('fps')}")
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
            "         clip. No segment can show the prior correcting anything, and --select\n"
            "         worst degenerates to an arbitrary choice. That is a RESULT — check\n"
            "         analysis/limit_violations.csv — not a reason to render more frames.",
            file=sys.stderr,
        )

    payload = {
        "source_npz": str(args.npz),
        "source_checkpoint": sidecar.get("source_checkpoint"),
        "source_input": sidecar.get("source_input"),
        "n_frames": n_frames,
        "fps": sidecar.get("fps", 30.0),
        "select": args.select,
        "smal_file": str(args.smal_file),
    }

    if args.dataset is not None:
        if not args.dataset.is_file():
            raise SystemExit(f"ERROR: dataset not found: {args.dataset}")
        test_items, cam_names, session, frame_idx, n_items_total = rebuild_test_items(
            args.dataset, args.seed, args.train_ratio, args.val_ratio, n_frames
        )
        tracks = build_tracks(test_items, cam_names, session, frame_idx)
        print(f"[pick]   tracks : {len(tracks)} (session, camera) pairs; "
              f"lengths {min(len(r) for r in tracks.values())}..{max(len(r) for r in tracks.values())}")
        segs = pick_from_tracks(
            tracks, test_items, frame_idx, score, args.segments, args.length, args.select, args.one_per_camera
        )
        payload.update(time_ordered=True, dataset=str(args.dataset), seed=args.seed,
                       train_ratio=args.train_ratio, val_ratio=args.val_ratio, segments=segs)
    else:
        print(
            "\n[pick] WARNING: no --dataset given, so segments are RAW ROW WINDOWS.\n"
            "       Consecutive rows of the npz are one instant seen from several cameras,\n"
            "       not consecutive time — rendering them makes the animal appear to spin.\n"
            "       Pass --dataset <the HDF5> to get real per-camera tracks.\n",
            file=sys.stderr,
        )
        segs = pick_raw_windows(score, n_frames, args.segments, args.length, args.select)
        payload.update(time_ordered=False, segments=segs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"\n[pick] wrote {args.out}")
    for s in segs:
        if s.get("camera"):
            print(
                f"        {s['name']}: {s['n_frames']} frames from camera {s['camera']} "
                f"(session {s['session']}), frames {s['frame_first']}..{s['frame_last']}, "
                f"median gap {s['median_frame_gap']:.0f}, mean overshoot {s['mean_overshoot_deg']:.3f} deg"
            )
        else:
            print(f"        {s['name']}: rows {s['start']}..{s['start'] + s['length']} (NOT time-ordered)")
    if args.dataset is not None:
        gaps = [s["median_frame_gap"] for s in segs]
        pct = 100.0 * n_frames / max(n_items_total, 1)
        print(
            f"\n[pick] These are single-camera tracks in frame order — real motion, not the\n"
            f"       camera-cycling that raw row order gives. The test split holds ~{pct:.0f}% of\n"
            f"       the dataset's view-items, so a track is time-SPARSE: median frame gap\n"
            f"       {min(gaps):.0f}-{max(gaps):.0f}, i.e. playback runs roughly that many times faster than life.\n"
            f"       Pass --fps {30.0 / max(np.median(gaps), 1):.1f} to the render for approximately real-time motion."
        )
    print(
        "\n[pick] Every arm must render THESE rows. Pass the same file to\n"
        "       render_clip_npz.py --segments-file for each arm."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
