#!/usr/bin/env python3
"""
Resolve the two undetermined facts about the SBeA release, empirically.

    1. How are the 96 columns of id3d.mat's `coords3d` laid out?
       Four orderings are plausible and the MAT file carries no metadata to
       distinguish them.
    2. Which axis of caliParas.mat's `cam_mat_all` (3,4,4) is the camera?

Both are settled by the same test: reproject the 3D points through each
candidate projection matrix and count how many land inside the image. The
correct combination puts ~all points on the mouse; every wrong one scatters
them off-frame or behind the camera. This is unit-free, so it does not matter
whether SBeA stores millimetres or metres.

Run:
    conda activate "$HPCWORK/conda_envs/pytorch3d"
    python diagnostics/sbea_decode_PROBE.py \
        "/hpcwork/<user>/datasets/SBeA/fig2_data/pose tracking/rec11-A1A2-20220803" \
        --gt-dir "/hpcwork/<user>/datasets/SBeA/SM_fig1_data/gt_data"

Pass the session STEM — the path without the -caliParas.mat / -id3d.mat suffix.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

N_KP = 16
N_ANIMALS = 2


def rule(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# ------------------------------------------------------- MATLAB unwrapping

def unwrap(o):
    """Peel MATLAB's 1x1 cell/object wrappers until something real appears."""
    while isinstance(o, np.ndarray) and o.dtype == object and o.size == 1:
        o = o.flat[0]
    return o


def dump(o, name: str = "", depth: int = 0, max_depth: int = 4) -> None:
    pad = "  " * depth
    o = unwrap(o)
    if depth > max_depth:
        print(f"{pad}{name}: ...")
        return
    fields = getattr(o, "_fieldnames", None)
    if fields:
        print(f"{pad}{name}: struct{{{', '.join(fields)}}}")
        for f in fields:
            dump(getattr(o, f), f, depth + 1, max_depth)
    elif isinstance(o, np.ndarray):
        if o.dtype == object:
            print(f"{pad}{name}: cell{o.shape}")
            for i, sub in enumerate(o.flat):
                if i >= 6:
                    print(f"{pad}  ... {o.size - 6} more")
                    break
                dump(sub, f"[{i}]", depth + 1, max_depth)
        else:
            flat = o.ravel()
            preview = np.array2string(flat[:6], precision=4, suppress_small=True)
            print(f"{pad}{name}: array{o.shape} {o.dtype}  {preview}")
    else:
        print(f"{pad}{name}: {type(o).__name__}  {o}")


# ------------------------------------------------------------- calibration

def load_calibration(stem: Path):
    from scipy.io import loadmat

    path = Path(str(stem) + "-caliParas.mat")
    md = loadmat(path, struct_as_record=False, squeeze_me=False)

    rule(f"CALIBRATION  —  {path.name}")
    root = None
    for k, v in md.items():
        if not k.startswith("__"):
            print(f"\n[{k}]")
            dump(v, k)
            if root is None:
                root = unwrap(v)

    cam_mat = np.asarray(getattr(root, "cam_mat_all"))
    print(f"\ncam_mat_all: shape {cam_mat.shape}")
    return cam_mat, root


def candidate_projections(cam_mat: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Every reading of a (3,4,4) array as four 3x4 projection matrices."""
    out: dict[str, list[np.ndarray]] = {}
    if cam_mat.shape == (3, 4, 4):
        out["cam_mat_all[:, :, k]"] = [cam_mat[:, :, k] for k in range(4)]
        out["cam_mat_all[k]"] = [cam_mat[k] for k in range(4)]           # (4,4) -> take first 3 rows
        out["cam_mat_all[k][:3]"] = [cam_mat[k][:3] for k in range(4)]
        out["cam_mat_all[:, k, :]"] = [cam_mat[:, k, :] for k in range(4)]
    # Normalise: every candidate must end up 3x4.
    cleaned = {}
    for name, mats in out.items():
        ok = []
        for m in mats:
            m = np.asarray(m, dtype=np.float64)
            if m.shape == (4, 4):
                m = m[:3]
            if m.shape == (3, 4):
                ok.append(m)
        if len(ok) == 4:
            cleaned[name] = ok
    return cleaned


# ---------------------------------------------------------------- 3D poses

def load_coords(stem: Path):
    from scipy.io import loadmat

    path = Path(str(stem) + "-id3d.mat")
    md = loadmat(path, struct_as_record=False, squeeze_me=True)
    coords = np.asarray(md["coords3d"], dtype=np.float64)
    names = md.get("name3d")
    names = [str(n) for n in np.atleast_1d(names)]

    rule(f"3D POSES  —  {path.name}")
    print(f"coords3d : {coords.shape}  {coords.dtype}")
    print(f"name3d   : {names}")
    finite = np.isfinite(coords)
    print(f"finite   : {100 * finite.mean():.1f}%   "
          f"fully-NaN frames: {int((~finite).all(axis=1).sum())}")
    with np.errstate(invalid="ignore"):
        print(f"range    : min {np.nanmin(coords):.2f}  max {np.nanmax(coords):.2f}  "
              f"(units unknown — the reprojection test does not care)")
    return coords, names


def layout_hypotheses(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Reshape (F, 96) into (F, A, J, 3) under each plausible column order."""
    f = coords.shape[0]
    h: dict[str, np.ndarray] = {}

    # A) animal-major, then coordinate blocks:  a1[X*16, Y*16, Z*16], a2[...]
    h["animal / coord-block / kp"] = (
        coords.reshape(f, N_ANIMALS, 3, N_KP).transpose(0, 1, 3, 2))

    # B) animal-major, then per-keypoint xyz triples: a1[(x,y,z)*16], a2[...]
    h["animal / kp / xyz"] = coords.reshape(f, N_ANIMALS, N_KP, 3)

    # C) coordinate blocks outermost, spanning both animals:
    #    X for all 32 points, then Y, then Z
    h["coord-block / animal / kp"] = (
        coords.reshape(f, 3, N_ANIMALS, N_KP).transpose(0, 2, 3, 1))

    # D) keypoint-major triples spanning both animals interleaved
    h["kp / animal / xyz"] = (
        coords.reshape(f, N_KP, N_ANIMALS, 3).transpose(0, 2, 1, 3))

    return h


# ------------------------------------------------------------------- score

def image_size(stem: Path) -> tuple[int, int]:
    import cv2

    for cam in (0, 1):
        p = Path(f"{stem}-camera-{cam}.avi")
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            hgt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if w and hgt:
                print(f"\nimage size from {p.name}: {w} x {hgt}   ({n} frames)")
                return w, hgt
    print("\n!! no video found — falling back to 640x480")
    return 640, 480


def project(P: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pts (N,3) -> (N,2) pixels and (N,) depth."""
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = hom @ P.T
    w = cam[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = cam[:, :2] / w[:, None]
    return uv, w


def score(pts: np.ndarray, mats: list[np.ndarray], w: int, h: int) -> float:
    """Fraction of points that project in front of the camera and on-image."""
    good = total = 0
    for P in mats:
        uv, depth = project(P, pts)
        ok = np.isfinite(uv).all(axis=1) & (depth > 0)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        good += int(ok.sum())
        total += len(pts)
    return good / max(total, 1)


def sample_points(arr: np.ndarray, n_frames: int = 300) -> np.ndarray:
    """(F,A,J,3) -> (N,3) finite points from evenly spaced frames."""
    f = arr.shape[0]
    idx = np.linspace(0, f - 1, min(n_frames, f)).astype(int)
    pts = arr[idx].reshape(-1, 3)
    return pts[np.isfinite(pts).all(axis=1)]


def spread(arr: np.ndarray) -> float:
    """Median within-animal diameter — a sanity cross-check on the winner."""
    idx = np.linspace(0, arr.shape[0] - 1, min(200, arr.shape[0])).astype(int)
    d = []
    for fr in arr[idx]:
        for a in fr:
            a = a[np.isfinite(a).all(axis=1)]
            if len(a) > 2:
                d.append(np.linalg.norm(a[:, None] - a[None], axis=-1).max())
    return float(np.median(d)) if d else float("nan")


# ------------------------------------------------------------ bodypart names

def bodypart_names(gt_dir: Path | None) -> list[str]:
    if not gt_dir or not gt_dir.is_dir():
        return []
    csvs = sorted(gt_dir.glob("*.csv"))
    if not csvs:
        return []
    with open(csvs[0]) as fh:
        fh.readline()                       # scorer
        parts = fh.readline().rstrip("\n").split(",")[1:]
    seen: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.append(p)
    rule(f"BODYPART ORDER  —  {csvs[0].name}")
    for i, nm in enumerate(seen):
        print(f"  {i:>2}  {nm}")
    print(f"\n{len(seen)} unique bodyparts — this is the joint_lookup.csv source of truth.")
    return seen


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="session path WITHOUT the -caliParas.mat suffix")
    ap.add_argument("--gt-dir", default=None, help="SM_fig1_data/gt_data, for bodypart names")
    args = ap.parse_args()

    stem = Path(args.stem)
    if not Path(str(stem) + "-id3d.mat").exists():
        print(f"ERROR: {stem}-id3d.mat not found. Pass the session stem, not a file.",
              file=sys.stderr)
        return 1

    cam_mat, _root = load_calibration(stem)
    coords, names = load_coords(stem)

    if coords.shape[1] != N_ANIMALS * N_KP * 3:
        print(f"\n!! expected {N_ANIMALS * N_KP * 3} columns, got {coords.shape[1]} — "
              f"the animal/keypoint counts differ from the paper. Stopping.")
        return 1

    w, h = image_size(stem)
    projections = candidate_projections(cam_mat)
    hypotheses = layout_hypotheses(coords)

    rule("RESOLVING: column layout x projection-matrix axis")
    print("Score = fraction of reprojected points in front of the camera AND on-image,")
    print("averaged over all four views. The correct pair should stand alone.\n")

    print(f"{'column layout':<30}{'projection reading':<26}{'score':>8}")
    print("-" * 64)
    results = []
    for (hname, arr), (pname, mats) in itertools.product(hypotheses.items(),
                                                         projections.items()):
        s = score(sample_points(arr), mats, w, h)
        results.append((s, hname, pname, arr))
        print(f"{hname:<30}{pname:<26}{s:>7.1%}")

    results.sort(key=lambda r: -r[0])
    best_s, best_h, best_p, best_arr = results[0]
    runner_up = results[1][0] if len(results) > 1 else 0.0

    rule("VERDICT")
    if best_s < 0.5:
        print(f"INCONCLUSIVE — best score only {best_s:.1%}.")
        print("No layout reprojects onto the image. Likely causes: the projection")
        print("matrices need R/t composed separately, or the 3D is in a world frame")
        print("the matrices do not map from. Paste this whole report back.")
        return 0

    print(f"column layout      : {best_h}")
    print(f"projection reading : {best_p}")
    print(f"score              : {best_s:.1%}   (runner-up {runner_up:.1%})")
    print(f"animal identities  : {names}")
    print(f"median within-animal diameter: {spread(best_arr):.2f} (dataset units)")
    if best_s - runner_up < 0.15:
        print("\n!! WARNING: the top two are close. Do not trust this without the")
        print("   gt_data reprojection check before converting all 40 sessions.")
    else:
        print("\nUnambiguous. Safe to write the converter against this layout.")

    print(f"\ntracks array for points3d.h5 would be: "
          f"({best_arr.shape[0]}, {best_arr.shape[1]}, {best_arr.shape[2]}, 3)")

    bodypart_names(Path(args.gt_dir) if args.gt_dir else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
