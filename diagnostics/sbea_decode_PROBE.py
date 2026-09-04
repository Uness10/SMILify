#!/usr/bin/env python3
"""
v3 — Resolve the SBeA coords3d column layout and the world-frame convention.

WHAT CHANGED FROM v2, AND WHY
-----------------------------
v2 ranked candidates by "fraction of reprojected points on-image". That is a
weak signal: a scrambled layout still puts points roughly inside the arena
volume, so everything scored 20-60% and nothing separated. v2's own output
carried the real evidence — a median within-animal diameter of 838 units when
the cameras sit 714-1002 units from the origin, i.e. millimetres, i.e. a mouse
supposedly 84 cm across.

So v3 ranks by RIGID-BODY CONSISTENCY instead. Distances between keypoints on
one animal are invariant to any rotation, translation or choice of frame, so
this test is independent of the calibration question entirely. The skull is
rigid: the left-ear-to-right-ear distance must be near-constant across frames.
Under a wrong layout that distance mixes one point's X with another's Y and
becomes noise. Coefficient of variation therefore separates the layouts
cleanly, and being a ratio it needs no knowledge of the units.

Only once the layout is fixed does v3 search the frame convention. id3d.mat is
"rotated to ground" per the tracker README, while cam_mat_all projects from the
camera-3 frame (camera 3 has |t| = 0). The rotation/translation stored in
caliParas is that ground transform and must be undone before reprojecting.

Run:
    python diagnostics/sbea_decode_PROBE.py \
        "<...>/fig2_data/pose tracking/rec11-A1A2-20220803" \
        --gt-dir "<...>/SM_fig1_data/gt_data"
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

N_KP, N_ANIMALS, N_DIM = 16, 2, 3

# Index into the DLC bodypart order, confirmed from the gt_data CSV header.
NOSE, EAR_L, EAR_R, NECK = 0, 1, 2, 3
BACK, ROOT_TAIL, MID_TAIL, TIP_TAIL = 12, 13, 14, 15

# Pairs spanning rigid or near-rigid structure. The skull pair is the strongest
# signal; the tail segments are individually rigid even though the chain flexes.
RIGID_PAIRS = [
    ("ear_L-ear_R", EAR_L, EAR_R),
    ("nose-neck", NOSE, NECK),
    ("root-mid tail", ROOT_TAIL, MID_TAIL),
    ("mid-tip tail", MID_TAIL, TIP_TAIL),
]


def rule(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


# ------------------------------------------------------- MATLAB unwrapping

def unwrap(o):
    while isinstance(o, np.ndarray) and o.dtype == object and o.size == 1:
        o = o.flat[0]
    return o


def load_calibration(stem: Path):
    from scipy.io import loadmat

    md = loadmat(Path(f"{stem}-caliParas.mat"), struct_as_record=False, squeeze_me=False)
    root = next(unwrap(v) for k, v in md.items() if not k.startswith("__"))
    cam_mat = np.asarray(root.cam_mat_all, dtype=np.float64)
    R_g = np.asarray(unwrap(root.rotation), dtype=np.float64).reshape(3, 3)
    t_g = np.asarray(unwrap(root.translation), dtype=np.float64).ravel()[:3]

    rule("CALIBRATION")
    print(f"cam_mat_all {cam_mat.shape} -> reading [:, :, k] as 4 projection matrices")
    print(f"ground rotation R:\n{np.array2string(R_g, precision=4)}")
    print(f"ground translation t: {np.array2string(t_g, precision=3)}")
    print(f"det(R) = {np.linalg.det(R_g):.6f}   (should be +1 for a pure rotation)")
    return [cam_mat[:, :, k] for k in range(cam_mat.shape[2])], R_g, t_g


def decompose(P: np.ndarray):
    from scipy.linalg import rq

    K, R = rq(P[:, :3])
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    if K[2, 2] != 0:
        K = K / K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    if np.linalg.det(R) < 0:
        R, t = -R, -t
    return K, R, t


# ---------------------------------------------------------------- 3D poses

def load_coords(stem: Path):
    from scipy.io import loadmat

    md = loadmat(Path(f"{stem}-id3d.mat"), struct_as_record=False, squeeze_me=True)
    coords = np.asarray(md["coords3d"], dtype=np.float64)
    names = [str(n) for n in np.atleast_1d(md.get("name3d"))]
    rule("3D POSES")
    print(f"coords3d {coords.shape}   identities {names}")
    return coords, names


def all_layouts(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Every way (F, 96) unflattens into (F, animal, keypoint, coord).

    96 = 2 x 16 x 3 and the three factors are distinct, so each of the six
    reshape orders assigns axis roles unambiguously.
    """
    f = coords.shape[0]
    role = {N_ANIMALS: "animal", N_KP: "kp", N_DIM: "xyz"}
    out: dict[str, np.ndarray] = {}
    for perm in itertools.permutations((N_ANIMALS, N_KP, N_DIM)):
        arr = coords.reshape(f, *perm)
        ax = {perm[i]: i + 1 for i in range(3)}
        arr = np.transpose(arr, (0, ax[N_ANIMALS], ax[N_KP], ax[N_DIM]))
        out[" / ".join(role[p] for p in perm)] = arr
    return out


# ----------------------------------------------------------- rigidity test

def rigidity(arr: np.ndarray, n_frames: int = 500) -> dict:
    """Distance statistics for rigid pairs. Frame-invariant by construction."""
    idx = np.linspace(0, arr.shape[0] - 1, min(n_frames, arr.shape[0])).astype(int)
    sub = arr[idx]                                    # (n, A, J, 3)
    stats: dict = {}

    cvs = []
    for label, i, j in RIGID_PAIRS:
        d = np.linalg.norm(sub[:, :, i] - sub[:, :, j], axis=-1).ravel()
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) < 10:
            stats[label] = (np.nan, np.nan)
            continue
        med, cv = float(np.median(d)), float(np.std(d) / max(np.mean(d), 1e-9))
        stats[label] = (med, cv)
        cvs.append(cv)

    span = np.linalg.norm(sub[:, :, NOSE] - sub[:, :, TIP_TAIL], axis=-1).ravel()
    span = span[np.isfinite(span)]
    stats["nose-tip span"] = float(np.median(span)) if len(span) else np.nan

    d = np.linalg.norm(sub[:, :, :, None, :] - sub[:, :, None, :, :], axis=-1)
    stats["diameter"] = float(np.median(np.nanmax(d, axis=(2, 3))))
    stats["mean_cv"] = float(np.mean(cvs)) if cvs else np.inf

    ear = stats.get("ear_L-ear_R", (np.nan, np.nan))[0]
    stats["span/ear"] = float(stats["nose-tip span"] / ear) if ear and ear > 0 else np.nan
    return stats


# -------------------------------------------------------- frame conventions

def frame_variants(R_g: np.ndarray, t_g: np.ndarray) -> dict:
    """Candidate maps from the stored 3D frame into the calibration frame."""
    Rt = R_g.T
    return {
        "as stored (no transform)": lambda X: X,
        "R^T (X - t)   [undo ground]": lambda X: (X - t_g) @ Rt.T,
        "R X + t       [apply ground]": lambda X: X @ R_g.T + t_g,
        "R^T X         [rotation only]": lambda X: X @ Rt.T,
        "R (X - t)": lambda X: (X - t_g) @ R_g.T,
    }


def project(P: np.ndarray, pts: np.ndarray):
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = hom @ P.T
    w = cam[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = cam[:, :2] / w[:, None]
    return uv, w


def onimage(pts: np.ndarray, mats, w: int, h: int) -> float:
    good = total = 0
    for P in mats:
        uv, depth = project(P, pts)
        ok = np.isfinite(uv).all(axis=1) & (depth > 0)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        good += int(ok.sum())
        total += len(pts)
    return good / max(total, 1)


def sample(arr: np.ndarray, n: int = 400) -> np.ndarray:
    idx = np.linspace(0, arr.shape[0] - 1, min(n, arr.shape[0])).astype(int)
    pts = arr[idx].reshape(-1, 3)
    return pts[np.isfinite(pts).all(axis=1)]


# ------------------------------------------------------------------- video

def video_info(stem: Path, n_rows: int):
    import cv2

    rule("VIDEO")
    size, counts = None, []
    for cam in range(8):
        p = Path(f"{stem}-camera-{cam}.avi")
        if not p.exists():
            continue
        cap = cv2.VideoCapture(str(p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        size = size or (w, h)
        counts.append(n)
        print(f"  camera-{cam}: {w}x{h}  {n} frames")
    if not size:
        return 640, 480, None
    n_vid = counts[0]
    if n_vid and n_rows % n_vid == 0:
        print(f"\n  coords3d {n_rows} rows = {n_rows // n_vid}x the {n_vid}-frame video.")
        print("  Candidate pairings tested below.")
    return size[0], size[1], n_vid


def test_subsampling(arr, mats, R_g, t_g, w, h, n_vid, frame_fn):
    """Which 900 of the 27000 rows correspond to the released excerpt?"""
    if not n_vid or arr.shape[0] == n_vid:
        return
    stride = arr.shape[0] // n_vid
    rule(f"WHICH ROWS MATCH THE {n_vid}-FRAME EXCERPT?")
    print("Reprojection cannot distinguish these — every candidate is valid 3D.")
    print("Listed for the record; settle it by overlaying on real frames.\n")
    cands = {
        f"first {n_vid} rows": slice(0, n_vid),
        f"last {n_vid} rows": slice(-n_vid, None),
        f"every {stride}th row": slice(0, None, stride),
        f"middle {n_vid} rows": slice((arr.shape[0] - n_vid) // 2,
                                      (arr.shape[0] - n_vid) // 2 + n_vid),
    }
    for label, sl in cands.items():
        pts = frame_fn(sample(arr[sl]))
        print(f"  {label:<24} on-image {onimage(pts, mats, w, h):6.1%}")


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--gt-dir", default=None)
    args = ap.parse_args()

    stem = Path(args.stem)
    if not Path(f"{stem}-id3d.mat").exists():
        print(f"ERROR: {stem}-id3d.mat not found.", file=sys.stderr)
        return 1

    mats, R_g, t_g = load_calibration(stem)
    coords, names = load_coords(stem)
    w, h, n_vid = video_info(stem, coords.shape[0])

    print("\nrecovered intrinsics:")
    for k, P in enumerate(mats):
        K, _R, t = decompose(P)
        print(f"  cam{k}: fx={K[0, 0]:7.1f} fy={K[1, 1]:7.1f} "
              f"cx={K[0, 2]:6.1f} cy={K[1, 2]:6.1f} |t|={np.linalg.norm(t):7.1f}")

    # ---- STEP 1: layout, by rigid-body consistency (no projection involved)
    layouts = all_layouts(coords)
    rule("STEP 1 — COLUMN LAYOUT, by rigid-body consistency")
    print("A rigid pair's length cannot vary. CV = std/mean of that length over")
    print("frames; the correct layout minimises it. Frame-invariant, unit-free.\n")
    print(f"{'layout (outer/mid/inner)':<26}{'ear-ear':>10}{'CV':>8}"
          f"{'nose-tip':>10}{'diam':>8}{'span/ear':>10}{'meanCV':>9}")
    print("-" * 81)

    scored = []
    for name, arr in layouts.items():
        s = rigidity(arr)
        ear_med, ear_cv = s["ear_L-ear_R"]
        scored.append((s["mean_cv"], name, arr, s))
        print(f"{name:<26}{ear_med:>10.1f}{ear_cv:>8.3f}"
              f"{s['nose-tip span']:>10.1f}{s['diameter']:>8.1f}"
              f"{s['span/ear']:>10.1f}{s['mean_cv']:>9.3f}")

    scored.sort(key=lambda r: r[0])
    best_cv, best_name, best_arr, best_stats = scored[0]
    runner_cv = scored[1][0]

    print(f"\nbest: {best_name}   mean CV {best_cv:.3f}   (runner-up {runner_cv:.3f})")
    print("\nSanity anchors for a mouse, if units are mm:")
    print("  ear-to-ear 10-30 | nose-to-tail-tip 120-260 | span/ear ratio 6-15")
    ok = (best_cv < 0.25 and 5 < best_stats["span/ear"] < 20)
    print(f"  -> {'PLAUSIBLE' if ok else 'STILL IMPLAUSIBLE — see notes below'}")

    # ---- STEP 2: frame convention, by reprojection
    rule("STEP 2 — WORLD-FRAME CONVENTION, by reprojection")
    print(f"Layout fixed to '{best_name}'. id3d is ground-aligned; cam_mat_all")
    print("projects from the camera-3 frame, so the ground transform must be undone.\n")
    print(f"{'convention':<32}{'on-image':>10}")
    print("-" * 42)
    pts0 = sample(best_arr)
    frames = frame_variants(R_g, t_g)
    fscored = []
    for label, fn in frames.items():
        sc = onimage(fn(pts0), mats, w, h)
        fscored.append((sc, label, fn))
        print(f"{label:<32}{sc:>9.1%}")

    fscored.sort(key=lambda r: -r[0])
    best_f, flabel, ffn = fscored[0]
    print(f"\nbest: {flabel}   {best_f:.1%}   (runner-up {fscored[1][0]:.1%})")

    test_subsampling(best_arr, mats, R_g, t_g, w, h, n_vid, ffn)

    # ---- verdict
    rule("VERDICT")
    print(f"column layout : {best_name}")
    print(f"projection    : cam_mat_all[:, :, k]")
    print(f"frame         : {flabel}")
    print(f"identities    : {names}")
    print(f"tracks shape  : ({best_arr.shape[0]}, {best_arr.shape[1]}, {best_arr.shape[2]}, 3)")
    if best_f > 0.9 and ok:
        print("\nBoth halves resolved. Ready to write the converter.")
    else:
        print("\nNOT resolved. Paste this report back — the numbers say which half failed:")
        print("  layout implausible  -> coords3d is not a plain 2x16x3 flatten")
        print("  frame below ~90%    -> cam_mat_all needs the ground transform composed")
        print("                         differently, or maps from raw3d not id3d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
