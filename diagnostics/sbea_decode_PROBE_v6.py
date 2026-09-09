#!/usr/bin/env python3
"""
v6 — Stop guessing the transform; measure it.

WHERE v5 LEFT IT
----------------
On rec11, every one of seven rigid variants scored at chance (best 5.5%,
two at exactly 0.0%). A wrong-but-related frame convention does not look
like that -- it looks like one variant clearly ahead. Chance across the whole
family means the reprojection is not landing on the mice at all, so the fault
is upstream of the ground transform. Adding an eighth guess is not a plan.

v5 did establish two things worth keeping:
  * layout animal/kp/xyz holds (diam/span 1.008, relMAD 0.074)
  * segmentations[frame] is a LIST OF 2 RLEs at 964x1288 -- per-animal
    silhouettes exist, in all four cameras, for all 27000 recording frames.

That second fact is what makes measurement possible. Two instance masks per
camera per frame is enough to triangulate each animal's centroid in the
CALIBRATION frame, with no reference to coords3d at all. Pair those against
the animal centroids coords3d reports in the STORED frame and the mapping
between the frames is a least-squares problem, not a search.

WHAT THIS PROBE ESTABLISHES, IN ORDER
-------------------------------------
1. CAMERA ORDER. Does cam_mat_all[:, :, k] belong to camera-k.avi? Tested
   without coords3d: under the right assignment the four rays through the
   four masks nearly intersect, and under a permuted one they do not. This
   is the single likeliest cause of a chance-level result, and it has never
   been checked.

2. THE TRANSFORM ITSELF. Umeyama (with scale) from the stored animal
   midpoints onto the triangulated ones. Scale is left free deliberately: if
   it comes back far from 1, coords3d is not in the calibration frame's units
   and no rotation-plus-translation guess could ever have worked.

3. WHETHER A RIGID MAP EXISTS AT ALL. The residual answers the suitability
   question directly. Small residual: the converter uses the recovered
   transform and the dataset is fine. Large residual under every camera
   permutation: coords3d is not rigidly related to these cameras, and
   multi-view image supervision from this release is impossible -- which is
   a finding about the release, not a bug to fix.

Run:
    python diagnostics/sbea_decode_PROBE_v6.py \
        "$HPCWORK/datasets/SBeA/fig2_data/pose tracking/rec11-A1A2-20220803" \
        --frames 80
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

N_KP, N_ANIMALS, N_DIM = 16, 2, 3


def rule(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def unwrap(o):
    while isinstance(o, np.ndarray) and o.dtype == object and o.size == 1:
        o = o.flat[0]
    return o


# ------------------------------------------------------------------ loading

def load_calibration(stem: Path):
    from scipy.io import loadmat
    md = loadmat(Path(f"{stem}-caliParas.mat"), struct_as_record=False, squeeze_me=False)
    root = next(unwrap(v) for k, v in md.items() if not k.startswith("__"))
    cam = np.asarray(root.cam_mat_all, dtype=np.float64)
    R_g = np.asarray(unwrap(root.rotation), dtype=np.float64).reshape(3, 3)
    t_g = np.asarray(unwrap(root.translation), dtype=np.float64).ravel()[:3]
    return cam, R_g, t_g


def load_coords(stem: Path):
    from scipy.io import loadmat
    md = loadmat(Path(f"{stem}-id3d.mat"), struct_as_record=False, squeeze_me=True)
    c = np.asarray(md["coords3d"], dtype=np.float64)
    return c.reshape(c.shape[0], N_ANIMALS, N_KP, N_DIM), [str(n) for n in np.atleast_1d(md.get("name3d"))]


def decode_rle_counts(s):
    if isinstance(s, bytes):
        s = s.decode("ascii")
    cnts, p, m = [], 0, 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1; k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x); m += 1
    return cnts


def rle_centroid(seg):
    """Centroid (u, v) and area of one RLE, without expanding the mask.
    Column-major runs: linear index i -> column i//h, row i%h."""
    if not isinstance(seg, dict) or "counts" not in seg:
        return None
    h, w = seg["size"]
    c = seg["counts"]
    runs = c if isinstance(c, list) else decode_rle_counts(c)
    pos, val, n, su, sv = 0, False, 0, 0.0, 0.0
    for r in runs:
        if val and r:
            idx = np.arange(pos, pos + r)
            su += float((idx // h).sum())
            sv += float((idx % h).sum())
            n += r
        pos += r
        val = not val
    if n == 0:
        return None
    return np.array([su / n, sv / n]), n


def load_instance_centroids(stem, cameras, frames):
    """cent[cam][frame] -> list of (uv, area), one per instance mask."""
    rule("1. PER-ANIMAL MASK CENTROIDS")
    cent = {}
    for cam in cameras:
        p = Path(f"{stem}-camera-{cam}-correctedresult.json")
        if not p.exists():
            print(f"  camera-{cam}: missing"); continue
        with open(p) as f:
            data = json.load(f)
        segs_lists = [a.get("segmentations", []) for a in data.get("annotations", [])]
        per_frame, counts = {}, []
        for fid in frames:
            found = []
            for segs in segs_lists:
                if fid >= len(segs):
                    continue
                e = segs[fid]
                for item in (e if isinstance(e, list) else [e]):
                    r = rle_centroid(item)
                    if r is not None:
                        found.append(r)
            per_frame[int(fid)] = found
            counts.append(len(found))
        cent[cam] = per_frame
        u = np.unique(counts, return_counts=True)
        print(f"  camera-{cam}: instances per frame {dict(zip(u[0].tolist(), u[1].tolist()))}")
    return cent


# ---------------------------------------------------------- triangulation

def triangulate(uvs, Ps):
    A = []
    for (u, v), P in zip(uvs, Ps):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return None, np.inf
    X = X[:3] / X[3]
    err = []
    for (u, v), P in zip(uvs, Ps):
        h = P @ np.append(X, 1.0)
        if abs(h[2]) < 1e-12:
            return None, np.inf
        err.append(np.hypot(h[0] / h[2] - u, h[1] / h[2] - v))
    return X, float(np.mean(err))


def pair_and_triangulate(frame_cents, Ps, cameras):
    """Both mice, with the cross-camera instance pairing chosen by residual.
    Returns (X_a, X_b, residual) in the calibration frame."""
    if any(len(frame_cents.get(c, [])) != 2 for c in cameras):
        return None
    ref = cameras[0]
    best = None
    for flips in itertools.product([0, 1], repeat=len(cameras) - 1):
        uvA = [frame_cents[ref][0][0]]
        uvB = [frame_cents[ref][1][0]]
        for c, fl in zip(cameras[1:], flips):
            uvA.append(frame_cents[c][fl][0])
            uvB.append(frame_cents[c][1 - fl][0])
        Xa, ea = triangulate(uvA, Ps)
        Xb, eb = triangulate(uvB, Ps)
        if Xa is None or Xb is None:
            continue
        tot = ea + eb
        if best is None or tot < best[2]:
            best = (Xa, Xb, tot / 2)
    return best


def camera_order_test(cent, cam_mat, cameras, frames):
    """Which permutation of cam_mat_all's slices belongs to the .avi cameras?
    Decided purely by ray intersection -- coords3d plays no part."""
    rule("2. CAMERA ORDER  (does cam_mat_all[:, :, k] belong to camera-k?)")
    print("Under the correct assignment the rays through the four masks nearly")
    print("intersect. Under a permuted one they cannot. coords3d is not used.\n")
    n = cam_mat.shape[2]
    usable = [f for f in frames if all(len(cent.get(c, {}).get(int(f), [])) == 2 for c in cameras)]
    if len(usable) < 5:
        print(f"!! only {len(usable)} frames have 2 instances in every camera.")
        return None, np.inf
    print(f"{len(usable)} usable frames\n")
    print(f"{'permutation':<20}{'median reproj err (px)':>24}")
    print("-" * 46)

    results = []
    for perm in itertools.permutations(range(n), len(cameras)):
        Ps = [cam_mat[:, :, k] for k in perm]
        errs = []
        for f in usable[:40]:
            r = pair_and_triangulate({c: cent[c][int(f)] for c in cameras}, Ps, cameras)
            if r is not None and np.isfinite(r[2]):
                errs.append(r[2])
        if errs:
            results.append((float(np.median(errs)), perm))
    results.sort()
    for err, perm in results[:6]:
        tag = "  <- identity" if list(perm) == list(range(len(cameras))) else ""
        print(f"{str(perm):<20}{err:>24.1f}{tag}")
    if len(results) > 6:
        ident = next((e for e, p in results if list(p) == list(range(len(cameras)))), None)
        if ident is not None and ident > results[5][0]:
            print(f"{'...':<20}")
            print(f"{str(tuple(range(len(cameras)))):<20}{ident:>24.1f}  <- identity")
    best_err, best_perm = results[0]
    print(f"\nbest: {best_perm}  at {best_err:.1f} px median reprojection error")
    if best_err > 25:
        print("!! no permutation gives a clean intersection. Either the masks")
        print("   are not per-animal, or cam_mat_all is not a set of 3x4 camera")
        print("   matrices for these four videos.")
    return best_perm, best_err


# ----------------------------------------------------------------- Umeyama

def umeyama(src, dst, with_scale=True):
    """Least-squares similarity src -> dst. Returns (s, R, t, rmse)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(C)
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = U @ W @ Vt
    s = (sig * np.diag(W)).sum() / (S ** 2).sum() * len(src) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    pred = (s * (R @ src.T)).T + t
    return s, R, t, float(np.sqrt(((pred - dst) ** 2).sum(1).mean()))


def recover_transform(arr, cent, cam_mat, cameras, perm, frames, R_g, t_g, names):
    rule("3. THE TRANSFORM, BY LEAST SQUARES")
    print("Animal midpoints are identity-free: no assumption about which mask")
    print("is A1. Scale is left free -- if it is not ~1, coords3d is not in the")
    print("calibration frame's units and no rigid guess could have worked.\n")

    Ps = [cam_mat[:, :, k] for k in perm]
    src, dst, seps_3d, seps_st = [], [], [], []
    for f in frames:
        fid = int(f)
        if fid >= arr.shape[0]:
            continue
        if any(len(cent.get(c, {}).get(fid, [])) != 2 for c in cameras):
            continue
        r = pair_and_triangulate({c: cent[c][fid] for c in cameras}, Ps, cameras)
        if r is None or not np.isfinite(r[2]) or r[2] > 50:
            continue
        Xa, Xb, _ = r
        stored = np.nanmean(arr[fid], axis=1)             # (2, 3) per-animal centroid
        if not np.isfinite(stored).all():
            continue
        src.append(stored.mean(0))
        dst.append((Xa + Xb) / 2)
        seps_3d.append(np.linalg.norm(Xa - Xb))
        seps_st.append(np.linalg.norm(stored[0] - stored[1]))

    print(f"{len(src)} usable correspondences")
    if len(src) < 10:
        print("!! too few to solve.")
        return None
    src, dst = np.asarray(src), np.asarray(dst)

    # Independent scale check: inter-animal separation is frame-invariant.
    ratio = np.median(np.asarray(seps_3d) / np.maximum(np.asarray(seps_st), 1e-9))
    print(f"inter-animal separation: triangulated median {np.median(seps_3d):.1f}, "
          f"stored median {np.median(seps_st):.1f}  ->  ratio {ratio:.3f}")

    for label, ws in (("with scale", True), ("rigid (s=1)", False)):
        s, R, t, rmse = umeyama(src, dst, ws)
        ang = np.degrees(np.arccos(np.clip((np.trace(R @ R_g.T) - 1) / 2, -1, 1)))
        print(f"\n{label}:")
        print(f"  scale {s:.4f}   rmse {rmse:.2f} (units of the calibration frame)")
        print(f"  |t| {np.linalg.norm(t):.1f}   vs caliParas |t| {np.linalg.norm(t_g):.1f}")
        print(f"  rotation differs from caliParas rotation by {ang:.1f} deg")
        if ws:
            keep = (s, R, t, rmse)

    s, R, t, rmse = keep
    spread = float(np.linalg.norm(dst - dst.mean(0), axis=1).mean())
    print(f"\nresidual {rmse:.2f} against a point cloud of mean radius {spread:.1f}")
    if rmse < 0.15 * spread:
        print("-> A similarity transform DOES relate coords3d to these cameras.")
        print("   Use it in the converter; caliParas' own R/t are not the map.")
    else:
        print("-> NO similarity transform fits. coords3d is not rigidly related")
        print("   to this calibration, so these videos cannot supervise this 3D.")
    print(f"\nidentities {names}")
    return s, R, t, rmse


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--cameras", default="0,1,2,3")
    ap.add_argument("--frames", type=int, default=80)
    args = ap.parse_args()

    stem = Path(args.stem)
    if not Path(f"{stem}-id3d.mat").exists():
        print(f"ERROR: {stem}-id3d.mat not found.", file=sys.stderr); return 1
    cameras = [int(c) for c in args.cameras.split(",")]

    cam_mat, R_g, t_g = load_calibration(stem)
    arr, names = load_coords(stem)
    rule("SESSION")
    print(f"coords3d {arr.shape}   identities {names}")
    print(f"cam_mat_all {cam_mat.shape}   caliParas |t| {np.linalg.norm(t_g):.1f}")

    frames = np.linspace(0, arr.shape[0] - 1, args.frames + 2).astype(int)[1:-1]
    cent = load_instance_centroids(stem, cameras, frames)
    if not cent:
        print("!! no masks."); return 2

    perm, err = camera_order_test(cent, cam_mat, cameras, frames)
    if perm is None:
        return 3
    recover_transform(arr, cent, cam_mat, cameras, perm, frames, R_g, t_g, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
