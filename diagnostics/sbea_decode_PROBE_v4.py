#!/usr/bin/env python3
"""
v4 — Settle the two remaining unknowns in the SBeA -> SMILify conversion:
the world-frame convention, and which rows of coords3d the released 900-frame
video excerpt corresponds to.

WHY v3's CONCLUSION WAS WRONG
-----------------------------
v3 ranked the six reshape layouts by mean coefficient of variation of rigid
pair lengths and picked "xyz / kp / animal". That layout reports an
ear-to-ear distance of 701 units and a nose-to-tail-tip span of 102 units --
a mouse whose head is seven times wider than its body is long. CV is a ratio
of std to mean, so a layout that scrambles coordinates into a large, nearly
constant offset scores *better* than the truth. The ranking rule rewarded
exactly the wrong thing.

v3's own table carried the answer. The discriminating statistic is not CV but
CLOSURE: for a single animal the largest pairwise keypoint distance IS the
nose-to-tail-tip distance, so diameter/span must equal 1. Mixing two animals
into one "animal" slot inflates the diameter without touching the span.

    layout            diam/span   span/ear   verdict
    animal/kp/xyz         1.008       7.57   <- closes, and all anchors in range
    kp/animal/xyz         1.329      12.25   diameter exceeds the span: mixed
    kp/xyz/animal         1.332      10.55   mixed
    xyz/kp/animal         8.691       0.15   absurd
    animal/xyz/kp        24.185       0.05   absurd
    xyz/animal/kp        26.721       0.05   absurd

animal/kp/xyz is a plain C-order reshape to (F, 2, 16, 3), in millimetres
(ear-ear 18.8 mm, nose-to-tail-tip 142.3 mm, cameras 714-1002 mm out).
The residual CV of 0.29 is triangulation outliers, not a wrong layout --
std/mean is not robust; this probe reports median and MAD instead.

Because v3 ran STEP 2 on the scrambled layout, its 57.5% reprojection score
and its whole row-offset table are meaningless and are recomputed here.

WHAT v4 DOES DIFFERENTLY
------------------------
The fig2 sessions ship no per-animal 2D keypoints, so reprojection cannot be
scored against 2D ground truth. It can be scored against the VIS instance
masks in "*-correctedresult.json", which ARE per-frame and per-camera: under
the correct frame convention and the correct row alignment, a reprojected
keypoint lands inside a mouse mask. That single objective resolves both
unknowns at once, and it is searched jointly here.

Run:
    conda activate "$HPCWORK/conda_envs/pytorch3d"
    python diagnostics/sbea_decode_PROBE_v4.py \
        "$HPCWORK/datasets/SBeA/fig2_data/pose tracking/rec11-A1A2-20220803" \
        --out-dir diagnostics/sbea_v4_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_KP, N_ANIMALS, N_DIM = 16, 2, 3

# DLC bodypart order, confirmed against the gt_data CSV header.
NOSE, EAR_L, EAR_R, NECK = 0, 1, 2, 3
BACK, ROOT_TAIL, MID_TAIL, TIP_TAIL = 12, 13, 14, 15

RIGID_PAIRS = [
    ("ear_L-ear_R", EAR_L, EAR_R),
    ("nose-neck", NOSE, NECK),
    ("root-mid tail", ROOT_TAIL, MID_TAIL),
    ("mid-tip tail", MID_TAIL, TIP_TAIL),
]


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
    cam_mat = np.asarray(root.cam_mat_all, dtype=np.float64)
    R_g = np.asarray(unwrap(root.rotation), dtype=np.float64).reshape(3, 3)
    t_g = np.asarray(unwrap(root.translation), dtype=np.float64).ravel()[:3]
    return [cam_mat[:, :, k] for k in range(cam_mat.shape[2])], R_g, t_g


def load_coords(stem: Path):
    from scipy.io import loadmat

    md = loadmat(Path(f"{stem}-id3d.mat"), struct_as_record=False, squeeze_me=True)
    coords = np.asarray(md["coords3d"], dtype=np.float64)
    names = [str(n) for n in np.atleast_1d(md.get("name3d"))]
    return coords, names


# ------------------------------------------------- STEP 1: layout, by closure

def robust_stats(arr: np.ndarray, n_frames: int = 800) -> dict:
    """Median and MAD-based spread. Unlike std/mean, unaffected by the handful
    of badly triangulated frames that inflated v3's CV."""
    idx = np.linspace(0, arr.shape[0] - 1, min(n_frames, arr.shape[0])).astype(int)
    sub = arr[idx]
    out: dict = {}

    spreads = []
    for label, i, j in RIGID_PAIRS:
        d = np.linalg.norm(sub[:, :, i] - sub[:, :, j], axis=-1).ravel()
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) < 10:
            out[label] = (np.nan, np.nan)
            continue
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        rel = mad / max(med, 1e-9)
        out[label] = (med, rel)
        spreads.append(rel)

    span = np.linalg.norm(sub[:, :, NOSE] - sub[:, :, TIP_TAIL], axis=-1).ravel()
    span = span[np.isfinite(span)]
    out["span"] = float(np.median(span)) if len(span) else np.nan

    d = np.linalg.norm(sub[:, :, :, None, :] - sub[:, :, None, :, :], axis=-1)
    out["diameter"] = float(np.median(np.nanmax(d, axis=(2, 3))))
    out["mean_rel_mad"] = float(np.mean(spreads)) if spreads else np.inf
    out["closure"] = out["diameter"] / out["span"] if out["span"] else np.inf
    ear = out.get("ear_L-ear_R", (np.nan, np.nan))[0]
    out["span_over_ear"] = out["span"] / ear if ear else np.nan
    return out


def all_layouts(coords: np.ndarray) -> dict:
    import itertools

    f = coords.shape[0]
    role = {N_ANIMALS: "animal", N_KP: "kp", N_DIM: "xyz"}
    out = {}
    for perm in itertools.permutations((N_ANIMALS, N_KP, N_DIM)):
        arr = coords.reshape(f, *perm)
        ax = {perm[i]: i + 1 for i in range(3)}
        out[" / ".join(role[p] for p in perm)] = np.transpose(arr, (0, ax[N_ANIMALS], ax[N_KP], ax[N_DIM]))
    return out


def choose_layout(coords: np.ndarray):
    rule("STEP 1 - COLUMN LAYOUT, by geometric closure")
    print("For ONE animal the largest pairwise keypoint distance is the")
    print("nose-to-tail-tip distance, so diameter/span == 1. A layout that mixes")
    print("the two mice inflates the diameter and leaves the span alone.\n")
    print(f"{'layout':<22}{'ear-ear':>9}{'relMAD':>9}{'span':>9}{'diam':>9}"
          f"{'diam/span':>11}{'span/ear':>10}")
    print("-" * 79)

    scored = []
    for name, arr in all_layouts(coords).items():
        s = robust_stats(arr)
        ear, ear_mad = s["ear_L-ear_R"]
        # Plausibility gate: mouse anthropometry, in millimetres.
        plausible = (8 < ear < 40) and (100 < s["span"] < 300) and (5 < s["span_over_ear"] < 20)
        score = abs(s["closure"] - 1.0) + (0.0 if plausible else 10.0)
        scored.append((score, name, arr, s, plausible))
        print(f"{name:<22}{ear:>9.1f}{ear_mad:>9.3f}{s['span']:>9.1f}"
              f"{s['diameter']:>9.1f}{s['closure']:>11.3f}{s['span_over_ear']:>10.2f}"
              f"{'' if plausible else '   (implausible)'}")

    scored.sort(key=lambda r: r[0])
    score, name, arr, s, plausible = scored[0]
    print(f"\nbest: {name}   |diam/span - 1| = {abs(s['closure'] - 1):.3f}"
          f"   (runner-up {scored[1][1]}, {abs(scored[1][3]['closure'] - 1):.3f})")
    print(f"units look like millimetres: ear-ear {s['ear_L-ear_R'][0]:.1f}, "
          f"nose-to-tail-tip {s['span']:.1f}")
    if not plausible or abs(s["closure"] - 1) > 0.15:
        print("\n!! No layout closes. coords3d is not a plain 2x16x3 flatten -- stop here.")
        return None, None
    return name, arr


# ------------------------------------------------------------- mask decoding

def decode_rle_counts(s):
    """COCO compressed-RLE 'counts' -> run lengths. Pure-python fallback so the
    probe runs without pycocotools."""
    if isinstance(s, bytes):
        s = s.decode("ascii")
    cnts, p, m = [], 0, 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += cnts[m - 2]
        cnts.append(x)
        m += 1
    return cnts


def rle_to_mask(seg) -> np.ndarray | None:
    if seg is None:
        return None
    h, w = seg["size"]
    counts = seg["counts"]
    if isinstance(counts, list):
        runs = counts
    else:
        runs = decode_rle_counts(counts)
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for r in runs:
        if val and r:
            flat[pos:pos + r] = True
        pos += r
        val = not val
    return flat.reshape((h, w), order="F")


def load_masks(stem: Path, cameras, frame_ids):
    """masks[cam][frame_id] -> bool array, or None where the track is absent."""
    rule("MASKS")
    masks = {}
    for cam in cameras:
        p = Path(f"{stem}-camera-{cam}-correctedresult.json")
        if not p.exists():
            print(f"  camera-{cam}: no correctedresult.json")
            continue
        with open(p) as f:
            data = json.load(f)
        anns = data.get("annotations", [])
        segs_per_ann = [a.get("segmentations", []) for a in anns]
        n_frames = max((len(s) for s in segs_per_ann), default=0)
        print(f"  camera-{cam}: {len(anns)} instance track(s), {n_frames} frames")
        per_frame = {}
        for fid in frame_ids:
            acc = None
            for segs in segs_per_ann:
                if fid >= len(segs):
                    continue
                m = rle_to_mask(segs[fid])
                if m is None:
                    continue
                acc = m if acc is None else (acc | m)
            per_frame[fid] = acc
        masks[cam] = per_frame
    return masks


# --------------------------------------------------- STEP 2: joint frame/offset

def frame_variants(R_g, t_g):
    Rt = R_g.T
    return {
        "as stored (no transform)": lambda X: X,
        "R^T (X - t)  [undo ground]": lambda X: (X - t_g) @ Rt.T,
        "R X + t      [apply ground]": lambda X: X @ R_g.T + t_g,
        "R^T X        [rotation only]": lambda X: X @ Rt.T,
        "R (X - t)": lambda X: (X - t_g) @ R_g.T,
        "R^T X - t": lambda X: X @ Rt.T - t_g,
        "R (X + t)": lambda X: (X + t_g) @ R_g.T,
    }


def project(P, pts):
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = hom @ P.T
    w = cam[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = cam[:, :2] / w[:, None]
    return uv, w


def row_scores(arr_flat, P, mask, W, H, fn):
    """Score EVERY row of coords3d at once against one camera/frame mask.

    Returns (n_rows,) = fraction of that row's keypoints landing inside the
    mask. Vectorised because the offset search below is exhaustive: a grid
    search can step straight over the answer, since the objective has support
    only within a few frames of the true alignment.
    """
    R, J, _ = arr_flat.shape
    pts = fn(arr_flat.reshape(-1, 3))
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = hom @ P.T
    depth = cam[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = cam[:, :2] / depth[:, None]

    # Sign convention is not guaranteed: take whichever depth sign the bulk of
    # the reconstruction falls on rather than assuming z > 0.
    finite = np.isfinite(depth)
    sgn = np.sign(np.median(depth[finite])) if finite.any() else 1.0
    ok = np.isfinite(uv).all(axis=1) & (depth * sgn > 0)
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if mask is None:
        hit = ok
    else:
        u = np.clip(uv[:, 0], 0, W - 1).astype(np.int32)
        v = np.clip(uv[:, 1], 0, H - 1).astype(np.int32)
        hit = ok & mask[v, u]
    return hit.reshape(R, J).mean(axis=1)


def align_scores(tables, sample_frames, stride, n_rows):
    """Combine per-(frame, camera) row-score tables into a score per offset."""
    span = int(sample_frames[-1]) * stride
    n_off = n_rows - span
    if n_off <= 0:
        return np.zeros(0)
    total = np.zeros(n_off)
    for (f, _cam), s in tables.items():
        shift = int(f) * stride
        total += s[shift:shift + n_off]
    return total / max(len(tables), 1)


def search(arr, mats, masks, cameras, R_g, t_g, n_video, W, H):
    rule("STEP 2 - FRAME CONVENTION x ROW ALIGNMENT, scored against VIS masks")
    print("A reprojected keypoint must land inside a mouse mask. Both unknowns")
    print("are searched together: a wrong offset also breaks a correct frame.")
    print("Every offset is tested, not a grid -- the objective is sharp, so a")
    print("grid would step over the answer. Chance is the mask area fraction,")
    print("a few percent, so a real alignment is unmistakable.\n")

    n_rows = arr.shape[0]
    arr_flat = np.ascontiguousarray(arr.reshape(n_rows, -1, 3))
    arr_flat = np.nan_to_num(arr_flat, nan=-1e9)
    variants = frame_variants(R_g, t_g)
    cams = [c for c in cameras if c in masks and c in mats] or [c for c in cameras if c in mats]
    stride_dec = max(1, n_rows // n_video)

    families = [("contiguous", 1)]
    if stride_dec > 1:
        families.append((f"decimated x{stride_dec}", stride_dec))

    # Phase A -- pick the convention cheaply: 3 frames, 2 cameras.
    probe_frames = np.linspace(0, n_video - 1, 3).astype(int)
    probe_cams = cams[:2]
    print(f"{'convention':<32}{'family':<16}{'best':>8}{'offset':>9}")
    print("-" * 65)
    ranked = []
    for label, fn in variants.items():
        for fam, st in families:
            fmax = probe_frames[-1] * st
            if fmax >= n_rows:
                continue
            tables = {(f, c): row_scores(arr_flat, mats[c], masks.get(c, {}).get(f), W, H, fn)
                      for f in probe_frames for c in probe_cams}
            sc = align_scores(tables, probe_frames, st, n_rows)
            if len(sc) == 0:
                continue
            o = int(np.argmax(sc))
            ranked.append((float(sc[o]), label, fam, st, o))
            print(f"{label:<32}{fam:<16}{sc[o]:>7.1%}{o:>9}")
    if not ranked:
        return None, 0, 1, 0.0
    ranked.sort(reverse=True)
    _, flabel, fam, st, _ = ranked[0]
    print(f"\nconvention -> {flabel}   family -> {fam}")

    # Phase B -- confirm and localise with the full frame/camera set.
    frames = np.linspace(0, n_video - 1, 12).astype(int)
    fn = variants[flabel]
    tables = {(f, c): row_scores(arr_flat, mats[c], masks.get(c, {}).get(f), W, H, fn)
              for f in frames for c in cams}
    sc = align_scores(tables, frames, st, n_rows)
    peak = float(sc.max())
    # Neighbouring offsets tie whenever the animals move less than a mask width
    # between frames, so report the plateau and take its centre rather than
    # letting argmax pick an arbitrary member of it.
    plateau = np.flatnonzero(sc >= 0.98 * peak)
    contiguous_run = plateau[(plateau >= plateau[0]) & (plateau <= plateau[0] + 200)]
    lo, hi = int(contiguous_run.min()), int(contiguous_run.max())
    off = (lo + hi) // 2
    score = float(sc[off])
    outside = sc.copy()
    outside[max(0, lo - 10):hi + 11] = -1
    rival = int(np.argmax(outside))
    print(f"peak {peak:.1%} over offsets {lo}..{hi}  ->  taking {off} ({score:.1%})")
    print(f"best unrelated offset {rival}: {sc[rival]:.1%}")
    if hi - lo > 30:
        print("!! wide plateau: the alignment is only determined to +-"
              f"{(hi - lo) // 2} frames. Settle it on the overlays.")
    return flabel, off, st, score


# ---------------------------------------------------------------- overlays

def write_overlays(stem, arr, mats, cameras, fn, off, st, out_dir, n_video):
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    rule("OVERLAYS")
    for fid in np.linspace(0, n_video - 1, 4).astype(int):
        row = off + int(fid) * st
        pts = arr[row]
        for cam in cameras:
            vp = Path(f"{stem}-camera-{cam}.avi")
            if not vp.exists():
                continue
            cap = cv2.VideoCapture(str(vp))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, img = cap.read()
            cap.release()
            if not ok:
                continue
            for a, colour in enumerate([(0, 200, 255), (255, 120, 0)]):
                p = pts[a][np.isfinite(pts[a]).all(axis=1)]
                if len(p) == 0:
                    continue
                uv, depth = project(mats[cam], fn(p))
                fin = np.isfinite(depth)
                sgn = np.sign(np.median(depth[fin])) if fin.any() else 1.0
                for (u, v), d in zip(uv, depth):
                    if not np.isfinite(u) or d * sgn <= 0:
                        continue
                    cv2.circle(img, (int(u), int(v)), 3, colour, -1)
            dst = out_dir / f"overlay_f{int(fid):04d}_cam{cam}.png"
            cv2.imwrite(str(dst), img)
            print(f"  wrote {dst}")
    print("\nOpen these. If the dots sit on the mice, the conversion is settled.")


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="session stem, e.g. .../rec11-A1A2-20220803")
    ap.add_argument("--out-dir", default="diagnostics/sbea_v4_out")
    ap.add_argument("--cameras", default="0,1,2,3")
    args = ap.parse_args()

    stem = Path(args.stem)
    if not Path(f"{stem}-id3d.mat").exists():
        print(f"ERROR: {stem}-id3d.mat not found.", file=sys.stderr)
        return 1
    cameras = [int(c) for c in args.cameras.split(",")]

    mats_list, R_g, t_g = load_calibration(stem)
    mats = {c: mats_list[c] for c in cameras if c < len(mats_list)}
    coords, names = load_coords(stem)

    rule("SESSION")
    print(f"coords3d {coords.shape}   identities {names}")
    print(f"det(R) = {np.linalg.det(R_g):.6f}   t = {np.array2string(t_g, precision=2)}")

    import cv2

    cap = cv2.VideoCapture(f"{stem}-camera-{cameras[0]}.avi")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"video {W}x{H}, {n_video} frames; coords3d has {coords.shape[0]} rows")

    name, arr = choose_layout(coords)
    if arr is None:
        return 2

    # Union of the frames both search phases will ask for.
    sample_frames = sorted(set(np.linspace(0, n_video - 1, 12).astype(int).tolist())
                           | set(np.linspace(0, n_video - 1, 3).astype(int).tolist()))
    masks = load_masks(stem, cameras, sample_frames)
    if not masks:
        print("\n!! No masks -- cannot score. Falling back to on-image only.")

    flabel, off, st, score = search(arr, mats, masks, cameras, R_g, t_g, n_video, W, H)

    write_overlays(stem, arr, mats, cameras, frame_variants(R_g, t_g)[flabel],
                   off, st, Path(args.out_dir), n_video)

    rule("VERDICT")
    print(f"column layout : {name}   -> tracks ({coords.shape[0]}, 2, 16, 3)")
    print(f"frame         : {flabel}")
    print(f"row alignment : video frame f  ->  coords3d row {off} + {st}*f")
    print(f"mask agreement: {score:.1%}")
    print(f"identities    : {names}")
    if score > 0.75:
        print("\nRESOLVED. The converter can be written against these three facts.")
    else:
        print("\nNOT resolved. Check the overlays: if the dots form a mouse-shaped")
        print("cloud in the wrong place, the frame convention is the problem; if")
        print("they are scattered, the layout or the calibration indexing is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
