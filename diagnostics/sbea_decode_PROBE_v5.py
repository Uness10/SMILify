#!/usr/bin/env python3
"""
v5 — Same goal as v4, restructured around what v4 discovered on the real data.

WHAT v4 FOUND, AND WHY IT CHANGES THE SEARCH
--------------------------------------------
v4 confirmed the layout on the real session: animal/kp/xyz closes at
diam/span = 1.008, ear-ear 18.8 mm, nose-to-tail-tip 142.0 mm, relMAD 0.079.
That question is settled and v5 keeps the test only as a guard.

v4 then crashed, and the crash was informative twice over:

  1. segmentations[frame] is a LIST, not an RLE dict. v5 decodes whatever is
     actually there -- compressed RLE, uncompressed RLE, a list of either, or
     COCO polygons -- and prints the format it found before using it.

  2. camera-0 reported 27000 mask frames, not 900. The VIS masks are indexed
     by RECORDING frame, exactly like coords3d's 27000 rows, while the
     released .avi is a 900-frame excerpt. So there are two independent
     alignment questions, and v4 conflated them:

        (A) mask frame  <-> coords3d row   -- expected identity, small window
        (B) .avi  frame <-> coords3d row   -- genuinely unknown, whole range

v5 answers them separately. (A) needs no offset search worth the name, so it
isolates the frame convention cleanly. (B) then runs with the convention
already fixed, and is scored against the .avi's own background-subtracted
foreground rather than the VIS masks, because the masks cannot say which
recording frames the excerpt covers.

(B) is what decides whether the images can be paired with the 3D at all, and
therefore whether this dataset can train anything.

Run:
    python diagnostics/sbea_decode_PROBE_v5.py \
        "$HPCWORK/datasets/SBeA/fig2_data/pose tracking/rec11-A1A2-20220803" \
        --out-dir diagnostics/sbea_v5_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_KP, N_ANIMALS, N_DIM = 16, 2, 3
NOSE, EAR_L, EAR_R, NECK = 0, 1, 2, 3
BACK, ROOT_TAIL, MID_TAIL, TIP_TAIL = 12, 13, 14, 15
RIGID_PAIRS = [("ear_L-ear_R", EAR_L, EAR_R), ("nose-neck", NOSE, NECK),
               ("root-mid tail", ROOT_TAIL, MID_TAIL), ("mid-tip tail", MID_TAIL, TIP_TAIL)]


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
    return np.asarray(md["coords3d"], dtype=np.float64), [str(n) for n in np.atleast_1d(md.get("name3d"))]


# -------------------------------------------------------- layout (guard only)

def robust_stats(arr, n_frames=800):
    idx = np.linspace(0, arr.shape[0] - 1, min(n_frames, arr.shape[0])).astype(int)
    sub, out, spreads = arr[idx], {}, []
    for label, i, j in RIGID_PAIRS:
        d = np.linalg.norm(sub[:, :, i] - sub[:, :, j], axis=-1).ravel()
        d = d[np.isfinite(d) & (d > 0)]
        if len(d) < 10:
            out[label] = (np.nan, np.nan); continue
        med = float(np.median(d)); rel = float(np.median(np.abs(d - med))) / max(med, 1e-9)
        out[label] = (med, rel); spreads.append(rel)
    span = np.linalg.norm(sub[:, :, NOSE] - sub[:, :, TIP_TAIL], axis=-1).ravel()
    span = span[np.isfinite(span)]
    out["span"] = float(np.median(span)) if len(span) else np.nan
    d = np.linalg.norm(sub[:, :, :, None, :] - sub[:, :, None, :, :], axis=-1)
    out["diameter"] = float(np.median(np.nanmax(d, axis=(2, 3))))
    out["closure"] = out["diameter"] / out["span"] if out["span"] else np.inf
    ear = out.get("ear_L-ear_R", (np.nan, np.nan))[0]
    out["span_over_ear"] = out["span"] / ear if ear else np.nan
    out["mean_rel_mad"] = float(np.mean(spreads)) if spreads else np.inf
    return out


def check_layout(coords):
    """v4 settled this on the real data; re-run as a guard, not a search."""
    rule("STEP 1 - LAYOUT GUARD (animal/kp/xyz, established in v4)")
    arr = coords.reshape(coords.shape[0], N_ANIMALS, N_KP, N_DIM)
    s = robust_stats(arr)
    print(f"ear-ear {s['ear_L-ear_R'][0]:.1f} mm   nose-to-tail-tip {s['span']:.1f} mm   "
          f"diam/span {s['closure']:.3f}   relMAD {s['mean_rel_mad']:.3f}")
    if abs(s["closure"] - 1) > 0.15:
        print("!! closure broken -- this session does not match the established layout.")
        return None
    print("layout holds.")
    return arr


# --------------------------------------------------------- segmentation decode

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


def describe_seg(seg, depth=0):
    """One-line description of whatever shape a segmentation entry has."""
    pad = "  " * depth
    if seg is None:
        return f"{pad}None"
    if isinstance(seg, dict):
        c = seg.get("counts")
        kind = "list-RLE" if isinstance(c, list) else f"str-RLE({len(c) if c else 0} chars)"
        return f"{pad}dict keys={sorted(seg.keys())} size={seg.get('size')} counts={kind}"
    if isinstance(seg, list):
        if not seg:
            return f"{pad}[] (empty)"
        head = seg[0]
        if isinstance(head, (int, float)):
            return f"{pad}list of {len(seg)} numbers (polygon: {len(seg) // 2} points)"
        return f"{pad}list of {len(seg)}:\n" + describe_seg(head, depth + 1)
    return f"{pad}{type(seg).__name__}"


def seg_to_mask(seg, H, W):
    """Decode any of the shapes COCO/YouTubeVIS actually ships. Returns bool
    (H, W), or None when the entry carries no instance."""
    if seg is None:
        return None

    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        h, w = seg["size"]
        c = seg["counts"]
        runs = c if isinstance(c, list) else decode_rle_counts(c)
        flat = np.zeros(h * w, dtype=bool)
        pos, val = 0, False
        for r in runs:
            if val and r:
                flat[pos:pos + r] = True
            pos += r; val = not val
        return flat.reshape((h, w), order="F")

    if isinstance(seg, list):
        if not seg:
            return None
        if isinstance(seg[0], (int, float)):          # single polygon, flat
            import cv2
            pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2).astype(np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            return m.astype(bool)
        acc = None                                     # list of polygons / RLEs
        for sub in seg:
            m = seg_to_mask(sub, H, W)
            if m is None:
                continue
            acc = m if acc is None else (acc | m)
        return acc
    return None


def load_masks(stem, cameras, frame_ids, H, W):
    """masks[cam][frame_id] -> bool (H, W). Reports the on-disk format first."""
    rule("MASKS")
    masks, n_mask_frames = {}, None
    for cam in cameras:
        p = Path(f"{stem}-camera-{cam}-correctedresult.json")
        if not p.exists():
            print(f"  camera-{cam}: no correctedresult.json"); continue
        with open(p) as f:
            data = json.load(f)
        anns = data.get("annotations", [])
        segs_per_ann = [a.get("segmentations", []) for a in anns]
        n = max((len(s) for s in segs_per_ann), default=0)
        n_mask_frames = n if n_mask_frames is None else max(n_mask_frames, n)
        print(f"  camera-{cam}: {len(anns)} track(s), {n} frames")
        if cam == cameras[0] and segs_per_ann:
            for segs in segs_per_ann[:2]:
                ex = next((s for s in segs if s not in (None, [])), None)
                print("    first non-empty entry ->")
                print("    " + describe_seg(ex).replace("\n", "\n    "))
        per_frame = {}
        for fid in frame_ids:
            acc = None
            for segs in segs_per_ann:
                if fid >= len(segs):
                    continue
                m = seg_to_mask(segs[fid], H, W)
                if m is None:
                    continue
                if m.shape != (H, W):
                    continue
                acc = m if acc is None else (acc | m)
            per_frame[int(fid)] = acc
        masks[cam] = per_frame
    return masks, n_mask_frames


# ---------------------------------------------------------------- projection

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
    depth = cam[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = cam[:, :2] / depth[:, None]
    return uv, depth


def row_scores(arr_flat, P, mask, W, H, fn):
    """Fraction of each row's keypoints landing inside `mask`, for every row."""
    R, J, _ = arr_flat.shape
    pts = fn(arr_flat.reshape(-1, 3))
    uv, depth = project(P, pts)
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


# ------------------------------- (A) frame convention, masks are row-indexed

def solve_convention(arr_flat, mats, masks, cameras, R_g, t_g, n_rows, W, H, window=300):
    rule("STEP 2 (A) - FRAME CONVENTION   [masks are recording-indexed]")
    print("The VIS masks carry one entry per recording frame, the same index")
    print("coords3d rows use, so mask frame f should be row f. Only a small")
    print("delta is searched, to confirm that rather than assume it. With the")
    print("alignment near-fixed, the convention is isolated.\n")

    frames = [f for f in sorted(masks.get(cameras[0], {})) if masks[cameras[0]][f] is not None]
    if not frames:
        print("!! no decodable masks -- cannot run (A).")
        return None, 0, 0.0
    cams = [c for c in cameras if c in masks and c in mats]
    deltas = np.arange(-window, window + 1)
    print(f"{'convention':<32}{'best':>8}{'delta':>8}")
    print("-" * 48)

    ranked = []
    for label, fn in frame_variants(R_g, t_g).items():
        tables = {(f, c): row_scores(arr_flat, mats[c], masks[c].get(f), W, H, fn)
                  for f in frames for c in cams}
        curve = {}
        for d in deltas:
            rows = np.asarray(frames) + d
            if rows.min() < 0 or rows.max() >= n_rows:
                continue
            curve[int(d)] = float(np.mean([tables[(f, c)][f + d] for f in frames for c in cams]))
        if not curve:
            ranked.append((0.0, label, 0)); print(f"{label:<32}{'n/a':>8}"); continue
        top = max(curve.values())
        # Neighbouring deltas tie whenever the mice move less than a mask width
        # per frame. Identity is the hypothesis under test, so among tied
        # deltas report the one nearest zero rather than an arbitrary one.
        best_d = min((d for d, v in curve.items() if v >= 0.999 * top), key=abs)
        best = (curve[best_d], best_d)
        ranked.append((best[0], label, best[1]))
        print(f"{label:<32}{best[0]:>7.1%}{best[1]:>8}")

    ranked.sort(reverse=True)
    score, flabel, delta = ranked[0]
    print(f"\nbest: {flabel}   {score:.1%}   at delta {delta}   "
          f"(runner-up {ranked[1][1]}, {ranked[1][0]:.1%})")
    if score < 0.6:
        print("!! weak. Either cam_mat_all indexes cameras differently, or the")
        print("   masks are not row-aligned after all. Check the overlays.")
    return flabel, delta, score


# --------------------------------- (B) which rows the released .avi covers

def video_foreground(path, sample_frames, n_bg=40):
    """Background-subtracted foreground per sampled frame. The VIS masks cannot
    answer (B) -- they index the recording, not the excerpt -- so the excerpt
    has to supply its own silhouettes."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_idx = np.linspace(0, n - 1, min(n_bg, n)).astype(int)
    stack = []
    for i in bg_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            stack.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if not stack:
        cap.release(); return {}, n
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)

    out = {}
    k = np.ones((9, 9), np.uint8)
    for f in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fg = cv2.absdiff(g, bg) > 25
        out[int(f)] = cv2.dilate(fg.astype(np.uint8), k).astype(bool)
    cap.release()
    return out, n


def solve_excerpt(stem, arr_flat, mats, cameras, fn, n_rows, n_video, W, H):
    rule("STEP 2 (B) - WHICH ROWS THE 900-FRAME .avi COVERS")
    print("Exhaustive over every candidate, not a grid: the objective has")
    print("support only within a few frames of the truth. Two families are")
    print("tested -- a contiguous excerpt, and the recording decimated to the")
    print("excerpt's length.\n")

    frames = np.linspace(0, n_video - 1, 10).astype(int)
    fgs, cams = {}, []
    for c in cameras:
        p = Path(f"{stem}-camera-{c}.avi")
        if not p.exists() or c not in mats:
            continue
        fg, _ = video_foreground(p, frames)
        if fg:
            fgs[c] = fg; cams.append(c)
    if not cams:
        print("!! no readable video."); return 0, 1, 0.0
    print(f"foreground extracted for cameras {cams}")

    stride_dec = max(1, n_rows // n_video)
    families = [("contiguous", 1)] + ([(f"decimated x{stride_dec}", stride_dec)]
                                      if stride_dec > 1 else [])
    results = []
    for fam, st in families:
        tables = {(f, c): row_scores(arr_flat, mats[c], fgs[c].get(int(f)), W, H, fn)
                  for f in frames for c in cams}
        span = int(frames[-1]) * st
        n_off = n_rows - span
        if n_off <= 0:
            continue
        total = np.zeros(n_off)
        for (f, _c), s in tables.items():
            total += s[int(f) * st: int(f) * st + n_off]
        total /= len(tables)
        peak = float(total.max())
        plat = np.flatnonzero(total >= 0.98 * peak)
        lo, hi = int(plat.min()), int(plat[plat <= plat.min() + 200].max())
        off = (lo + hi) // 2
        rival = total.copy(); rival[max(0, lo - 20):hi + 21] = -1
        rv = float(rival.max()) if (rival >= 0).any() else float("nan")
        results.append((peak, fam, st, off, lo, hi, rv))
        shown = "n/a (no unrelated candidates)" if np.isnan(rv) else f"{rv:.1%}"
        print(f"  {fam:<16} peak {peak:6.1%} at offsets {lo}..{hi} -> {off}"
              f"   next unrelated {shown}")

    if not results:
        return 0, 1, 0.0
    results.sort(reverse=True)
    peak, fam, st, off, lo, hi, rival = results[0]
    sep = 0.0 if np.isnan(rival) else peak - rival
    print(f"\nbest: {fam}, row = {off} + {st}*f   ({peak:.1%}, "
          f"{'no rival' if np.isnan(rival) else f'rival {rival:.1%}'})")
    if peak < 0.5 or (not np.isnan(rival) and sep < 0.1):
        print("!! not separated. The excerpt may not come from this recording,")
        print("   or the foreground threshold needs tuning. Check the overlays.")
    if hi - lo > 30:
        print(f"!! plateau {hi - lo} frames wide: alignment only good to +-{(hi - lo) // 2}.")
    return off, st, peak


# ---------------------------------------------------------------- overlays

def write_overlays(stem, arr, mats, cameras, fn, off, st, out_dir, n_video):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    rule("OVERLAYS")
    for fid in np.linspace(0, n_video - 1, 4).astype(int):
        row = off + int(fid) * st
        if row < 0 or row >= arr.shape[0]:
            continue
        for cam in cameras:
            vp = Path(f"{stem}-camera-{cam}.avi")
            if not vp.exists() or cam not in mats:
                continue
            cap = cv2.VideoCapture(str(vp))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, img = cap.read(); cap.release()
            if not ok:
                continue
            for a, colour in enumerate([(0, 200, 255), (255, 120, 0)]):
                p = arr[row][a][np.isfinite(arr[row][a]).all(axis=1)]
                if len(p) == 0:
                    continue
                uv, depth = project(mats[cam], fn(p))
                fin = np.isfinite(depth)
                sgn = np.sign(np.median(depth[fin])) if fin.any() else 1.0
                for (u, v), d in zip(uv, depth):
                    if np.isfinite(u) and d * sgn > 0:
                        cv2.circle(img, (int(u), int(v)), 3, colour, -1)
            dst = out_dir / f"overlay_f{int(fid):04d}_cam{cam}.png"
            cv2.imwrite(str(dst), img)
            print(f"  wrote {dst}")
    print("\nOpen these. Dots on the mice, one colour each, settles it.")


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--out-dir", default="diagnostics/sbea_v5_out")
    ap.add_argument("--cameras", default="0,1,2,3")
    ap.add_argument("--mask-frames", type=int, default=10)
    args = ap.parse_args()

    stem = Path(args.stem)
    if not Path(f"{stem}-id3d.mat").exists():
        print(f"ERROR: {stem}-id3d.mat not found.", file=sys.stderr); return 1
    cameras = [int(c) for c in args.cameras.split(",")]

    mats_list, R_g, t_g = load_calibration(stem)
    mats = {c: mats_list[c] for c in cameras if c < len(mats_list)}
    coords, names = load_coords(stem)

    import cv2
    cap = cv2.VideoCapture(f"{stem}-camera-{cameras[0]}.avi")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()

    rule("SESSION")
    print(f"coords3d {coords.shape}   identities {names}")
    print(f"video {W}x{H}, {n_video} frames;  coords3d has {coords.shape[0]} rows")

    arr = check_layout(coords)
    if arr is None:
        return 2
    n_rows = arr.shape[0]
    arr_flat = np.nan_to_num(np.ascontiguousarray(arr.reshape(n_rows, -1, 3)), nan=-1e9)

    mask_frames = np.linspace(0, n_rows - 1, args.mask_frames + 2).astype(int)[1:-1]
    masks, n_mask = load_masks(stem, cameras, mask_frames, H, W)
    if n_mask and abs(n_mask - n_rows) > 1:
        print(f"\n!! masks have {n_mask} frames but coords3d has {n_rows} rows.")
        print("   (A) assumes they share an index; treat its delta with care.")

    flabel, delta, sA = solve_convention(arr_flat, mats, masks, cameras, R_g, t_g, n_rows, W, H)
    if flabel is None:
        return 3
    fn = frame_variants(R_g, t_g)[flabel]

    off, st, sB = solve_excerpt(stem, arr_flat, mats, cameras, fn, n_rows, n_video, W, H)
    write_overlays(stem, arr, mats, cameras, fn, off, st, Path(args.out_dir), n_video)

    rule("VERDICT")
    print(f"column layout : animal / kp / xyz  ->  ({n_rows}, 2, 16, 3)")
    print(f"frame         : {flabel}          (mask agreement {sA:.1%}, delta {delta})")
    print(f"mask index    : coords3d row = mask frame + {delta}")
    print(f"excerpt       : coords3d row = {off} + {st} * avi_frame   ({sB:.1%})")
    print(f"identities    : {names}")
    if sA > 0.75 and sB > 0.5:
        print("\nRESOLVED. Converter can be written against these four facts.")
    else:
        print("\nPARTIAL. (A) fixes the geometry; (B) is what pairs images with 3D.")
        print("Without (B) the 3D and the masks are still usable, but the .avi")
        print("frames cannot be attached to them -- and no images means no")
        print("training data, only evaluation targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
