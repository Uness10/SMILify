#!/usr/bin/env python3
"""Find clean gait-cycle windows by looking at the inference footage itself.

The collage MP4s that run_singleview_inference.py writes already contain the
keypoints: SMALJointDrawer draws joint j with the colour
config.MARKER_COLORS[j] = (255 - 255j/55, 255j/55, 100) in RGB, so a marker's
colour is its joint index. Panel 0 carries the target (ground-truth reprojected)
keypoints, panel 1 the model's own reprojected joints; the panels share a
coordinate frame, so the distance between them is a per-frame fit-quality proxy.

H.264 4:2:0 chroma subsampling blurs the hue, so the index is estimated per pixel
from the red channel and each joint is taken as the connected component of pixels
whose estimated index is closest to it, gated by the previous frame's position.

    python find_gait_windows.py video.mp4 --panel-width 512 --cycles 4

Prints ranked candidate windows and writes gait_windows.png.

Joints drawn at y < 12 are the "not visible" sentinel row that draw_smal_joints.py
parks invisible joints on, so those rows are masked out.
"""
from __future__ import annotations

import argparse
import numpy as np
import cv2
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

NJ = 55
STEP = 255.0 / NJ
JN = ['b_t', 'b_a_1', 'b_a_2', 'b_a_3', 'b_a_4', 'b_a_5',
      'l_1_co_r', 'l_1_tr_r', 'l_1_fe_r', 'l_1_ti_r', 'l_1_ta_r', 'l_1_pt_r',
      'l_2_co_r', 'l_2_tr_r', 'l_2_fe_r', 'l_2_ti_r', 'l_2_ta_r', 'l_2_pt_r',
      'l_3_co_r', 'l_3_tr_r', 'l_3_fe_r', 'l_3_ti_r', 'l_3_ta_r', 'l_3_pt_r',
      'w_1_r', 'w_2_r',
      'l_1_co_l', 'l_1_tr_l', 'l_1_fe_l', 'l_1_ti_l', 'l_1_ta_l', 'l_1_pt_l',
      'l_2_co_l', 'l_2_tr_l', 'l_2_fe_l', 'l_2_ti_l', 'l_2_ta_l', 'l_2_pt_l',
      'l_3_co_l', 'l_3_tr_l', 'l_3_fe_l', 'l_3_ti_l', 'l_3_ta_l', 'l_3_pt_l',
      'w_1_l', 'w_2_l', 'b_h', 'ma_r', 'an_1_r', 'an_2_r', 'an_3_r',
      'ma_l', 'an_1_l', 'an_2_l', 'an_3_l']
LEGS = [('l_1_r', 'R1 front'), ('l_2_r', 'R2 middle'), ('l_3_r', 'R3 hind'),
        ('l_1_l', 'L1 front'), ('l_2_l', 'L2 middle'), ('l_3_l', 'L3 hind')]


def locate(rgb, targets, tol=1.6, y_min=12, prev=None, gate=45.0):
    p = rgb.astype(np.float32)
    R, G, B = p[..., 0], p[..., 1], p[..., 2]
    mask = (np.abs(B - 100) <= 32) & (np.abs(R + G - 255) <= 48) & ((p.max(2) - p.min(2)) > 45)
    mask[:y_min] = False
    idx = (255.0 - R) / STEP
    out = {}
    for j in targets:
        m = (mask & (np.abs(idx - j) <= tol)).astype(np.uint8)
        if m.sum() < 6:
            out[j] = (np.nan, np.nan)
            continue
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        cands = [(stats[i, cv2.CC_STAT_AREA], cent[i]) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= 6]
        if not cands:
            out[j] = (np.nan, np.nan)
            continue
        if prev is not None and j in prev and np.isfinite(prev[j][0]):
            near = [(a, c) for a, c in cands
                    if np.hypot(c[0] - prev[j][0], c[1] - prev[j][1]) <= gate]
            cands = near or cands
        out[j] = tuple(max(cands, key=lambda t: t[0])[1])
    return out


def track(path, targets, panel, panel_w, y_min):
    cap = cv2.VideoCapture(path)
    rows, prev = [], None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr[:, panel * panel_w:(panel + 1) * panel_w], cv2.COLOR_BGR2RGB)
        d = locate(rgb, targets, y_min=y_min, prev=prev)
        rows.append(d)
        prev = {k: v for k, v in d.items() if np.isfinite(v[0])} or prev
    cap.release()
    return np.array([[r[j] for j in targets] for r in rows], float)


def clean(xy, F, med=5, max_gap=8, jump=15.0):
    out = xy.copy()
    for c in range(2):
        v = out[:, c]
        ok = np.isfinite(v)
        if ok.sum() < 10:
            continue
        m = median_filter(np.where(ok, v, np.nanmedian(v)), size=med, mode="nearest")
        v[ok & (np.abs(v - m) > jump)] = np.nan
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        t = np.arange(F)
        vi = np.interp(t, t[ok], v[ok])
        bad = np.zeros(F, bool)
        gaps = np.flatnonzero(~ok)
        if len(gaps):
            for sgrp in np.split(gaps, np.flatnonzero(np.diff(gaps) > 1) + 1):
                if len(sgrp) > max_gap:
                    bad[sgrp] = True
        vi[bad] = np.nan
        out[:, c] = vi
    return out


def swings(sig, min_swing=3, min_stance=5):
    s = savgol_filter(np.where(np.isfinite(sig), np.nan_to_num(sig, nan=np.nanmedian(sig)), 0), 9, 2)
    sw = np.gradient(s) > 0
    for _ in range(3):
        b = np.concatenate([[0], np.flatnonzero(np.diff(sw.astype(int))) + 1, [len(sw)]])
        ch = False
        for a, c in zip(b[:-1], b[1:]):
            if c - a < (min_swing if sw[a] else min_stance):
                sw[a:c] = not sw[a]
                ch = True
        if not ch:
            break
    i = np.flatnonzero(sw)
    if not len(i):
        return []
    return [(int(r[0]), int(r[-1]) + 1)
            for r in np.split(i, np.flatnonzero(np.diff(i) > 1) + 1) if len(r) >= min_swing]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--panel", type=int, default=0, help="0 = GT keypoints, 1 = predicted")
    ap.add_argument("--panel-width", type=int, default=512)
    ap.add_argument("--foot-segment", default="pt", choices=["ta", "pt"])
    ap.add_argument("--cycles", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--y-min", type=int, default=12)
    ap.add_argument("--out", default="gait_windows.png")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="legs to drop, e.g. l_1_l when its marker is not tracked reliably")
    a = ap.parse_args()

    want = ["b_h", "b_a_5"] + [f"{lg}"[:3] + a.foot_segment + f"{lg}"[3:] for lg, _ in LEGS]
    want = ["b_h", "b_a_5"] + [f"l_{lg.split('_')[1]}_{a.foot_segment}_{lg.split('_')[2]}"
                               for lg, _ in LEGS]
    tg = [JN.index(n) for n in want]
    A = track(a.video, tg, a.panel, a.panel_width, a.y_min)
    F = len(A)
    rate = np.isfinite(A[:, :, 0]).mean(0)
    print(f"{F} frames; marker detection rate")
    for n, r in zip(want, rate):
        print(f"  {n:12s} {r:.2f}" + ("   <-- unreliable" if r < 0.9 else ""))

    C = {n: clean(A[:, i], F) for i, n in enumerate(want)}
    head, tail = C["b_h"], C["b_a_5"]
    e = head - tail
    L = np.linalg.norm(e, axis=1, keepdims=True)
    e = e / L
    nrm = np.stack([-e[:, 1], e[:, 0]], 1)
    origin = (head + tail) / 2

    FOOT, keep = {}, []
    for i, (lg, lab) in enumerate(LEGS):
        n = want[2 + i]
        d = C[n] - origin
        FOOT[lg] = np.stack([(d * e).sum(1), (d * nrm).sum(1)], 1) / L[:, 0:1] * 100
        if rate[2 + i] >= 0.9 and lg not in a.exclude:
            keep.append(lg)
    print("legs used for scoring:", ", ".join(keep))

    EV = {lg: swings(FOOT[lg][:, 0]) for lg in keep}
    per_all = np.concatenate([np.diff([s[0] for s in EV[lg]]) for lg in keep if len(EV[lg]) > 2])
    T = float(np.median(per_all))
    print(f"median step period {T:.1f} frames")

    heading = np.degrees(np.unwrap(np.arctan2(e[:, 1], e[:, 0])))
    print(f"body heading range {np.ptp(heading):.1f} deg, "
          f"head-tail distance CV {float(np.std(L)/np.mean(L)):.3f}")

    for nc in a.cycles:
        W = int(round(nc * T))
        best = []
        for s0 in range(0, F - W, 2):
            s1 = s0 + W
            sc, ok = [], True
            for lg in keep:
                lo = [x[0] for x in EV[lg] if s0 <= x[0] < s1]
                if len(lo) < nc or np.isfinite(FOOT[lg][s0:s1, 0]).mean() < 0.995:
                    ok = False
                    break
                p = np.diff(lo)
                sc.append(np.std(p) / np.mean(p))
                amps = [np.ptp(FOOT[lg][l:min(s1, l + int(T)), 0]) for l in lo[:-1]]
                if len(amps) > 1:
                    sc.append(np.std(amps) / np.mean(amps))
            if ok:
                sc.append(np.std(heading[s0:s1]) / 10.0)
                best.append((float(np.mean(sc)), s0, s1))
        best.sort()
        print(f"\n{nc}-cycle windows ({W} frames) — lower score = more regular:")
        shown = []
        for s, s0, s1 in best:
            if any(abs(s0 - x) < 0.7 * T for x in shown):
                continue
            shown.append(s0)
            print(f"  frames {s0:4d}-{s1:4d}   score {s:.4f}")
            if len(shown) >= 5:
                break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HUE = {"1": "#0072B2", "2": "#D55E00", "3": "#009E73"}
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1, 1]})
    for lg, lab in LEGS:
        st = "-" if lg.endswith("r") else "--"
        axes[0].plot(FOOT[lg][:, 0], st, color=HUE[lg.split("_")[1]], lw=1.2,
                     alpha=1.0 if lg in keep else 0.3, label=lab + ("" if lg in keep else " (dropped)"))
    axes[0].set_ylabel("foot position along body axis (%BL)")
    axes[0].legend(fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axes[1].plot(heading, color="#1c1c1c", lw=1); axes[1].set_ylabel("heading (deg)")
    axes[2].plot(L[:, 0], color="#1c1c1c", lw=1); axes[2].set_ylabel("head-tail (px)")
    axes[2].set_xlabel("frame")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
