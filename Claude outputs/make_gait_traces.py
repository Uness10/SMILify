#!/usr/bin/env python3
"""Joint angles over a few step cycles, across the single-view joint-limit sweep.

Every arm is read from its exported clip array, so all arms share one indexing
space (the 50 630-item single-view test split) and the traces are guaranteed to
describe the same frames.

    <root>/prior_study_results_matched/sv_reference/clip_sv_reference.npz   control, lambda=0
    <root>/prior_study_results/lam1e-*/sv_constrained/clip_sv_constrained.npz
    <root>/prior_study_results/sv_reference/clip_sv_reference.npz           old base (optional)

The frame window comes from renders/segments.json, which lists, per segment,
the test-split indices in time order.  Arms that are missing are reported and
skipped, so this still runs if only some clips exist.

Run from the repo root:
    python report_joint_limit_prior/make_gait_traces.py

Env overrides:
    SEG=0|1|2     which segment           (default 0)
    OFF=0         first sample in segment (default 0)
    N=40          samples to plot         (default 40)
"""
import os, json, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAD2DEG = 180.0 / np.pi
SEG = int(os.environ.get("SEG", "0"))
OFF = int(os.environ.get("OFF", "0"))
NUM = int(os.environ.get("N",   "40"))

ROOT     = Path(os.environ.get("SMILIFY_ROOT", Path(__file__).resolve().parents[1]))
PSR      = ROOT / "prior_study_results"
SEGMENTS = PSR / "renders/segments.json"
RENDERJS = PSR / "renders/lam100/render.json"      # only for the sample interval
LIMITS   = ROOT / "docs/joint_dofs/stick_insect_joint_limits.csv"
OUTDIR   = Path(__file__).resolve().parent / "figures"

PAPER, INK, INKSOFT = "#FAF7F4", "#1E2430", "#4A5260"
HAIR,  MUTED        = "#E4DDD5", "#9A968F"
TEAL,  CORAL        = "#0F6F68", "#C85A44"

# label, npz path (relative to ROOT), colour, lw, linestyle
ARMS = [
    ("control $\\lambda=0$ (epoch-matched)",
     "prior_study_results_matched/sv_reference/clip_sv_reference.npz", INK, 2.1, "-"),
    ("base (+0 ep)",
     "prior_study_results/sv_reference/clip_sv_reference.npz", "#9AA0A8", 1.2, (0, (3, 2))),
    (r"$\lambda=10^{-4}$",
     "prior_study_results/lam1e-4/sv_constrained/clip_sv_constrained.npz", "#8FBDB7", 1.4, "-"),
    (r"$\lambda=10^{-3}$",
     "prior_study_results/lam1e-3/sv_constrained/clip_sv_constrained.npz", "#4E948C", 1.3, "-"),
    (r"$\lambda=10^{-2}$",
     "prior_study_results/lam1e-2/sv_constrained/clip_sv_constrained.npz", TEAL, 1.4, "-"),
    (r"$\lambda=10^{-1}$",
     "prior_study_results/lam1e-1/sv_constrained/clip_sv_constrained.npz", CORAL, 2.1, "-"),
]

PANELS = [
    ("l_2_tr_r", "x", "Coxa–trochanter", "levation / depression"),
    ("l_2_ti_r", "x", "Femur–tibia",     "flexion / extension"),
    ("l_2_fe_r", "x", "Femur (locked)",  "chronic violator"),
]

# --------------------------------------------------------------------------- #
if not SEGMENTS.exists():
    sys.exit(f"missing {SEGMENTS}\n(run from the repo root, or set SMILIFY_ROOT)")
if not LIMITS.exists():
    sys.exit(f"missing {LIMITS}")

segs = json.load(open(SEGMENTS))["segments"]
idx_all = np.asarray(segs[SEG]["indices"], dtype=int)
idx = idx_all[OFF:OFF + NUM]
if len(idx) < 2:
    sys.exit(f"window empty: segment {SEG} has {len(idx_all)} samples, OFF={OFF} N={NUM}")

# seconds per sample: median video-frame gap / fps, from render.json when present
dt = 5.0 / 30.0
if RENDERJS.exists():
    rj = json.load(open(RENDERJS))
    s = next((x for x in rj.get("segments", []) if x.get("name") == segs[SEG]["name"]), None)
    src_fps = 30.0
    if s and s.get("median_frame_gap"):
        dt = float(s["median_frame_gap"]) / src_fps
print(f"segment {SEG} ({segs[SEG]['name']}): {len(idx_all)} samples; "
      f"plotting {len(idx)} from offset {OFF}  "
      f"(test-split rows {idx[0]}..{idx[-1]}, {dt:.3f} s/sample, "
      f"{len(idx)*dt:.1f} s total)")

# --------------------------------------------------------------------------- #
def load(rel):
    p = ROOT / rel
    if not p.exists():
        return None, f"missing {rel}"
    z = np.load(p)
    if "poses" not in z:
        return None, f"no 'poses' in {rel} (keys: {list(z.keys())})"
    poses = z["poses"]
    if idx.max() >= poses.shape[0]:
        return None, f"{rel}: only {poses.shape[0]} frames, need index {idx.max()}"
    return poses[idx] * RAD2DEG, f"ok  {poses.shape}"


data, present = {}, []
print()
for label, rel, colour, lw, ls in ARMS:
    arr, msg = load(rel)
    print(f"  {label:38s} {msg}")
    if arr is not None:
        data[rel] = arr
        present.append((label, rel, colour, lw, ls))
if not present:
    sys.exit("\nno clip arrays found - nothing to plot")
ctrl_rel = ARMS[0][1]
if ctrl_rel not in data:
    print("\nWARNING: control clip absent; the first available arm is used as reference")
ref_rel = ctrl_rel if ctrl_rel in data else present[0][1]

# --------------------------------------------------------------------------- #
meta_json = (ROOT / ref_rel).with_suffix(".json")
joint_names = json.load(open(meta_json))["joint_names"]
LIM = {(r["repo_joint"], r["axis_local"]):
       (float(r["local_min_deg"]), float(r["local_max_deg"]))
       for r in csv.DictReader(open(LIMITS))}

t = np.arange(len(idx)) * dt
fig, axes = plt.subplots(1, len(PANELS), figsize=(12.4, 3.9), facecolor=PAPER)
print(f"\n{'joint':11s} {'ax':2s} {'ROM ref':>8s} {'ROM min':>8s} {'kept':>6s} "
      f"{'viol ref':>9s} {'viol min':>9s}")

for ax, (joint, axis, caption, note) in zip(axes, PANELS):
    j, a = joint_names.index(joint), "xyz".index(axis)
    lo, hi = LIM[(joint, axis)]
    series = {rel: data[rel][:, j, a] for _, rel, _, _, _ in present}
    ref = series[ref_rel]
    strong = series[present[-1][1]]

    allv = np.concatenate(list(series.values()))
    pad  = max(0.18 * np.ptp(allv), 3.0)
    head = 2.35 if ax is axes[0] else 1.15
    foot = 1.55 if ((ref < lo) | (ref > hi)).any() else 1.15
    ymin, ymax = allv.min() - pad * foot, allv.max() + pad * head

    ax.set_facecolor(PAPER)
    ax.axhspan(lo, hi, color=TEAL, alpha=0.065, zorder=0)
    for y in (lo, hi):
        if ymin < y < ymax:
            ax.axhline(y, color=TEAL, lw=1.0, ls=(0, (4, 3)), alpha=0.7, zorder=1)
    for label, rel, colour, lw, ls in present:
        ax.plot(t, series[rel], color=colour, lw=lw, ls=ls, label=label, zorder=3,
                solid_capstyle="round")

    v_r = ((ref < lo) | (ref > hi)).mean() * 100
    v_s = ((strong < lo) | (strong > hi)).mean() * 100
    rom_r, rom_s = np.ptp(ref), np.ptp(strong)
    print(f"{joint:11s} {axis:2s} {rom_r:7.1f}° {rom_s:7.1f}° {rom_s/rom_r*100:5.0f}% "
          f"{v_r:8.1f}% {v_s:8.1f}%")

    ax.set_title(f"{caption}\n" + r"$\bf{" + joint.replace("_", r"\_") + r"}$"
                 + f" · {axis}-axis · {note}", fontsize=8.6, color=INK, pad=6)
    ax.set_ylim(ymin, ymax); ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)", fontsize=8.4, color=INKSOFT)
    ax.tick_params(labelsize=7.6, colors=INKSOFT, length=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(HAIR)
    ax.grid(axis="y", color=HAIR, lw=0.5, alpha=0.75)
    ax.set_axisbelow(True)

    stat = f"range of motion  {rom_r:.0f}° → {rom_s:.0f}°  ({rom_s/rom_r*100:.0f}% kept)"
    if v_r > 0:
        stat += f"\nframes out of range  {v_r:.0f}% → {v_s:.0f}%"
    ax.text(0.025, 0.025, stat, transform=ax.transAxes, va="bottom",
            fontsize=7.4, color=INKSOFT, linespacing=1.5)
    ax.text(0.975, 0.025, f"authored  {lo:.0f}…{hi:.0f}°", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=7.0, color=MUTED)

axes[0].set_ylabel("joint angle (deg)", fontsize=8.4, color=INKSOFT)
axes[0].legend(fontsize=6.8, frameon=False, labelcolor=INKSOFT, loc="upper left",
               ncol=2, columnspacing=1.0, handlelength=1.8, borderpad=0.1)
fig.text(0.5, 0.008,
         f"single-view arms, middle right leg · segment {SEG:02d} "
         f"samples {OFF}–{OFF+len(idx)-1} ({len(idx)*dt:.1f} s) · "
         "reference = epoch-matched $\\lambda=0$ control · "
         "shaded band = authored anatomical range",
         ha="center", fontsize=7.1, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))

OUTDIR.mkdir(exist_ok=True)
out = OUTDIR / f"fig_gait_traces_seg{SEG:02d}_{OFF:03d}"
fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAPER)
fig.savefig(out.with_suffix(".pdf"), facecolor=PAPER)
print(f"\nwrote {out.with_suffix('.png')}\n      {out.with_suffix('.pdf')}")
