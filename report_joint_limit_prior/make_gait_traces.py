#!/usr/bin/env python3
"""Joint angles over a few step cycles, across the joint-limit sweep.

Reads the per-frame parameter dumps the render pipeline already wrote
(prior_study_results/renders/<arm>/*_parameters.pkl) — so it needs no
checkpoints, no GPU and no cluster access.

Each dump holds `joint_rotations` (54, 3), axis-angle radians, indexed
joint_names[1:]  (joint 0 = global root, stored separately as
`global_rotation`).  Frames are every 5th frame of a 30 fps clip, and the
first 40 dumps are render segment 00 => 6.7 s at 6 Hz.

Usage:  python make_gait_traces.py
"""
import pickle, glob, json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAD2DEG = 180.0 / np.pi
FPS, STRIDE, SEG_FRAMES = 30.0, 5, 40

# repo root = parent of report_joint_limit_prior/ ; override with SMILIFY_ROOT
import os
ROOT    = Path(os.environ.get("SMILIFY_ROOT", Path(__file__).resolve().parents[1]))
RENDERS = ROOT / "prior_study_results/renders"
LIMITS  = ROOT / "docs/joint_dofs/stick_insect_joint_limits.csv"
NAMES   = ROOT / "prior_study_results/lam1e-1/mv_constrained/clip_mv_constrained.json"
OUTDIR  = Path(__file__).resolve().parent / "figures"

# ---- deck palette ---------------------------------------------------------
PAPER, INK, INKSOFT = "#FAF7F4", "#1E2430", "#4A5260"
HAIR,  MUTED        = "#E4DDD5", "#9A968F"
TEAL,  CORAL        = "#0F6F68", "#C85A44"

ARMS = [   # label, directory, colour, linewidth
    ("no prior",           "sv_reference", INK,       2.1),
    (r"$\lambda=10^{-4}$", "lam1e-4",      "#8FBDB7", 1.4),
    (r"$\lambda=10^{-2}$", "lam1e-2",      TEAL,      1.4),
    (r"$\lambda=10^{-1}$", "lam1e-1",      CORAL,     2.1),
]

PANELS = [   # joint, axis, anatomical caption, short note
    ("l_2_tr_r", "x", "Coxa–trochanter",  "levation / depression"),
    ("l_2_ti_r", "x", "Femur–tibia",      "flexion / extension"),
    ("l_2_fe_r", "x", "Femur (locked)",   "chronic violator"),
]


def v_ref_pre(series, lo, hi):
    """True when the reference leaves the authored range (stat block is 2 lines)."""
    r = series["sv_reference"]
    return bool(((r < lo) | (r > hi)).any())


def load_arm(d):
    fs = sorted(glob.glob(str(RENDERS / d / "*_parameters.pkl")))[:SEG_FRAMES]
    return np.stack([pickle.load(open(f, "rb"))["joint_rotations"] for f in fs]) * RAD2DEG


LIM = {(r["repo_joint"], r["axis_local"]): (float(r["local_min_deg"]), float(r["local_max_deg"]))
       for r in csv.DictReader(open(LIMITS))}

joint_names = json.load(open(NAMES))["joint_names"]
data = {d: load_arm(d) for _, d, _, _ in ARMS}
t = np.arange(SEG_FRAMES) * STRIDE / FPS

fig, axes = plt.subplots(1, len(PANELS), figsize=(12.4, 3.75), facecolor=PAPER)

for ax, (joint, axis, caption, note) in zip(axes, PANELS):
    j, a = joint_names.index(joint) - 1, "xyz".index(axis)
    lo, hi = LIM[(joint, axis)]
    series = {d: data[d][:, j, a] for _, d, _, _ in ARMS}

    # y-range from the traces, not from the authored band (which can be huge)
    allv = np.concatenate(list(series.values()))
    pad = max(0.18 * np.ptp(allv), 3.0)
    head = 1.95 if ax is axes[0] else 1.15      # headroom only where the legend sits
    foot = 1.55 if v_ref_pre(series, lo, hi) else 1.15   # room for a 2-line stat
    ymin, ymax = allv.min() - pad * foot, allv.max() + pad * head

    ax.set_facecolor(PAPER)
    ax.axhspan(lo, hi, color=TEAL, alpha=0.065, zorder=0)
    for y in (lo, hi):
        if ymin < y < ymax:
            ax.axhline(y, color=TEAL, lw=1.0, ls=(0, (4, 3)), alpha=0.7, zorder=1)

    for label, d, colour, lw in ARMS:
        ax.plot(t, series[d], color=colour, lw=lw, label=label, zorder=3,
                solid_capstyle="round")

    ref, strong = series["sv_reference"], series["lam1e-1"]
    v_ref = ((ref < lo) | (ref > hi)).mean() * 100
    v_str = ((strong < lo) | (strong > hi)).mean() * 100
    rom_r, rom_s = np.ptp(ref), np.ptp(strong)

    ax.set_title(f"{caption}\n" + r"$\bf{" + joint.replace("_", r"\_") + r"}$"
                 + f" · {axis}-axis · {note}", fontsize=8.6, color=INK, pad=6)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)", fontsize=8.4, color=INKSOFT)
    ax.tick_params(labelsize=7.6, colors=INKSOFT, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(HAIR)
    ax.grid(axis="y", color=HAIR, lw=0.5, alpha=0.75)
    ax.set_axisbelow(True)

    stat = (f"range of motion  {rom_r:.0f}° → {rom_s:.0f}°  "
            f"({rom_s / rom_r * 100:.0f}% kept)")
    if v_ref > 0:
        stat += f"\nframes out of range  {v_ref:.0f}% → {v_str:.0f}%"
    ax.text(0.025, 0.025, stat, transform=ax.transAxes, va="bottom",
            fontsize=7.4, color=INKSOFT, linespacing=1.5)
    ax.text(0.975, 0.025, f"authored  {lo:.0f}…{hi:.0f}°", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=7.0, color=MUTED)

axes[0].set_ylabel("joint angle (deg)", fontsize=8.4, color=INKSOFT)
axes[0].legend(fontsize=7.6, frameon=False, labelcolor=INKSOFT, loc="upper left",
               ncol=2, columnspacing=1.1, handlelength=1.6, borderpad=0.1)

fig.text(0.5, 0.008,
         "single-view arms, middle right leg, render segment 00 "
         "(6.7 s at 6 Hz)   ·   shaded band = authored anatomical range",
         ha="center", fontsize=7.1, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
OUTDIR.mkdir(exist_ok=True)
out = OUTDIR / "fig_gait_traces"
fig.savefig(out.with_suffix(".png"), dpi=200, facecolor=PAPER)
fig.savefig(out.with_suffix(".pdf"), facecolor=PAPER)
print("wrote", out.with_suffix(".png"))
