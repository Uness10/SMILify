#!/usr/bin/env python3
"""Gait analysis of a SMILify animation export (.npz + .json).

    python analyse_gait.py --npz run_sv_ref.npz --smal-file 3D_model_prep/SMILy_STICK.pkl \
        --windows 14-182 252-378 --label sv_ref --out figs/

Produces, per run:
  fig1_ThC_two_dominant_axes   protraction/retraction and levation/depression of all
                               six thorax-coxa joints over the selected gait cycles
  fig2_hinge_angles            CTr ("C-TF") and FTi hinge angles - the downstream
                               joints that are now constrained to one DoF
  fig3_axis_dominance          per-joint PCA of the axis-angle vectors: how much
                               rotation each axis carries, and how the two dominant
                               axes line up with the anatomical ones
  fig4_tarsus_horizontal       tarsus position relative to the root, body frame,
  fig5_tarsus_sagittal         one trace per step cycle so successive steps overlay
  fig6_gait_diagram            swing/stance bars + foot longitudinal position
  angles.csv, tarsi.csv, axis_dominance.csv, steps.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smil_fk
import anatomical as ana

# --- palette: hue = leg pair, line style = side (composite encoding, CVD-checked)
HUE = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
STYLE = {"r": "-", "l": "--"}
INK, MUTED, GRID = "#1c1c1c", "#5c5c5c", "#d8d8d4"
SURFACE = "#fcfcfb"
SWING_SHADE = "#00000010"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.grid": True, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "legend.frameon": False,
})


# ------------------------------------------------------------------ step cycles
def detect_steps(foot_long, smooth=9, min_swing=3, min_stance=5):
    """Swing = the foot moves anteriorly relative to the body; stance = posteriorly.

    The foot's longitudinal position in the body frame is a sawtooth: slow
    posterior drift while the tarsus is planted and the body walks over it, then a
    fast anterior return. Classifying on the *sign* of the smoothed velocity (with
    a minimum run length to absorb regressor jitter) is therefore both simple and
    physically meaningful, and it recovers the duty factor correctly - a fixed
    percentile threshold does not.

    Returns [(liftoff, touchdown), ...] frame indices.
    """
    s = savgol_filter(np.asarray(foot_long, float), max(5, smooth | 1), 2)
    d = np.gradient(s)
    sw = d > 0
    # absorb runs shorter than the minimum into their neighbour
    for _ in range(3):
        idx = np.flatnonzero(np.diff(sw.astype(int))) + 1
        bounds = np.concatenate([[0], idx, [len(sw)]])
        changed = False
        for a, b in zip(bounds[:-1], bounds[1:]):
            lim = min_swing if sw[a] else min_stance
            if b - a < lim:
                sw[a:b] = not sw[a]
                changed = True
        if not changed:
            break
    idx = np.flatnonzero(sw)
    if not len(idx):
        return []
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(int(r[0]), int(r[-1]) + 1) for r in runs if len(r) >= min_swing]


def shade_swing(ax, steps, lo, hi):
    for a, b in steps:
        if b >= lo and a <= hi:
            ax.axvspan(max(a, lo), min(b, hi), color=SWING_SHADE, lw=0, zorder=0)


# ------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, help="animation export written by --export_animation")
    p.add_argument("--json", default=None, help="sidecar (defaults to <npz stem>.json)")
    p.add_argument("--smal-file", required=True, help="3D_model_prep/SMILy_STICK.pkl")
    p.add_argument("--windows", nargs="+", default=["all"],
                   help="frame ranges to plot, e.g. 14-182 252-378 (default: all)")
    p.add_argument("--label", default="run", help="name used in titles and filenames")
    p.add_argument("--out", default="gait_figs", help="output directory")
    p.add_argument("--fps", type=float, default=None, help="override the sidecar fps")
    p.add_argument("--propagate-scaling", action="store_true",
                   help="set if the checkpoint was trained/rendered with propagate_scaling=True")
    p.add_argument("--no-per-frame-betas", action="store_true",
                   help="use the averaged betas instead of betas_per_frame")
    p.add_argument("--frame-offset", type=int, default=0,
                   help="dataset frame index of npz frame 0, if the export does not start at 0; "
                        "--windows are then given in dataset frame numbers")
    p.add_argument("--foot-segment", default="ta", choices=["ta", "pt"],
                   help="'ta' = tarsus (default, what the supervisor asked for), 'pt' = pretarsus tip")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.npz)
    side = args.json or os.path.splitext(args.npz)[0] + ".json"
    meta = json.load(open(side)) if os.path.exists(side) else {}
    fps = args.fps or float(meta.get("fps", d["fps"] if "fps" in d else 30.0))

    poses = d["poses"].astype(float)                      # (F, J, 3) axis-angle, 0 = global
    F = poses.shape[0]
    betas = d["betas_per_frame"] if ("betas_per_frame" in d and not args.no_per_frame_betas) else d["betas"]
    lbs = d["log_beta_scales"].astype(float) if "log_beta_scales" in d else None
    btr = d["betas_trans"].astype(float) if "betas_trans" in d else None

    model = smil_fk.load_model(args.smal_file)
    if poses.shape[1] != model.n_joints:
        sys.exit(f"npz has {poses.shape[1]} joints but the model has {model.n_joints}")

    joints, _, _ = smil_fk.forward_kinematics(
        model, poses, betas=np.asarray(betas, float), log_beta_scales=lbs, betas_trans=btr,
        propagate_scaling=args.propagate_scaling, zero_global=True,
    )
    joints = joints - joints[:, 0:1, :]                   # root-relative body frame

    # scale: rest-pose head-to-abdomen-tip distance = 1 body length
    BL = float(np.linalg.norm(model.J_rest[model.joint("b_h")] - model.J_rest[model.joint("b_a_5")]))
    J_pc = joints / BL * 100.0                            # % body length

    angles = {}
    for leg in ana.LEGS:
        a_ = dict(ana.leg_angles(joints, model, *leg))
        a_.update(ana.thc_intrinsic(poses, model, *leg))
        a_.update(ana.hinge_rotation(poses, model, *leg, "tr"))
        a_.update(ana.hinge_rotation(poses, model, *leg, "ti"))
        a_.update(ana.hinge_rotation(poses, model, *leg, "fe"))
        angles[leg] = a_
    FOOT = {leg: J_pc[:, model.joint(ana.jname(leg[0], leg[1], args.foot_segment))] for leg in ana.LEGS}
    STEPS = {leg: detect_steps(FOOT[leg][:, 0]) for leg in ana.LEGS}

    wins = []
    for w in args.windows:
        if w == "all":
            wins.append((0, F - 1))
        else:
            a, b = w.split("-")
            a, b = int(a) - args.frame_offset, int(b) - args.frame_offset
            if b <= 0 or a >= F:
                sys.exit(f"window {w} is outside the exported frame range "
                         f"[{args.frame_offset}, {args.frame_offset + F - 1}]")
            wins.append((max(a, 0), min(b, F - 1)))

    print(f"{args.label}: {F} frames @ {fps} fps")
    print(f"  rig: {os.path.basename(args.smal_file)}  "
          f"({model.n_joints} joints, {model.shapedirs.shape[2]} betas, "
          f"body length {BL:.4f} model units)")
    print("  rest pose (absolute angles are measured from these — they must match "
          "the rig the checkpoint was trained on):")
    print(f"    {'leg':<11}{'alpha':>8}{'beta':>8}{'gamma':>8}{'delta':>8}")
    for leg in ana.LEGS:
        r = ana.rest_angles(model, *leg)
        print(f"    {ana.LEG_LABEL[leg]:<11}{r['alpha']:8.1f}{r['beta']:8.1f}"
              f"{r['gamma']:8.1f}{r['delta']:8.1f}")
    for leg in ana.LEGS:
        per = np.diff([s[0] for s in STEPS[leg]])
        if len(per):
            print(f"  {ana.LEG_LABEL[leg]:10s} {len(STEPS[leg]):2d} steps, "
                  f"period {np.median(per):.1f} frames ({np.median(per)/fps*1000:.0f} ms), "
                  f"duty {1 - np.mean([(b-a) for a, b in STEPS[leg]])/np.median(per):.2f} "
                  f"(stance fraction)")

    # ---------------------------------------------------------------- fig 1 & 2
    def series_figure(fname, rows, title, ylabels):
        for wi, (lo, hi) in enumerate(wins):
            t = np.arange(lo, hi + 1)
            fig, axes = plt.subplots(len(rows), 2, figsize=(12.5, 2.4 * len(rows)),
                                     sharex=True, squeeze=False)
            for ri, (rk, rlab) in enumerate(rows):
                for ci, key in enumerate(rk):
                    ax = axes[ri][ci]
                    for legn in (1, 2, 3):
                        for s in ("r", "l"):
                            if (legn, s) not in [(rlab, "r"), (rlab, "l")]:
                                continue
                            leg = (legn, s)
                            ax.plot(t, angles[leg][key][lo:hi + 1], STYLE[s], color=HUE[legn],
                                    lw=1.6, label=ana.LEG_LABEL[leg])
                    shade_swing(ax, STEPS[(rlab, "r")], lo, hi)
                    ax.set_ylabel(ylabels[key], fontsize=8)
                    if ri == 0:
                        ax.set_title(ylabels[key].split("\n")[0], color=MUTED, pad=18)
                    ax.legend(fontsize=7, ncol=2, loc="lower center",
                              bbox_to_anchor=(0.5, 1.005), borderaxespad=0)
                    ax.margins(y=0.08)
            for ax in axes[-1]:
                ax.set_xlabel("frame")
            fig.suptitle(f"{title} — {args.label} — frames {lo}–{hi} "
                         f"({(hi-lo)/fps:.1f} s @ {fps:g} fps)", y=1.0, fontsize=11)
            fig.text(0.005, 0.005, "shaded = swing phase of the right leg of that pair",
                     fontsize=7, color=MUTED)
            fig.tight_layout()
            fig.savefig(f"{args.out}/{fname}_{args.label}_w{wi}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

    K1 = ("ThC_alpha_protraction_abs", "ThC_beta_levation_abs")
    series_figure("fig1_ThC_two_dominant_axes", [(K1, 1), (K1, 2), (K1, 3)],
                  "Thorax-coxa (ThC) — the two dominant axes of the joint that moves most",
                  {"ThC_alpha_protraction_abs": "protraction / retraction\nα (deg, + anterior)",
                   "ThC_beta_levation_abs": "levation / depression\nβ (deg, + dorsal)"})

    K2 = ("CTr_delta", "FTi_gamma")
    series_figure("fig2_hinge_angles", [(K2, 1), (K2, 2), (K2, 3)],
                  "The downstream joints that are now hinges (1 DoF)",
                  {"CTr_delta": "CTr / C-TF\nδ (deg, interior)",
                   "FTi_gamma": "FTi\nγ (deg, 180 = straight)"})

    # ------------------------------------------------------------------- fig 3
    seg_of = {"co": "ThC (ball, 3 DoF)", "tr": "CTr (hinge)", "fe": "trochanter-femur (fused)",
              "ti": "FTi (hinge)", "ta": "TiTa"}
    lo, hi = wins[0]
    win_idx = np.arange(lo, hi + 1)
    rows_csv = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    xs, labels, colors = [], [], []
    for seg in ("co", "tr", "fe", "ti", "ta"):
        for leg in ana.LEGS:
            ji = model.joint(ana.jname(leg[0], leg[1], seg))
            V, sd, proj, mu = ana.dominant_axes(poses, ji, win_idx)
            nrm = ana.leg_plane_normal(model, *leg)
            rows_csv.append(dict(
                joint=ana.jname(leg[0], leg[1], seg), kind=seg_of[seg], leg=ana.LEG_LABEL[leg],
                sd1_deg=sd[0], sd2_deg=sd[1], sd3_deg=sd[2],
                frac_axis1=sd[0]**2 / max((sd**2).sum(), 1e-12),
                frac_axis12=(sd[0]**2 + sd[1]**2) / max((sd**2).sum(), 1e-12),
                axis1=np.round(V[0], 3).tolist(), axis2=np.round(V[1], 3).tolist(),
                axis1_vs_protraction_z=abs(V[0][2]), axis1_vs_levation_x=abs(V[0][0]),
                axis1_vs_legplane_normal=abs(float(V[0] @ nrm)),
                range_deg=float(np.ptp(proj[win_idx, 0])),
            ))
            xs.append(sd); labels.append(f"{seg}·{ana.LEG_LABEL[leg][:2]}"); colors.append(HUE[leg[0]])
    xs = np.array(xs)
    order = np.argsort(-xs[:, 0])
    ax = axes[0]
    ypos = np.arange(len(order))
    ax.barh(ypos, xs[order, 0], color=[colors[i] for i in order], height=.72, label="axis 1")
    ax.barh(ypos, xs[order, 1], color="#00000055", height=.34, label="axis 2")
    ax.set_yticks(ypos); ax.set_yticklabels([labels[i] for i in order], fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel("s.d. of rotation about that axis (deg)")
    ax.set_title(f"how much each joint actually moves — frames {lo}–{hi}", color=MUTED)
    ax.legend(handles=[Line2D([], [], lw=7, color="#9a9a9a", label="dominant axis (wide bar; hue = leg pair)"),
                       Line2D([], [], lw=3.5, color="#00000055", label="second axis (narrow bar)")],
              fontsize=8, loc="lower right")
    ax = axes[1]
    for r in rows_csv:
        if r["kind"].startswith("ThC"):
            ax.scatter(r["sd1_deg"], r["axis1_vs_protraction_z"], color="#0072B2", s=34)
        elif "hinge" in r["kind"]:
            ax.scatter(r["sd1_deg"], r["axis1_vs_legplane_normal"], color="#D55E00", s=34, marker="s")
    ax.set_xlabel("s.d. of the dominant axis (deg)")
    ax.set_ylabel("|cos| vs the expected anatomical axis", fontsize=8)
    ax.set_ylim(0, 1.06)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color="#0072B2", label="ThC vs protraction axis (+z)"),
                       Line2D([], [], ls="", marker="s", color="#D55E00", label="CTr/FTi vs leg-plane normal")],
              fontsize=8)
    ax.set_title("do the dominant axes match the anatomy?", color=MUTED)
    fig.suptitle(f"Which joints move, and about which axes — {args.label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig3_axis_dominance_{args.label}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------- fig 4 & 5
    def traj_figure(fname, ax_a, ax_b, la, lb, title):
        for wi, (lo, hi) in enumerate(wins):
            fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
            for k, leg in enumerate([(1, "r"), (2, "r"), (3, "r"), (1, "l"), (2, "l"), (3, "l")]):
                ax = axes[k // 3][k % 3]
                P = FOOT[leg]
                cyc = [s for s in STEPS[leg] if lo <= s[0] <= hi]
                ax.plot(P[lo:hi + 1, ax_a], P[lo:hi + 1, ax_b], color="#00000018", lw=4, solid_capstyle="round")
                segs = []
                for i in range(len(cyc)):
                    a = cyc[i][0]
                    b = cyc[i + 1][0] if i + 1 < len(cyc) else min(hi, a + 60)
                    if b - a < 8:
                        continue
                    segs.append((a, b))
                    sh = 0.35 + 0.65 * (i / max(len(cyc) - 1, 1))
                    ax.plot(P[a:b, ax_a], P[a:b, ax_b], color=HUE[leg[0]], alpha=sh, lw=1.5,
                            ls=STYLE[leg[1]])
                if segs:
                    n = min(b - a for a, b in segs)
                    stack = np.stack([P[a:a + n, [ax_a, ax_b]] for a, b in segs])
                    ax.plot(stack[:, :, 0].mean(0), stack[:, :, 1].mean(0), color=INK, lw=2.2,
                            label=f"mean of {len(segs)} cycles")
                    ax.scatter(*stack[:, 0].mean(0), color=INK, s=34, zorder=5)
                    ax.legend(fontsize=7, loc="best")
                ax.set_title(f"{ana.LEG_LABEL[leg]}  ({len(segs)} cycles)", color=HUE[leg[0]])
                ax.set_xlabel(la); ax.set_ylabel(lb)
                ax.set_aspect("equal", adjustable="datalim")
            fig.suptitle(f"{title} — {args.label} — frames {lo}–{hi}\n"
                         f"tarsus ('{args.foot_segment}') position relative to the root joint, "
                         f"global rotation removed; % body length", fontsize=11)
            fig.tight_layout()
            fig.savefig(f"{args.out}/{fname}_{args.label}_w{wi}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

    traj_figure("fig4_tarsus_horizontal", 0, 1, "anterior  x  (%BL)", "left  y  (%BL)",
                "Foot trajectory, horizontal plane")
    traj_figure("fig5_tarsus_sagittal", 0, 2, "anterior  x  (%BL)", "dorsal  z  (%BL)",
                "Foot trajectory, sagittal plane")

    # ------------------------------------------------------------------- fig 6
    for wi, (lo, hi) in enumerate(wins):
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(12.5, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.5]})
        order = [(1, "l"), (2, "l"), (3, "l"), (1, "r"), (2, "r"), (3, "r")]
        for i, leg in enumerate(order):
            for a, b in STEPS[leg]:
                if b >= lo and a <= hi:
                    a0.barh(i, min(b, hi) - max(a, lo), left=max(a, lo), height=.55,
                            color=HUE[leg[0]], alpha=.95)
            a1.plot(np.arange(lo, hi + 1), FOOT[leg][lo:hi + 1, 0], STYLE[leg[1]],
                    color=HUE[leg[0]], lw=1.4, label=ana.LEG_LABEL[leg])
        a0.set_yticks(range(6)); a0.set_yticklabels([ana.LEG_LABEL[l] for l in order], fontsize=8)
        a0.set_title("swing phases (bars) — tripod groups appear as alternating blocks",
                     color=MUTED, pad=6)
        a0.invert_yaxis(); a0.grid(axis="y", visible=False)
        a1.set_ylabel("foot position along body axis (%BL, + anterior)")
        a1.set_xlabel("frame")
        a1.legend(fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  borderaxespad=0)
        a1.margins(y=0.12)
        fig.suptitle(f"Gait diagram — {args.label} — frames {lo}–{hi}", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{args.out}/fig6_gait_diagram_{args.label}_w{wi}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # --------------------------------------------------------------------- csv
    import csv
    with open(f"{args.out}/angles_{args.label}.csv", "w", newline="") as f:
        keys = list(next(iter(angles.values())).keys())
        w = csv.writer(f)
        w.writerow(["frame", "time_s"] + [f"{ana.LEG_LABEL[l].split()[0]}_{k}" for l in ana.LEGS for k in keys])
        for i in range(F):
            w.writerow([i, i / fps] + [f"{angles[l][k][i]:.4f}" for l in ana.LEGS for k in keys])
    with open(f"{args.out}/tarsi_{args.label}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s"] + [f"{ana.LEG_LABEL[l].split()[0]}_{c}_pctBL"
                                          for l in ana.LEGS for c in "xyz"])
        for i in range(F):
            w.writerow([i, i / fps] + [f"{FOOT[l][i, c]:.4f}" for l in ana.LEGS for c in range(3)])
    with open(f"{args.out}/axis_dominance_{args.label}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys())); w.writeheader(); w.writerows(rows_csv)
    with open(f"{args.out}/steps_{args.label}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["leg", "liftoff_frame", "touchdown_frame"])
        for leg in ana.LEGS:
            for a, b in STEPS[leg]:
                w.writerow([ana.LEG_LABEL[leg], a, b])
    print(f"wrote figures + csv to {args.out}/")


if __name__ == "__main__":
    main()
