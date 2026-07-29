#!/usr/bin/env python3
"""Baseline pose analysis for the unconstrained neural stick-insect model.

Purpose (supervisor's brief): quantify *how* and *in what form* benefits would
arise from adding a user-defined joint-angle prior. This script characterises the
CURRENT (unconstrained) model so its output drops straight into a side-by-side
comparison with a later constrained run.

It consumes the artifacts already produced by the existing pipeline:
  * the animation clip written by ``run_multiview_inference --export_animation``
    (``<stem>.npz`` + ``<stem>.json``), and
  * (optionally) the ``benchmark_report.txt`` written by ``benchmark_model``.

Verified on-disk schema (do not assume, these were probed against the code):
  .npz keys : poses (F, J, 3) axis-angle [joint 0 = global root rot],
              trans (F, 3), betas (B,), betas_per_frame (F, B), fps,
              optional log_beta_scales / betas_trans / mesh_scale
  .json     : joint_names (list, len J), parents, n_joints, n_betas,
              rotation_representation == "axis_angle", root_joint_index == 0
  limits    : (J, 3, 2) radians, per-axis [min, max] in the SAME axis-angle space
              as ``poses`` (matches smal_fitter/priors/joint_limits_prior.py and
              the joint_limit_regularization hinge loss). Root joint fixed [0,0].

NOTE on the comparison axis: authored limits are defined per axis-angle *component*,
and the training-time hinge loss clamps each component independently. So per-axis
comparison of poses[:, j, axis] against limits[j, axis] is the correct apples-to-apples
measure against the prior. We ALSO report rotation magnitude ||aa|| per joint as a
representation-robust intuition for "how much this joint moves".

Author: baseline study scaffold. Safe to run with --self-test (no real data needed).
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless / cluster safe
import matplotlib.pyplot as plt  # noqa: E402

RAD2DEG = 180.0 / np.pi

# Default "important" joints = the six leg tips (pretarsus), mirroring the
# joint_importance block in multiview_sticks_UNET_optimal.json.
DEFAULT_IMPORTANT = ["l_1_pt_l", "l_2_pt_l", "l_3_pt_l", "l_1_pt_r", "l_2_pt_r", "l_3_pt_r"]

AXES = ["x", "y", "z"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_animation(npz_path: Path, json_path: Optional[Path]) -> Dict:
    """Load the exported clip. Returns a dict with poses/trans/joint_names/fps/parents."""
    data = np.load(npz_path)
    poses = np.asarray(data["poses"], dtype=np.float64)  # (F, J, 3)
    if poses.ndim != 3 or poses.shape[2] != 3:
        raise ValueError(f"Expected poses of shape (F, J, 3), got {poses.shape}")
    trans = np.asarray(data["trans"], dtype=np.float64) if "trans" in data else None
    fps = float(data["fps"]) if "fps" in data.files else 60.0

    joint_names: Optional[List[str]] = None
    parents: Optional[List[int]] = None
    if json_path is not None and json_path.exists():
        side = json.loads(Path(json_path).read_text())
        joint_names = side.get("joint_names")
        parents = side.get("parents")
        rep = side.get("rotation_representation", "axis_angle")
        if rep != "axis_angle":
            print(
                f"[warn] sidecar rotation_representation='{rep}', expected 'axis_angle'. "
                "Per-axis limit comparison assumes axis-angle.",
                file=sys.stderr,
            )
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(poses.shape[1])]
    if len(joint_names) != poses.shape[1]:
        raise ValueError(f"joint_names length {len(joint_names)} != poses J dim {poses.shape[1]}")

    return {
        "poses": poses,
        "trans": trans,
        "fps": fps,
        "joint_names": list(joint_names),
        "parents": parents,
        "n_frames": poses.shape[0],
        "n_joints": poses.shape[1],
    }


def load_limits(
    smal_file: Optional[Path],
    limits_arg: Optional[Path],
    n_joints: int,
    joint_names: List[str],
) -> Tuple[Optional[np.ndarray], str]:
    """Return (limits (J,3,2) or None, source_description).

    Priority: explicit --limits file > joint_limits key inside the model .pkl.
    A wide-open +/- pi table is treated as "no meaningful prior" and reported as such
    (still returned, so violation counts read as ~0 and make the point visible).
    """
    limits = None
    source = "none"

    if limits_arg is not None and Path(limits_arg).exists():
        p = Path(limits_arg)
        if p.suffix == ".npy":
            limits = np.load(p, allow_pickle=True).astype(np.float64)
            source = f"file:{p.name}"
        elif p.suffix == ".json":
            d = json.loads(p.read_text())  # {joint_name: [[min,max]x3]}
            limits = np.array([d[jn] for jn in joint_names], dtype=np.float64)
            source = f"file:{p.name}"
    elif smal_file is not None and Path(smal_file).exists():
        with open(smal_file, "rb") as f:
            dd = pickle.load(f, encoding="latin1")
        jl = dd.get("joint_limits", None) if hasattr(dd, "get") else None
        if jl is not None:
            limits = np.asarray(jl, dtype=np.float64)
            source = f"pkl:{Path(smal_file).name}:joint_limits"

    if limits is not None:
        if limits.shape != (n_joints, 3, 2):
            raise ValueError(f"limits shape {limits.shape}, expected {(n_joints, 3, 2)}")
        # Detect a wide-open (no real prior) table.
        widths = limits[..., 1] - limits[..., 0]
        non_root = widths[1:]
        if np.all(non_root >= (2 * np.pi - 1e-3)):
            source += " (WIDE-OPEN / no effective prior)"

    return limits, source


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def per_axis_stats(poses: np.ndarray, joint_names: List[str]) -> List[Dict]:
    """Per joint per axis distribution stats over frames (excludes root joint 0)."""
    rows = []
    for j in range(1, poses.shape[1]):  # skip root
        for a in range(3):
            v = poses[:, j, a]
            rows.append(
                {
                    "joint": joint_names[j],
                    "joint_idx": j,
                    "axis": AXES[a],
                    "min_rad": float(np.min(v)),
                    "max_rad": float(np.max(v)),
                    "mean_rad": float(np.mean(v)),
                    "std_rad": float(np.std(v)),
                    "rom_rad": float(np.max(v) - np.min(v)),
                    "min_deg": float(np.min(v) * RAD2DEG),
                    "max_deg": float(np.max(v) * RAD2DEG),
                    "rom_deg": float((np.max(v) - np.min(v)) * RAD2DEG),
                }
            )
    return rows


def magnitude_stats(poses: np.ndarray, joint_names: List[str]) -> List[Dict]:
    """Rotation magnitude ||axis-angle|| per joint (representation-robust ROM)."""
    mag = np.linalg.norm(poses, axis=2)  # (F, J)
    rows = []
    for j in range(1, poses.shape[1]):
        v = mag[:, j]
        rows.append(
            {
                "joint": joint_names[j],
                "joint_idx": j,
                "mean_mag_deg": float(np.mean(v) * RAD2DEG),
                "max_mag_deg": float(np.max(v) * RAD2DEG),
                "std_mag_deg": float(np.std(v) * RAD2DEG),
            }
        )
    return rows


def limit_violations(poses: np.ndarray, joint_names: List[str], limits: np.ndarray) -> List[Dict]:
    """Per joint per axis: how far the unconstrained model strays outside the authored range."""
    F = poses.shape[0]
    rows = []
    for j in range(1, poses.shape[1]):
        for a in range(3):
            v = poses[:, j, a]
            lo, hi = limits[j, a, 0], limits[j, a, 1]
            under = np.maximum(lo - v, 0.0)
            over = np.maximum(v - hi, 0.0)
            n_viol = int(np.sum((under > 0) | (over > 0)))
            overshoot = np.maximum(under, over)
            rows.append(
                {
                    "joint": joint_names[j],
                    "axis": AXES[a],
                    "limit_lo_deg": float(lo * RAD2DEG),
                    "limit_hi_deg": float(hi * RAD2DEG),
                    "pct_frames_violating": 100.0 * n_viol / max(F, 1),
                    "mean_overshoot_deg": float(np.mean(overshoot) * RAD2DEG),
                    "max_overshoot_deg": float(np.max(overshoot) * RAD2DEG),
                }
            )
    return rows


def parse_benchmark_report(path: Optional[Path]) -> Dict[str, Optional[float]]:
    """Best-effort scrape of MPJPE / PCK from benchmark_report.txt."""
    out: Dict[str, Optional[float]] = {
        "mpjpe_mm": None,
        "median_mpjpe_mm": None,
        "pck_5px_native": None,
        "pck_5px_input": None,
    }
    if path is None or not Path(path).exists():
        return out
    text = Path(path).read_text(errors="ignore")

    def grab(pattern: str) -> Optional[float]:
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    out["mpjpe_mm"] = grab(r"MPJPE\s*\(mm\)\s*:?\s*([\d.]+)")
    out["median_mpjpe_mm"] = grab(r"Median\s*MPJPE\s*\(mm\)\s*:?\s*([\d.]+)")
    # PCK lines vary; capture the first @5px native/input if present.
    out["pck_5px_native"] = grab(r"native.*?PCK@5(?:px)?\s*:?\s*([\d.]+)")
    out["pck_5px_input"] = grab(r"input.*?PCK@5(?:px)?\s*:?\s*([\d.]+)")
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_angle_distributions(poses, joint_names, important, limits, out_path, label):
    idx = [joint_names.index(j) for j in important if j in joint_names]
    if not idx:
        idx = list(range(1, min(7, poses.shape[1])))
        important = [joint_names[i] for i in idx]
    n = len(idx)
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.4 * n), squeeze=False)
    for r, j in enumerate(idx):
        for a in range(3):
            ax = axes[r][a]
            v = poses[:, j, a] * RAD2DEG
            ax.hist(v, bins=40, color="#3b6ea5", alpha=0.85)
            if limits is not None:
                lo, hi = limits[j, a, 0] * RAD2DEG, limits[j, a, 1] * RAD2DEG
                ax.axvline(lo, color="#c0392b", ls="--", lw=1.2)
                ax.axvline(hi, color="#c0392b", ls="--", lw=1.2)
            if a == 0:
                ax.set_ylabel(joint_names[j], fontsize=8)
            if r == 0:
                ax.set_title(f"axis {AXES[a]} (deg)", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Joint-angle distributions [{label}] — dashed red = authored limits", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_range_of_motion(mag_rows, out_path, label):
    names = [r["joint"] for r in mag_rows]
    vals = [r["max_mag_deg"] for r in mag_rows]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(names))))
    ax.barh(names, vals, color="#2e8b57")
    ax.set_xlabel("Max rotation magnitude ||axis-angle|| (deg)")
    ax.set_title(f"Per-joint range of motion [{label}]")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(anim, important, out_dir, label):
    poses, trans, fps = anim["poses"], anim["trans"], anim["fps"]
    joint_names = anim["joint_names"]
    t = np.arange(anim["n_frames"]) / max(fps, 1e-6)

    # 1) global translation over time
    if trans is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        for a, lab in enumerate(["x", "y", "z"]):
            ax.plot(t, trans[:, a], label=f"trans {lab}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("translation")
        ax.set_title(f"Global body translation over time [{label}]")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_translation.png", dpi=150)
        plt.close(fig)

    # 2) important-joint angle-vs-time (magnitude)
    idx = [joint_names.index(j) for j in important if j in joint_names]
    if idx:
        mag = np.linalg.norm(poses, axis=2) * RAD2DEG
        fig, ax = plt.subplots(figsize=(9, 4))
        for j in idx:
            ax.plot(t, mag[:, j], label=joint_names[j], lw=1.0)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rotation magnitude (deg)")
        ax.set_title(f"Leg-tip joint angle over time [{label}]")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_legtip_angles.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_csv(rows: List[Dict], path: Path, label: str):
    if not rows:
        return
    fields = ["label"] + list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"label": label, **r})


def write_summary(out_dir: Path, label: str, anim, axis_rows, viol_rows, acc, limits_source):
    F = anim["n_frames"]
    lines = [f"# Baseline pose study — `{label}`", ""]
    lines += [
        f"- Frames analysed: **{F}**  |  joints: **{anim['n_joints']}**  |  fps: {anim['fps']:.1f}",
        f"- Authored-limits source: `{limits_source}`",
        "",
        "## Accuracy (from benchmark_report.txt)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| MPJPE (mm) | {acc['mpjpe_mm']} |",
        f"| Median MPJPE (mm) | {acc['median_mpjpe_mm']} |",
        f"| PCK@5px (native) | {acc['pck_5px_native']} |",
        f"| PCK@5px (input) | {acc['pck_5px_input']} |",
        "",
    ]
    if viol_rows:
        viol_sorted = sorted(viol_rows, key=lambda r: r["pct_frames_violating"], reverse=True)
        n_any = sum(1 for r in viol_rows if r["pct_frames_violating"] > 0)
        lines += [
            "## Limit violations (unconstrained model vs authored ranges)",
            "",
            f"- Joint-axes violating the prior at least once: **{n_any} / {len(viol_rows)}**",
            "",
            "Top 10 most-violated joint axes:",
            "",
            "| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |",
            "|---|---|---|---|---|",
        ]
        for r in viol_sorted[:10]:
            lines.append(
                f"| {r['joint']} | {r['axis']} | {r['pct_frames_violating']:.1f} | "
                f"{r['mean_overshoot_deg']:.1f} | {r['max_overshoot_deg']:.1f} |"
            )
        lines.append("")
    else:
        lines += ["## Limit violations", "", "_No authored limits available — see finding below._", ""]
    # Widest ROM joints
    rom_sorted = sorted(axis_rows, key=lambda r: r["rom_deg"], reverse=True)[:10]
    lines += [
        "## Widest range of motion (per axis)",
        "",
        "| joint | axis | ROM (deg) | min (deg) | max (deg) |",
        "|---|---|---|---|---|",
    ]
    for r in rom_sorted:
        lines.append(f"| {r['joint']} | {r['axis']} | {r['rom_deg']:.1f} | {r['min_deg']:.1f} | {r['max_deg']:.1f} |")
    lines.append("")
    (out_dir / "baseline_summary.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Self-test (synthetic data)
# --------------------------------------------------------------------------- #
def make_synthetic(tmp: Path) -> Tuple[Path, Path, Path]:
    """Fabricate a small stick-like clip + authored limits so the whole pipeline
    can be validated without the 21GB dataset or a GPU."""
    rng = np.random.default_rng(0)
    J = 55
    F = 300
    names = ["b_t"] + [f"j_{i}" for i in range(1, J)]
    # Make the six 'leg tip' names real so important-joint selection exercises.
    for k, nm in enumerate(DEFAULT_IMPORTANT):
        names[1 + k] = nm
    poses = rng.normal(0, 0.3, size=(F, J, 3)).astype(np.float32)
    poses[:, 0, :] = 0.0  # root fixed
    # Inject deliberate out-of-range excursions on a couple of joints.
    poses[:, 1, 1] += 1.4
    trans = np.cumsum(rng.normal(0, 0.01, size=(F, 3)), axis=0).astype(np.float32)
    npz = tmp / "synthetic_clip.npz"
    np.savez(
        npz,
        poses=poses,
        trans=trans,
        betas=np.zeros(10, np.float32),
        betas_per_frame=np.zeros((F, 10), np.float32),
        fps=np.float32(60.0),
    )
    side = {
        "joint_names": names,
        "parents": [-1] + list(range(0, J - 1)),
        "n_joints": J,
        "n_betas": 10,
        "rotation_representation": "axis_angle",
        "root_joint_index": 0,
    }
    js = tmp / "synthetic_clip.json"
    js.write_text(json.dumps(side))
    limits = np.zeros((J, 3, 2), np.float64)
    limits[:, :, 0] = -0.8
    limits[:, :, 1] = 0.8
    limits[0] = 0.0
    lim = tmp / "synthetic_limits.npy"
    np.save(lim, limits)
    return npz, js, lim


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(npz, json_path, smal_file, limits_arg, benchmark, label, important, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    anim = load_animation(Path(npz), Path(json_path) if json_path else None)
    limits, limits_source = load_limits(
        Path(smal_file) if smal_file else None,
        Path(limits_arg) if limits_arg else None,
        anim["n_joints"],
        anim["joint_names"],
    )

    axis_rows = per_axis_stats(anim["poses"], anim["joint_names"])
    mag_rows = magnitude_stats(anim["poses"], anim["joint_names"])
    viol_rows = limit_violations(anim["poses"], anim["joint_names"], limits) if limits is not None else []
    acc = parse_benchmark_report(Path(benchmark) if benchmark else None)

    write_csv(axis_rows, out_dir / "per_axis_stats.csv", label)
    write_csv(mag_rows, out_dir / "magnitude_stats.csv", label)
    if viol_rows:
        write_csv(viol_rows, out_dir / "limit_violations.csv", label)

    plot_angle_distributions(
        anim["poses"], anim["joint_names"], important, limits, out_dir / "joint_angle_distributions.png", label
    )
    plot_range_of_motion(mag_rows, out_dir / "range_of_motion.png", label)
    plot_trajectories(anim, important, traj_dir, label)

    write_summary(out_dir, label, anim, axis_rows, viol_rows, acc, limits_source)

    print(f"[ok] baseline analysis written to {out_dir}")
    print(f"     limits source: {limits_source}")
    if not viol_rows:
        print(
            "     [finding] no authored joint_limits found -> violation table skipped. "
            "Author limits (see README) or pass --limits to enable the comparison."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=str, help="Exported animation .npz (from --export_animation)")
    ap.add_argument("--json", type=str, default=None, help="Sidecar .json (defaults to <npz stem>.json)")
    ap.add_argument("--smal-file", type=str, default=None, help="Model .pkl (read joint_limits if present)")
    ap.add_argument("--limits", type=str, default=None, help="Authored limits override (.npy (J,3,2) or .json ranges)")
    ap.add_argument("--benchmark", type=str, default=None, help="benchmark_report.txt for accuracy scrape")
    ap.add_argument("--label", type=str, default="unconstrained", help="Column label for comparison tables")
    ap.add_argument(
        "--important-joints",
        type=str,
        default=",".join(DEFAULT_IMPORTANT),
        help="Comma-separated joints to feature in distribution/trajectory plots",
    )
    ap.add_argument("--out", type=str, default="prior_study_baseline", help="Output directory")
    ap.add_argument("--self-test", action="store_true", help="Run on synthetic data (no real dataset needed)")
    args = ap.parse_args()

    important = [j.strip() for j in args.important_joints.split(",") if j.strip()]

    if args.self_test:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="prior_study_selftest_"))
        npz, js, lim = make_synthetic(tmp)
        run(npz, js, None, lim, None, "selftest_synthetic", important, args.out)
        print(f"[self-test] synthetic inputs in {tmp}")
        return

    if not args.npz:
        ap.error("--npz is required (or use --self-test)")
    json_path = args.json or str(Path(args.npz).with_suffix(".json"))
    run(args.npz, json_path, args.smal_file, args.limits, args.benchmark, args.label, important, args.out)


if __name__ == "__main__":
    main()
