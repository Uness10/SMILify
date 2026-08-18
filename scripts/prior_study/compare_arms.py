#!/usr/bin/env python3
"""Join the four study arms into the single before/after table the study exists to produce.

The study asks one question: **does adding authored joint limits change pose
realism and accuracy, and does the answer differ between single-view and
multi-view?** Each arm is scored independently by ``run_prior_study.sh``; this
script reads those four result folders and emits the comparison.

Arms (all scored against the SAME authored limits, including the references):

    sv_reference    the downloaded single-view checkpoint, unmodified
    sv_constrained  sv_reference + N epochs with joint_limit_regularization > 0
    mv_reference    the downloaded multi-view checkpoint, unmodified
    mv_constrained  mv_reference + N epochs with joint_limit_regularization > 0

Metrics, per arm:

  Realism (what the prior should improve)
    violating_axes      joint-axes violating their range in >=1 frame, of those authored
    mean_viol_rate      mean over authored axes of "% of frames outside the range"
    mean_overshoot_deg  mean overshoot across violating axes
    max_overshoot_deg   worst single excursion

  Accuracy (what must not get worse)
    mpjpe_mm, median_mpjpe_mm, pck_5px_native, pck_5px_input

Deltas are reported constrained - reference, so **negative is better for
violations and MPJPE, positive is better for PCK**.

Usage
-----
    python scripts/prior_study/compare_arms.py \
        --results-root prior_study_results \
        --out prior_study_results/comparison

Or name the folders explicitly:

    python scripts/prior_study/compare_arms.py \
        --arm prior_study_results/sv_reference \
        --arm prior_study_results/sv_constrained \
        --arm prior_study_results/mv_reference \
        --arm prior_study_results/mv_constrained \
        --out prior_study_results/comparison

Missing arms are reported, not fatal: a partial table still tells you something.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prior_study.analyze_baseline_pose import parse_benchmark_report  # noqa: E402

# The canonical arm names, and how they pair up for the delta rows.
PAIRS = [
    ("singleview", "sv_reference", "sv_constrained"),
    ("multiview", "mv_reference", "mv_constrained"),
]

ACC_KEYS = ["mpjpe_mm", "median_mpjpe_mm", "pck_5px_native", "pck_5px_input"]
# For each metric: True when a DECREASE is an improvement.
LOWER_IS_BETTER = {
    "mpjpe_mm": True,
    "median_mpjpe_mm": True,
    "pck_5px_native": False,
    "pck_5px_input": False,
    "violating_axes": True,
    "mean_viol_rate": True,
    "mean_overshoot_deg": True,
    "max_overshoot_deg": True,
}


def read_violations(arm_dir: Path) -> dict:
    """Summarise ``analysis/limit_violations.csv`` into four scalars.

    The CSV has one row per AUTHORED (non-free) joint-axis — ``limit_violations``
    skips axes whose range is wide open — so ``n_axes`` below is the number of
    axes the prior can actually act on, not 3 x n_joints.
    """
    out = {
        "n_axes": None,
        "violating_axes": None,
        "mean_viol_rate": None,
        "mean_overshoot_deg": None,
        "max_overshoot_deg": None,
    }
    path = arm_dir / "analysis" / "limit_violations.csv"
    if not path.is_file():
        return out

    rates, overshoots, maxima = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rate = float(row["pct_frames_violating"])
                mean_over = float(row["mean_overshoot_deg"])
                max_over = float(row["max_overshoot_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            rates.append(rate)
            overshoots.append(mean_over)
            maxima.append(max_over)

    if not rates:
        return out
    violating = [r for r in rates if r > 0.0]
    out["n_axes"] = len(rates)
    out["violating_axes"] = len(violating)
    out["mean_viol_rate"] = sum(rates) / len(rates)
    # Averaged over the axes that actually violate: averaging over all authored
    # axes would just re-encode the violation rate and hide the severity.
    viol_overshoots = [o for o, r in zip(overshoots, rates) if r > 0.0]
    out["mean_overshoot_deg"] = (sum(viol_overshoots) / len(viol_overshoots)) if viol_overshoots else 0.0
    out["max_overshoot_deg"] = max(maxima) if maxima else 0.0
    return out


def find_benchmark_report(arm_dir: Path, recorded: Optional[str]) -> Optional[Path]:
    """Locate benchmark_report.txt: the path arm.json recorded, else a glob."""
    if recorded:
        cand = Path(recorded)
        if not cand.is_absolute():
            cand = REPO_ROOT / cand
        if cand.is_file():
            return cand
    hits = sorted(arm_dir.glob("benchmark_*/benchmark_report.txt"))
    return hits[0] if hits else None


def load_arm(arm_dir: Path) -> Optional[dict]:
    if not arm_dir.is_dir():
        return None
    meta = {}
    meta_path = arm_dir / "arm.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass

    record = {
        "label": meta.get("label", arm_dir.name),
        "mode": meta.get("mode", "?"),
        "checkpoint": meta.get("checkpoint", "?"),
        "checkpoint_epoch": meta.get("checkpoint_epoch", "?"),
        "smal_file": meta.get("smal_file", "?"),
        "dir": str(arm_dir),
    }
    record.update(read_violations(arm_dir))
    report = find_benchmark_report(arm_dir, meta.get("benchmark_report"))
    record["benchmark_report"] = str(report) if report else None
    record.update(parse_benchmark_report(report))
    return record


def fmt(value, spec: str = ".2f") -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return format(value, spec)


def delta(constrained, reference, key: str, spec: str = ".2f") -> tuple[Optional[float], str]:
    """Return ``(delta, annotated_string)`` for one metric.

    ``spec`` must match the precision the metric is *reported* at: PCK sits
    around 0.9 and moves in the third decimal, so formatting its delta at 2dp
    prints a misleading "+0.00" for a change that is actually visible.

    The verdict reflects whether the change is an improvement for THAT metric,
    which is not the same as the sign: PCK going up is good, MPJPE going up is not.
    """
    a = constrained.get(key) if constrained else None
    b = reference.get(key) if reference else None
    if a is None or b is None:
        return None, "—"
    d = a - b
    if abs(d) < 1e-9:
        return d, "0 (=)"
    improved = (d < 0) if LOWER_IS_BETTER.get(key, True) else (d > 0)
    return d, f"{d:+{spec}} ({'better' if improved else 'worse'})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="prior_study_results", help="Folder holding the per-arm result dirs")
    p.add_argument("--arm", action="append", default=[], help="Explicit arm directory (repeatable)")
    p.add_argument("--out", default="prior_study_results/comparison", help="Output directory")
    p.add_argument("--extra-epochs", type=int, default=None, help="Recorded in the report header for provenance")
    p.add_argument("--joint-limit-weight", type=float, default=None, help="Recorded in the report header")
    args = p.parse_args()

    root = Path(args.results_root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    arms: dict[str, dict] = {}
    if args.arm:
        for a in args.arm:
            d = Path(a) if Path(a).is_absolute() else REPO_ROOT / a
            rec = load_arm(d)
            if rec:
                arms[rec["label"]] = rec
            else:
                print(f"[compare] missing arm directory: {d}", file=sys.stderr)
    else:
        for _mode, ref, con in PAIRS:
            for name in (ref, con):
                rec = load_arm(root / name)
                if rec:
                    arms[name] = rec

    if not arms:
        print(
            f"[compare] ERROR: no arm directories found under {root}.\n"
            f"          Run run_prior_study.sh once per arm first (labels: "
            f"{', '.join(n for _m, r, c in PAIRS for n in (r, c))}).",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- flat CSV: one row per arm -----------------------------------------
    csv_fields = [
        "label",
        "mode",
        "checkpoint",
        "checkpoint_epoch",
        "n_axes",
        "violating_axes",
        "mean_viol_rate",
        "mean_overshoot_deg",
        "max_overshoot_deg",
        *ACC_KEYS,
        "smal_file",
        "benchmark_report",
    ]
    with open(out_dir / "arms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for _mode, ref, con in PAIRS:
            for name in (ref, con):
                if name in arms:
                    w.writerow(arms[name])
        for name, rec in arms.items():
            if all(name not in (r, c) for _m, r, c in PAIRS):
                w.writerow(rec)

    # ---- markdown report ----------------------------------------------------
    lines = ["# Joint-limit prior — single-view vs multi-view", ""]
    header_bits = []
    if args.extra_epochs is not None:
        header_bits.append(f"fine-tune: **+{args.extra_epochs} epochs**")
    if args.joint_limit_weight is not None:
        header_bits.append(f"`joint_limit_regularization` = **{args.joint_limit_weight}**")
    if header_bits:
        lines += ["  |  ".join(header_bits), ""]
    lines += [
        "Each `*_constrained` arm is its `*_reference` checkpoint continued for the same number of "
        "epochs with the limit penalty enabled. **All four arms are scored against the same authored "
        "ranges**, so the reference rows are the honest \"before\" number.",
        "",
        "> **Caveat — the fine-tuning confound.** The reference arms received zero additional epochs, "
        "so any difference below is *the prior plus continued training*, not the prior alone. "
        "A `w_limit = 0` control fine-tuned for the same epochs would separate the two; it is not "
        "part of this study by design.",
        "",
        "Deltas are `constrained - reference`. Lower is better for violations and MPJPE; "
        "higher is better for PCK.",
        "",
    ]

    for mode, ref_name, con_name in PAIRS:
        ref, con = arms.get(ref_name), arms.get(con_name)
        lines += [f"## {mode}", ""]
        if not ref and not con:
            lines += [f"_Neither `{ref_name}` nor `{con_name}` found._", ""]
            continue
        if not ref or not con:
            have = ref_name if ref else con_name
            lines += [f"_Only `{have}` present — no delta can be computed._", ""]

        n_axes = (con or ref).get("n_axes")
        lines += [
            f"Authored axes scored: **{n_axes if n_axes is not None else '—'}**",
            "",
            "| metric | reference | constrained | delta |",
            "|---|---|---|---|",
        ]
        rows = [
            ("Violating axes (count)", "violating_axes", ".0f"),
            ("Mean violation rate (% frames)", "mean_viol_rate", ".2f"),
            ("Mean overshoot, violating axes (deg)", "mean_overshoot_deg", ".2f"),
            ("Max overshoot (deg)", "max_overshoot_deg", ".2f"),
            ("MPJPE (mm)", "mpjpe_mm", ".2f"),
            ("Median MPJPE (mm)", "median_mpjpe_mm", ".2f"),
            ("PCK@5px native", "pck_5px_native", ".4f"),
            ("PCK@5px input", "pck_5px_input", ".4f"),
        ]
        for title, key, spec in rows:
            _d, annotated = delta(con, ref, key, spec)
            lines.append(
                f"| {title} | {fmt(ref.get(key) if ref else None, spec)} | "
                f"{fmt(con.get(key) if con else None, spec)} | {annotated} |"
            )
        lines.append("")
        for name, rec in ((ref_name, ref), (con_name, con)):
            if rec:
                lines.append(f"- `{name}`: {rec['checkpoint']} (epoch {rec['checkpoint_epoch']})")
        lines.append("")

    # ---- cross-modality read ------------------------------------------------
    lines += ["## Single-view vs multi-view", ""]
    sv_ref, sv_con = arms.get("sv_reference"), arms.get("sv_constrained")
    mv_ref, mv_con = arms.get("mv_reference"), arms.get("mv_constrained")
    if all((sv_ref, sv_con, mv_ref, mv_con)):
        lines += [
            "| metric | single-view delta | multi-view delta |",
            "|---|---|---|",
        ]
        for title, key, spec in [
            ("Violating axes (count)", "violating_axes", ".0f"),
            ("Mean violation rate (% frames)", "mean_viol_rate", ".2f"),
            ("Mean overshoot (deg)", "mean_overshoot_deg", ".2f"),
            ("MPJPE (mm)", "mpjpe_mm", ".2f"),
            ("PCK@5px native", "pck_5px_native", ".4f"),
        ]:
            _sd, sv_txt = delta(sv_con, sv_ref, key, spec)
            _md, mv_txt = delta(mv_con, mv_ref, key, spec)
            lines.append(f"| {title} | {sv_txt} | {mv_txt} |")
        lines += [
            "",
            "The hypothesis worth testing here: multi-view already resolves depth ambiguity from "
            "geometry, so it should have fewer violations to fix and less to gain from the prior. "
            "A single-view delta noticeably larger than the multi-view one supports that reading.",
            "",
        ]
    else:
        missing = [n for n in ("sv_reference", "sv_constrained", "mv_reference", "mv_constrained") if n not in arms]
        lines += [f"_Incomplete: missing {', '.join(f'`{m}`' for m in missing)}._", ""]

    (out_dir / "comparison.md").write_text("\n".join(lines))

    print(f"[compare] arms found: {', '.join(sorted(arms))}")
    print(f"[compare] wrote {out_dir / 'comparison.md'}")
    print(f"[compare] wrote {out_dir / 'arms.csv'}")
    for name, rec in sorted(arms.items()):
        if rec.get("mean_viol_rate") is None:
            print(
                f"[compare] WARNING: {name} has no limit_violations.csv — it was analysed against a "
                f".pkl with no joint_limits ({rec.get('smal_file')}). Re-run that arm with "
                f"SMAL_FILE=3D_model_prep/SMILy_STICK_limits_authored.pkl.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
