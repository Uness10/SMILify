"""Regenerate the report figures with the epoch-matched lambda=0 controls.

Run from anywhere:  python report_joint_limit_prior/make_matched_figures.py
Reads   prior_study_results/, prior_study_results_matched/ and the
        benchmark_*_reference_cont_* directories at the repository root.
Writes  fig_*.pdf / fig_*.png (single-view) and fig_mv_*.pdf / .png
        (multi-view) into report_joint_limit_prior/figures/.

The epoch-matched control (sv-ref / mv-ref) is the lambda = 0 point in every
figure. The starting checkpoint (sv-base / mv-base) is drawn as an open grey
marker where it is shown at all.

Not regenerated: fig_pck.pdf and fig_mpjpe_quantiles.pdf (single-view), whose
constrained-arm benchmark reports are not in this repository.
"""
import os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "prior_study_results")
MAT = os.path.join(REPO, "prior_study_results_matched")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE = "#1f77b4"; ORANGE = "#d95f02"; GREY = "#7f7f7f"; BLACK = "0.15"
SHADES = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
LAMK = ["1e-4", "1e-3", "1e-2", "1e-1"]
LAMLBL = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
XTL = ["0"] + LAMLBL
BUCKETS = ["never\n0%", "rare\n0–1%", "some\n1–10%", "chronic\n>10%"]

# Accuracy of the arms (test split), from the benchmark reports / eval logs.
ACC = {
    "sv": {"base": 1.0920, "ref": 0.9054, "1e-4": 0.9718, "1e-3": 1.0075, "1e-2": 1.0232, "1e-1": 1.0479},
    "mv": {"base": 0.9639, "ref": 0.9042, "1e-4": 0.9920, "1e-3": 1.0750, "1e-2": 1.1414, "1e-1": 1.2111},
}
EPOCH = {"sv": ("386", "433", "435"), "mv": ("345", "369", "369")}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def parse_bench(p):
    body = open(p).read().split("==== BENCHMARK RESULTS")[1]
    out = {"pck": {}, "pct": {}}
    for ch in body.split("-- PCK @ ")[1:]:
        key = "native" if ch.startswith("native") else "input"
        curve = ch.split("MPJPE (mm)")[0].split("PCK curve:")[1]
        out["pck"][key] = {int(m.group(1)): float(m.group(2))
                           for m in re.finditer(r"PCK@(\d+)px: ([\d.]+)", curve)}
    for m in re.finditer(r"P(\d+): ([\d.]+)", body.split("MPJPE percentiles")[1]):
        out["pct"][int(m.group(1))] = float(m.group(2))
    return out


def load(mode):
    d = {"base": f"{RES}/{mode}_reference/analysis",
         "ref": f"{MAT}/{mode}_reference/analysis"}
    d.update({k: f"{RES}/lam{k}/{mode}_constrained/analysis" for k in LAMK})
    pa = {k: pd.read_csv(f"{v}/per_axis_stats.csv").set_index(["joint", "axis"]) for k, v in d.items()}
    lv = {k: pd.read_csv(f"{v}/limit_violations.csv").set_index(["joint", "axis"]) for k, v in d.items()}
    ref = lv["ref"].copy()
    span_all = ref.limit_hi_deg - ref.limit_lo_deg
    B = span_all[span_all < 359].index                     # 129 bounded axes
    S = dict(pa=pa, lv=lv, B=B, span=span_all.loc[B])
    lo, hi = ref.loc[B, "limit_lo_deg"], ref.loc[B, "limit_hi_deg"]
    S["rom"] = {k: pa[k].loc[B, "rom_deg"] for k in d}
    S["clip"] = np.maximum(np.minimum(pa["ref"].loc[B, "max_deg"], hi)
                           - np.maximum(pa["ref"].loc[B, "min_deg"], lo), 0.0)
    S["pv"] = ref.loc[B, "pct_frames_violating"]
    S["rate"] = {k: lv[k].pct_frames_violating.mean() for k in d}
    S["maxov"] = {k: lv[k].max_overshoot_deg.max() for k in d}
    return S


def fig_tradeoff(mode, S, name):
    K = ["ref"] + LAMK
    eb, er, ec = EPOCH[mode]
    series = [([S["rate"][k] for k in K], S["rate"]["base"], "Mean violation rate (%)", True),
              ([S["maxov"][k] for k in K], S["maxov"]["base"], "Maximum overshoot (deg)", False),
              ([ACC[mode][k] for k in K], ACC[mode]["base"], "MPJPE (mm)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    x = np.arange(5)
    for ax, (y, yb, lab, logy) in zip(axes, series):
        ax.plot(x, y, "-", color=BLUE, lw=2, zorder=2)
        ax.plot(x[1:], y[1:], "o", color=BLUE, ms=5, zorder=3,
                label=f"constrained arms (epoch {ec})")
        ax.plot(x[0], y[0], "s", color=BLACK, ms=6, zorder=4,
                label=rf"epoch-matched control, $\lambda = 0$ (epoch {er})")
        ax.plot(x[0], yb, "o", mfc="white", mec=GREY, mew=1.6, ms=8, zorder=4,
                label=f"starting checkpoint (epoch {eb})")
        if logy: ax.set_yscale("log")
        else: ax.margins(y=0.12)
        ax.set_xticks(x); ax.set_xticklabels(XTL); ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    save(fig, name)


def fig_worst(S, name, floor):
    lv = S["lv"]
    worst = lv["ref"].loc[S["B"]].sort_values("pct_frames_violating", ascending=False).head(12).index
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    w = 0.19; xs = np.arange(len(worst))
    ax.bar(xs - 2.0 * w, np.maximum(lv["ref"].loc[worst, "pct_frames_violating"], floor), w,
           color="white", edgecolor=GREY, lw=1.1, label=r"$\lambda = 0$ (control)")
    for i, k in enumerate(LAMK):
        ax.bar(xs + (i - 1.0) * w, np.maximum(lv[k].loc[worst, "pct_frames_violating"], floor), w,
               color=SHADES[i], label=rf"$\lambda = 10^{{-{4-i}}}$")
    ax.set_yscale("log"); ax.set_ylim(floor, 300)
    ax.set_ylabel("frames violating (%)")
    ax.set_xticks(xs); ax.set_xticklabels([f"{j}.{a}" for j, a in worst], rotation=35, ha="right", fontsize=8.5)
    ax.set_xlabel("the twelve axes most often violated by the epoch-matched control")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    save(fig, name)


def fig_retention(S, name, pooled):
    """pooled=True: sum(ROM)/sum(ROM_ref) per bucket (single-view report);
    pooled=False: mean of per-axis ratios (multi-view report)."""
    rom, clip, pv = S["rom"], S["clip"], S["pv"]
    bucket = pd.cut(pv, [-1e-9, 1e-9, 1.0, 10.0, 100.1], labels=BUCKETS)
    cnt = bucket.value_counts().reindex(BUCKETS)
    if pooled:
        agg = lambda a, b: a.groupby(bucket, observed=False).sum() / b.groupby(bucket, observed=False).sum()
    else:
        agg = lambda a, b: (a / b).groupby(bucket, observed=False).mean()
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    w = 0.2; xs = np.arange(4)
    for i, k in enumerate(LAMK):
        ax.bar(xs + (i - 1.5) * w, agg(rom[k], rom["ref"]).reindex(BUCKETS).values * 100, w,
               color=SHADES[i], label=rf"$\lambda = 10^{{-{4-i}}}$")
    cb = agg(clip, rom["ref"]).reindex(BUCKETS).values * 100
    for j, v in enumerate(cb):
        ax.plot([xs[j] - 2 * w, xs[j] + 2 * w], [v, v], "--", color=BLACK, lw=2.2,
                label="clipping bound" if j == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n({c} axes)" for l, c in zip(BUCKETS, cnt.values)])
    ax.set_xlabel("violation rate of the axis in the epoch-matched control")
    ax.set_ylabel("ROM retained (% of control)")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[1:] + h[:1], l[1:] + l[:1], ncol=5, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    save(fig, name)


def fig_scatter(S, name):
    rom, clip, pv = S["rom"], S["clip"], S["pv"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, k, lbl in zip(axes, ["1e-4", "1e-1"], [r"$10^{-4}$", r"$10^{-1}$"]):
        sc = ax.scatter(rom["ref"], rom[k], c=pv, cmap="YlOrRd", s=26, ec="0.3", lw=0.4,
                        vmin=0, vmax=100, zorder=3)
        for x0, c0 in zip(rom["ref"], clip):
            if c0 > 0:
                ax.plot([x0 * 0.94, x0 * 1.06], [c0, c0], "-", color="0.55", lw=0.8, zorder=1)
        lim = [3, 200]
        ax.plot(lim, lim, "--", color=GREY, lw=1.2, zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(r"control ROM (deg), $\lambda = 0$")
        ax.set_ylabel(rf"ROM (deg) at $\lambda = ${lbl}")
        ax.set_title(rf"$\lambda = ${lbl}")
    cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("control violation rate (% of frames)")
    save(fig, name)


def fig_ratio(S, name):
    rom, span = S["rom"], S["span"]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    bins = np.linspace(0, 4, 41)
    ax.hist(np.clip(rom["ref"] / span, 0, 3.999), bins=bins, color="white", ec=GREY, lw=1.2,
            label=r"$\lambda = 0$ (control)")
    for i, k in enumerate(["1e-4", "1e-1"]):
        ax.hist(np.clip(rom[k] / span, 0, 3.999), bins=bins, color=SHADES[i * 3], alpha=0.75,
                label=rf"$\lambda = 10^{{-{4 if i == 0 else 1}}}$")
    ax.axvline(1.0, color=BLACK, ls="--", lw=1.8)
    ax.text(1.03, ax.get_ylim()[1] * 0.93, "authored range", fontsize=9, color=BLACK)
    ax.set_xlabel("observed ROM / authored range (last bin includes values above 4)")
    ax.set_ylabel("number of axes (of 129)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, name)


# ------------------------------------------------------------ single-view
sv = load("sv")
fig_tradeoff("sv", sv, "fig_tradeoff")
fig_worst(sv, "fig_worst_axes", 2e-2)
fig_retention(sv, "fig_rom_retention", pooled=True)
fig_scatter(sv, "fig_rom_scatter")
fig_ratio(sv, "fig_rom_ratio")

# ------------------------------------------------------------ multi-view
mv = load("mv")
fig_tradeoff("mv", mv, "fig_mv_tradeoff")
fig_worst(mv, "fig_mv_worst_axes", 2e-3)
fig_retention(mv, "fig_mv_rom_retention", pooled=False)
fig_scatter(mv, "fig_mv_rom_scatter")
fig_ratio(mv, "fig_mv_rom_ratio")

BENCH = {"base": f"{RES}/mv_reference/benchmark_multiview_mv_reference_on_SMILySTICKS_centred_reprojected_FIXED/benchmark_report.txt",
         "ref": f"{REPO}/benchmark_multiview_mv_reference_cont_on_SMILySTICKS_centred_reprojected_FIXED/benchmark_report.txt"}
BENCH.update({k: f"{RES}/lam{k}/mv_constrained/benchmark_multiview_mv_constrained_lam{k}_on_SMILySTICKS_centred_reprojected_FIXED/benchmark_report.txt" for k in LAMK})
bench = {k: parse_bench(v) for k, v in BENCH.items()}

# fig_mv_pck
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))
for ax, res, title in zip(axes, ["native", "input"],
                          ["native override 1530×1530", "network input 224 px"]):
    th = sorted(bench["ref"]["pck"][res])
    ax.plot(th, [bench["base"]["pck"][res][t] for t in th], ":", color=GREY, lw=1.3,
            label="starting checkpoint (ep. 345)")
    ax.plot(th, [bench["ref"]["pck"][res][t] for t in th], "-s", color=BLACK, lw=1.6, ms=4,
            label=r"$\lambda = 0$ control (ep. 369)")
    for i, k in enumerate(LAMK):
        ax.plot(th, [bench[k]["pck"][res][t] for t in th], "-o", color=SHADES[i], lw=1.7, ms=4,
                label=rf"$\lambda = 10^{{-{4-i}}}$")
    ax.set_xscale("log"); ax.set_xticks(th); ax.set_xticklabels([str(t) for t in th], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("threshold (px)"); ax.set_ylabel("PCK"); ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)
fig.tight_layout()
save(fig, "fig_mv_pck")

# fig_mv_mpjpe_quantiles
P = [50, 75, 90, 95, 99]
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))
r0 = np.array([bench["ref"]["pct"][p] for p in P])
axes[0].plot(P, r0, "-s", color=BLACK, lw=1.6, ms=5, label=r"$\lambda = 0$ control (ep. 369)")
for i, k in enumerate(LAMK):
    v = np.array([bench[k]["pct"][p] for p in P])
    axes[0].plot(P, v, "-o", color=SHADES[i], lw=1.7, ms=5, label=rf"$\lambda = 10^{{-{4-i}}}$")
    axes[1].plot(P, (v - r0) / r0 * 100, "-o", color=SHADES[i], lw=1.7, ms=5)
axes[1].axhline(0, color=BLACK, ls="--", lw=1.3)
axes[0].set_xlabel("percentile of the per-sample MPJPE"); axes[0].set_ylabel("MPJPE (mm)")
axes[1].set_xlabel("percentile of the per-sample MPJPE")
axes[1].set_ylabel("change vs. $\\lambda = 0$ control (%)")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
save(fig, "fig_mv_mpjpe_quantiles")

# fig_mv_sv_compare
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
xx = np.arange(5); K = ["ref"] + LAMK
for nm, S, col, mk in [("single-view", sv, ORANGE, "s"), ("multi-view", mv, BLUE, "o")]:
    m = "sv" if nm == "single-view" else "mv"
    axes[0].plot(xx, [S["rate"][k] for k in K], "-" + mk, color=col, lw=1.8, ms=5, label=nm)
    y = np.array([ACC[m][k] for k in K])
    axes[1].plot(xx, y, "-" + mk, color=col, lw=1.8, ms=5, label=nm)
    axes[2].plot(xx, (y - y[0]) / y[0] * 100, "-" + mk, color=col, lw=1.8, ms=5, label=nm)
axes[0].set_yscale("log"); axes[0].set_ylabel("Mean violation rate (%)")
axes[1].set_ylabel("MPJPE (mm)"); axes[1].margins(y=0.12)
axes[2].axhline(0, color=GREY, ls="--", lw=1.2)
axes[2].set_ylabel("MPJPE vs. own control (%)")
for ax in axes:
    ax.set_xticks(xx); ax.set_xticklabels(XTL); ax.set_xlabel(r"$\lambda$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.16))
fig.tight_layout()
save(fig, "fig_mv_sv_compare")
