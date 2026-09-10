"""Regenerate every figure of mv_study.tex from prior_study_results/.

Run from anywhere:  python report_joint_limit_prior/make_mv_figures.py
Writes fig_mv_*.pdf and fig_mv_*.png into report_joint_limit_prior/figures/.
"""
import os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "prior_study_results")
OUT  = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})
BLUE = "#1f77b4"; GREY = "#7f7f7f"
SHADES = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
LAMK = ["1e-4", "1e-3", "1e-2", "1e-1"]
LAMV = [1e-4, 1e-3, 1e-2, 1e-1]
LAMLBL = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
XTL = ["0"] + LAMLBL

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)

# ----------------------------------------------------------------- load
DIRS = {"0": "mv_reference/analysis"}
DIRS.update({k: f"lam{k}/mv_constrained/analysis" for k in LAMK})
K = ["0"] + LAMK
pa = {k: pd.read_csv(f"{ROOT}/{v}/per_axis_stats.csv").set_index(["joint", "axis"]) for k, v in DIRS.items()}
lv = {k: pd.read_csv(f"{ROOT}/{v}/limit_violations.csv").set_index(["joint", "axis"]) for k, v in DIRS.items()}
ref = lv["0"].copy(); ref["span"] = ref.limit_hi_deg - ref.limit_lo_deg
B = ref[ref.span < 359].index                       # 129 bounded axes
span = ref.loc[B, "span"]; lo = ref.loc[B, "limit_lo_deg"]; hi = ref.loc[B, "limit_hi_deg"]
rom = {k: pa[k].loc[B, "rom_deg"] for k in K}
clip = np.maximum(np.minimum(pa["0"].loc[B, "max_deg"], hi) - np.maximum(pa["0"].loc[B, "min_deg"], lo), 0.0)

sweep = pd.read_csv(f"{ROOT}/sweep/sweep.csv")
mvs = sweep[sweep["mode"] == "multiview"].reset_index(drop=True)
svs = sweep[sweep["mode"] == "singleview"].reset_index(drop=True)

BENCH = {"0": f"{ROOT}/mv_reference/benchmark_multiview_mv_reference_on_SMILySTICKS_centred_reprojected_FIXED/benchmark_report.txt"}
BENCH.update({k: f"{ROOT}/lam{k}/mv_constrained/benchmark_multiview_mv_constrained_lam{k}_on_SMILySTICKS_centred_reprojected_FIXED/benchmark_report.txt" for k in LAMK})

def parse_bench(p):
    txt = open(p).read()
    body = txt.split("==== BENCHMARK RESULTS")[1]
    out = {"pck": {}, "pct": {}}
    chunks = body.split("-- PCK @ ")
    for ch in chunks[1:]:
        key = "native" if ch.startswith("native") else "input"
        blk = ch.split("MPJPE (mm)")[0]
        curve = blk.split("PCK curve:")[1]
        out["pck"][key] = {int(m.group(1)): float(m.group(2))
                           for m in re.finditer(r"PCK@(\d+)px: ([\d.]+)", curve)}
    for m in re.finditer(r"P(\d+): ([\d.]+)", body.split("MPJPE percentiles")[1]):
        out["pct"][int(m.group(1))] = float(m.group(2))
    return out
bench = {k: parse_bench(v) for k, v in BENCH.items()}

# =============================================== fig_mv_tradeoff
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
x = np.arange(5)
series = [(mvs.mean_viol_rate.values, "Mean violation rate (%)", True),
          (mvs.max_overshoot_deg.values, "Maximum overshoot (deg)", False),
          (mvs.mpjpe_mm.values, "MPJPE (mm)", False)]
for ax, (y, lab, logy) in zip(axes, series):
    ax.plot(x[:2], y[:2], "--", color=GREY, lw=1.4, zorder=1)
    ax.plot(x[1:], y[1:], "-o", color=BLUE, lw=2, ms=5, zorder=3,
            label="epoch-matched arms (epoch 369)")
    ax.plot(x[0], y[0], "o", mfc="white", mec=GREY, mew=1.6, ms=8, zorder=4,
            label=r"reference, $\lambda = 0$ (epoch 345)")
    if logy: ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(XTL); ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(lab)
axes[2].annotate("0.9639", (0, mvs.mpjpe_mm[0]), textcoords="offset points",
                 xytext=(11, -3), fontsize=9, color=GREY)
axes[2].annotate("1.2111", (4, mvs.mpjpe_mm[4]), textcoords="offset points",
                 xytext=(-36, -4), fontsize=9, color=GREY)
axes[2].set_ylim(0.94, 1.245)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.16))
fig.tight_layout()
save(fig, "fig_mv_tradeoff")

# =============================================== fig_mv_worst_axes
worst = ref.sort_values("pct_frames_violating", ascending=False).head(12).index
names = [f"{j}.{a}" for j, a in worst]
fig, ax = plt.subplots(figsize=(9.6, 4.3))
w = 0.19; xs = np.arange(len(worst)); FLOOR = 2e-3
ax.bar(xs - 2.0 * w, np.maximum(ref.loc[worst, "pct_frames_violating"], FLOOR), w,
       color="white", edgecolor=GREY, lw=1.1, label=r"$\lambda = 0$")
for i, k in enumerate(LAMK):
    ax.bar(xs + (i - 1.0) * w, np.maximum(lv[k].loc[worst, "pct_frames_violating"], FLOOR), w,
           color=SHADES[i], label=rf"$\lambda = 10^{{-{4-i}}}$")
ax.set_yscale("log"); ax.set_ylim(FLOOR, 300)
ax.set_ylabel("frames violating (%)")
ax.set_xticks(xs); ax.set_xticklabels([n for n in names], rotation=35, ha="right", fontsize=8.5)
ax.set_xlabel("the twelve axes most often violated by the reference arm")
ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout()
save(fig, "fig_mv_worst_axes")

# =============================================== fig_mv_pck
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))
for ax, res, title in zip(axes, ["native", "input"],
                          ["native override 1530\u00d71530", "network input 224\u2009px"]):
    th = sorted(bench["0"]["pck"][res])
    ax.plot(th, [bench["0"]["pck"][res][t] for t in th], "--o", color=GREY, lw=1.5, ms=4,
            mfc="white", label=r"$\lambda = 0$ (ep. 345)")
    for i, k in enumerate(LAMK):
        ax.plot(th, [bench[k]["pck"][res][t] for t in th], "-o", color=SHADES[i], lw=1.7, ms=4,
                label=rf"$\lambda = 10^{{-{4-i}}}$")
    ax.set_xscale("log"); ax.set_xticks(th); ax.set_xticklabels([str(t) for t in th], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("threshold (px)"); ax.set_ylabel("PCK"); ax.set_title(title)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[0].legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout()
save(fig, "fig_mv_pck")

# =============================================== fig_mv_mpjpe_quantiles
P = [50, 75, 90, 95, 99]
fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6))
r0 = np.array([bench["0"]["pct"][p] for p in P])
axes[0].plot(P, r0, "--o", color=GREY, lw=1.5, ms=5, mfc="white", label=r"$\lambda = 0$ (ep. 345)")
for i, k in enumerate(LAMK):
    v = np.array([bench[k]["pct"][p] for p in P])
    axes[0].plot(P, v, "-o", color=SHADES[i], lw=1.7, ms=5, label=rf"$\lambda = 10^{{-{4-i}}}$")
    axes[1].plot(P, (v - r0) / r0 * 100, "-o", color=SHADES[i], lw=1.7, ms=5)
axes[1].axhline(0, color=GREY, ls="--", lw=1.3)
axes[0].set_xlabel("percentile of the per-sample MPJPE"); axes[0].set_ylabel("MPJPE (mm)")
axes[1].set_xlabel("percentile of the per-sample MPJPE")
axes[1].set_ylabel("change vs. reference (%)")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
save(fig, "fig_mv_mpjpe_quantiles")

# =============================================== fig_mv_rom_retention
pv = ref.loc[B, "pct_frames_violating"]
edges = [-1e-9, 1e-9, 1.0, 10.0, 100.1]
labels = ["never\n0%", "rare\n0\u20131%", "some\n1\u201310%", "chronic\n>10%"]
bucket = pd.cut(pv, edges, labels=labels)
cnt = bucket.value_counts().reindex(labels)
fig, ax = plt.subplots(figsize=(9.4, 4.4))
w = 0.2; xs = np.arange(4)
for i, k in enumerate(LAMK):
    ret = (rom[k] / rom["0"]).groupby(bucket, observed=False).mean().reindex(labels) * 100
    ax.bar(xs + (i - 1.5) * w, ret.values, w, color=SHADES[i], label=rf"$\lambda = 10^{{-{4-i}}}$")
cb = (clip / rom["0"]).groupby(bucket, observed=False).mean().reindex(labels) * 100
for j, v in enumerate(cb.values):
    ax.plot([xs[j] - 2 * w, xs[j] + 2 * w], [v, v], "--", color="0.15", lw=2.2,
            label="clipping bound" if j == 0 else None)
ax.set_xticks(xs)
ax.set_xticklabels([f"{l}\n({c} axes)" for l, c in zip(labels, cnt.values)])
ax.set_xlabel("violation rate of the axis in the reference arm")
ax.set_ylabel("ROM retained (% of reference)")
h, l = ax.get_legend_handles_labels()
ax.legend(h[1:] + h[:1], l[1:] + l[:1], ncol=5, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout()
save(fig, "fig_mv_rom_retention")

# =============================================== fig_mv_rom_scatter
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
for ax, k in zip(axes, ["1e-4", "1e-1"]):
    sc = ax.scatter(rom["0"], rom[k], c=pv, cmap="YlOrRd", s=26, ec="0.3", lw=0.4,
                    vmin=0, vmax=100, zorder=3)
    for x0, c0 in zip(rom["0"], clip):
        ax.plot([x0 * 0.94, x0 * 1.06], [c0, c0], "-", color="0.55", lw=0.8, zorder=1)
    lim = [3, 200]
    ax.plot(lim, lim, "--", color=GREY, lw=1.2, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"reference ROM (deg), $\lambda = 0$")
    ax.set_ylabel(rf"ROM (deg) at $\lambda = 10^{{{k[0]}0^{{{k[-2:]}}}}}$".replace("10^{10^", "10^{"))
    ax.set_title(rf"$\lambda = {k[0]}0^{{{-int(k[-1])}}}$".replace("10^", "10^"))
axes[0].set_ylabel(r"ROM (deg) at $\lambda = 10^{-4}$"); axes[0].set_title(r"$\lambda = 10^{-4}$")
axes[1].set_ylabel(r"ROM (deg) at $\lambda = 10^{-1}$"); axes[1].set_title(r"$\lambda = 10^{-1}$")
cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
cb.set_label("reference violation rate (% of frames)")
save(fig, "fig_mv_rom_scatter")

# =============================================== fig_mv_rom_ratio
fig, ax = plt.subplots(figsize=(9.2, 4.0))
bins = np.linspace(0, 4, 41)
ax.hist(rom["0"] / span, bins=bins, color="white", ec=GREY, lw=1.2, label=r"$\lambda = 0$")
for i, k in enumerate(["1e-4", "1e-1"]):
    ax.hist(rom[k] / span, bins=bins, color=SHADES[i * 3], alpha=0.75,
            label=rf"$\lambda = 10^{{-{4 if i == 0 else 1}}}$")
ax.axvline(1.0, color="0.15", ls="--", lw=1.8)
ax.text(1.03, ax.get_ylim()[1] * 0.93, "authored range", fontsize=9, color="0.15")
ax.set_xlabel("observed ROM / authored range")
ax.set_ylabel("number of axes (of 129)")
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "fig_mv_rom_ratio")

# =============================================== fig_mv_sv_compare
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3))
xx = np.arange(5)
pairs = [("mean_viol_rate", "Mean violation rate (%)", True),
         ("mpjpe_mm", "MPJPE (mm)", False)]
for ax, (col, lab, logy) in zip(axes[:2], pairs):
    ax.plot(xx, svs[col].values, "-s", color="#d95f02", lw=1.8, ms=5, label="single-view")
    ax.plot(xx, mvs[col].values, "-o", color=BLUE, lw=1.8, ms=5, label="multi-view")
    if logy: ax.set_yscale("log")
    else: ax.margins(y=0.12)
    ax.set_xticks(xx); ax.set_xticklabels(XTL); ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(lab)
ax = axes[2]
for nm, s, col, mk in [("single-view", svs, "#d95f02", "s"), ("multi-view", mvs, BLUE, "o")]:
    rel = (s.mpjpe_mm.values - s.mpjpe_mm.values[0]) / s.mpjpe_mm.values[0] * 100
    ax.plot(xx, rel, "-" + mk, color=col, lw=1.8, ms=5, label=nm)
ax.axhline(0, color=GREY, ls="--", lw=1.2)
ax.set_xticks(xx); ax.set_xticklabels(XTL); ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("MPJPE vs. own reference (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.16))
fig.tight_layout()
save(fig, "fig_mv_sv_compare")
