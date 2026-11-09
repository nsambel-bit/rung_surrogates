import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os
def _save(fig, name, dpi=150):
    if _os.environ.get("PAPER"):
        fig._suptitle = None
        for t in list(fig.texts):
            t.remove()
        fig.tight_layout()
        fig.savefig(f"paper/figs/{name}.pdf")
    else:
        fig.savefig(f"figs/{name}.png", dpi=dpi)

from edge_profile import membership
INK, MUTED, GRID, SURF, SYN, ORIG, PIN = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
u = sys.argv[1]
title = {"etf": "9 sector ETFs", "all": "97 instruments"}[u]
df = pd.read_csv(f"data/edge_{u}.csv")
gens = ["block_bootstrap", "garch_fhs", "rung_full", "rung_symmetric", "rung_factor", "rung_diagonal", "rung_pooled"]
glab = ["Block bootstrap", "GARCH-FHS", "Rung: full", "Rung: symmetric", "Rung: factor", "Rung: diagonal", "Rung: pooled"]
strats = list(df[df.generator == "original"].strategy)
metrics = [("circ_ret", "Linear strategy, circular backtest\nannual return at unit avg gross (%)", 100),
           ("g_sharpe", "Daily gross-normalized strategy\nannualized Sharpe ratio", 1)]
fig, axes = plt.subplots(len(strats), 2, figsize=(9.5, 2.0 * len(strats) + 0.6) if _os.environ.get("PAPER") else (13, 2.3 * len(strats) + 1), sharey=True)
rng = np.random.default_rng(0)
for i, s in enumerate(strats):
    o = df[(df.generator == "original") & (df.strategy == s)].iloc[0]
    for j, (m, lab, sc) in enumerate(metrics):
        ax = axes[i, j]
        for g_i, g in enumerate(gens):
            x = df[(df.generator == g) & (df.strategy == s)][m].to_numpy() * sc
            pinned = g.startswith("rung_") and g[5:] in membership(s) and m == "circ_ret"
            yy = g_i + rng.uniform(-0.18, 0.18, len(x))
            ax.scatter(x, yy, s=10, color=PIN if pinned else SYN, alpha=0.55, lw=0)
            ax.plot([np.median(x)] * 2, [g_i - 0.3, g_i + 0.3], color=INK, lw=1.2)
        ax.axvline(o[m] * sc, color=ORIG, lw=2, zorder=5)
        allx = df[(df.strategy == s) & (df.generator != "original")][m].to_numpy() * sc
        lo, hi = np.percentile(allx, [1, 99])
        pad = 0.08 * (hi - lo + 1e-12)
        lo, hi = min(lo, o[m] * sc) - pad, max(hi, o[m] * sc) + pad
        ax.set_xlim(lo, hi)
        n_out = int(np.sum((allx < lo) | (allx > hi)))
        if n_out and not _os.environ.get("PAPER"):
            ax.text(1.0, 1.0, f"{n_out} outlying path(s) off-axis", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, color=MUTED)
        ax.set_yticks(range(len(gens)), glab)
        ax.invert_yaxis() if False else None
        ax.grid(axis="x", color=GRID, lw=0.6)
        if i == 0:
            ax.set_title(lab, loc="left", fontsize=10, color=INK)
        if j == 0:
            ax.text(-0.52 if _os.environ.get("PAPER") else -0.42, 0.5, s, transform=ax.transAxes, rotation=90, va="center", ha="center",
                    fontsize=10, fontweight="bold", color=INK)
    axes[i, 0].set_ylim(len(gens) - 0.5, -0.5)
fig.suptitle(f"Edge profile, {title}: original P&L (orange line) vs synthetic null distributions",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 1 - 0.55 / fig.get_figheight(), "Each dot is one synthetic path; black tick = median. Grey dots: the "
         "strategy lies inside the rung, so its linear P&L is pinned exactly.", ha="left", color=MUTED, fontsize=10)
fig.tight_layout(rect=(0.03, 0, 1, 1 - 0.75 / fig.get_figheight()))
_save(fig, f"edge_profile_{u}", 130)
