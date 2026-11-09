import numpy as np, pandas as pd, matplotlib
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

import matplotlib.dates as mdates
import matplotlib.ticker

px = pd.read_csv("data/prices.csv", parse_dates=["Date"]).set_index("Date")
px = px.drop(columns=["^GSPC", "^RUT", "^NDX", "^VIX", "GOOG"])
R = px.pct_change().iloc[1:]
dates, names = R.index, list(px.columns)
R = R.to_numpy()
rungs = ["full", "symmetric", "factor", "diagonal", "pooled"]
syn = {r: np.load(f"data/syn_sigma_{r}.npy") for r in rungs}
assets = ["AAPL", "JPM", "XLE"]

INK, MUTED, GRID, SURF, SYN = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False})

def wealth(x): return np.cumprod(1 + x)

# ---------- Figure 1: wealth paths ----------
fig, axes = plt.subplots(len(rungs), len(assets), figsize=(9, 10.5) if _os.environ.get("PAPER") else (13, 14), sharex=True)
for i, r in enumerate(rungs):
    for j, a in enumerate(assets):
        ax, k = axes[i, j], names.index(a)
        for p in range(syn[r].shape[0]):
            ax.plot(dates, wealth(syn[r][p, :, k]), color=SYN, lw=1.0, alpha=0.55,
                    label="Synthetic (5 paths)" if p == 0 else None)
        ax.plot(dates, wealth(R[:, k]), color=INK, lw=2.0, label="Original")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if i == 0: ax.set_title(a, fontsize=11, color=INK, loc="left", fontweight="bold")
        if j == 0: ax.set_ylabel(f"{r} rung\nwealth of $1 (log)", fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
h, l = axes[0, 0].get_legend_handles_labels()
if _os.environ.get("PAPER"):
    axes[0, 0].legend(h[::-1], l[::-1], loc="upper left", frameon=False, fontsize=8)
else:
    fig.legend(h[::-1], l[::-1], loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(0.98, 0.995))
fig.suptitle("Original vs synthetic wealth paths, by rung", x=0.02, ha="left", fontsize=14,
             color=INK, fontweight="bold", y=0.995)
fig.text(0.02, 0.972, "Mean, covariance and the rung's lagged statistics are pinned; paths, timing of "
         "crashes and drawdowns are free.", ha="left", color=MUTED, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.965))
_save(fig, f"wealth_paths_by_rung", 130)

# ---------- Figure 2: daily returns, one asset ----------
a = "AAPL"; k = names.index(a)
rows = [("Original", R[:, k], INK)] + [(f"{r} rung (path 1)", syn[r][0, :, k], SYN) for r in rungs]
fig, axes = plt.subplots(len(rows), 1, figsize=(9, 8.5) if _os.environ.get("PAPER") else (13, 12), sharex=True, sharey=True)
for ax, (lab, x, c) in zip(axes, rows):
    ax.plot(dates, x * 100, color=c, lw=0.5)
    ax.set_ylabel("%", fontsize=9)
    ax.set_title(lab, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.suptitle(f"{a} daily returns: original vs one synthetic path per rung", x=0.02, ha="left",
             fontsize=14, color=INK, fontweight="bold", y=0.995)
fig.text(0.02, 0.962, "Same return distribution and volatility; high-volatility clusters land at new dates.",
         ha="left", color=MUTED, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
_save(fig, f"daily_returns_AAPL", 130)

# ---------- Figure 3: equal-weight portfolio, all rungs ----------
ew = R.mean(1)
fig, axes = plt.subplots(1, len(rungs), figsize=(16, 4), sharey=True)
for ax, r in zip(axes, rungs):
    for p in range(syn[r].shape[0]):
        ax.plot(dates, wealth(syn[r][p].mean(1)), color=SYN, lw=1.0, alpha=0.55)
    ax.plot(dates, wealth(ew), color=INK, lw=2.0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_title(f"{r} rung", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(5)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0].set_ylabel("wealth of $1 (log)")
fig.suptitle("Equal-weight portfolio of all 97 instruments: original (black) vs 5 synthetic paths (blue)",
             x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
fig.tight_layout()
_save(fig, f"equal_weight_by_rung", 130)

# drawdown summary for the caption
def mdd(x): w = wealth(x); return (w / np.maximum.accumulate(w) - 1).min()
print("EW max drawdown orig %.1f%%" % (100 * mdd(ew)))
for r in rungs:
    print(r, ["%.1f%%" % (100 * mdd(syn[r][p].mean(1))) for p in range(5)],
          "terminal x", ["%.1f" % wealth(syn[r][p].mean(1))[-1] for p in range(5)], "orig %.1f" % wealth(ew)[-1])
