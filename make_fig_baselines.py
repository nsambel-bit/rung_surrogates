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

from matplotlib.colors import LinearSegmentedColormap
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
cmap = LinearSegmentedColormap.from_list("b", ["#f2f6fc", "#9cc0ee", "#2a78d6", "#0d3b7a"])
order = ["block_bootstrap", "var1_bootstrap", "garch_fhs", "phase_randomization", "iaaft",
         "rung_full", "rung_symmetric", "rung_factor", "rung_diagonal", "rung_pooled"]
labels = ["Block bootstrap", "VAR(1) + bootstrap", "GARCH-FHS", "Phase randomization", "Multivariate IAAFT",
          "Rung: full", "Rung: symmetric", "Rung: factor", "Rung: diagonal", "Rung: pooled"]
cls = ["full", "symmetric", "factor", "diagonal", "pooled"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.facecolor": SURF})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6) if _os.environ.get("PAPER") else (14, 5.6))
summ = []
for ax, (u, title) in zip(axes, [("etf", "9 sector ETFs"), ("all", "97 instruments (88 stocks + 9 ETFs)")]):
    d = pd.read_csv(f"data/exp_{u}.csv").groupby("generator").mean(numeric_only=True).loc[order]
    M = d[[f"tauc_{c}" for c in cls] + ["tau_static_sharpe"]].to_numpy()
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}" if v < 0.9995 else "1", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else INK, fontweight="bold" if v > 0.9995 else None)
    ax.set_xticks(range(6), [c for c in cls] + ["static\nSharpe"])
    ax.set_yticks(range(len(order)), labels if ax is axes[0] else [])
    ax.axhline(4.5, color=INK, lw=1.2)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", fontsize=11)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xlabel("strategy class whose ranking is scored", color=MUTED)
    d["universe"] = u; summ.append(d)
fig.suptitle("Kendall τ between original and synthetic strategy rankings (mean of 20 paths)",
             x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
fig.text(0.01, 0.925, "Rows: generators. Each rung generator gives τ = 1 exactly on its own class and on the "
         "classes nested inside it; baselines give no such guarantee.", ha="left", color=MUTED, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.91))
_save(fig, f"baselines_tau_matrix", 140)
pd.concat(summ).round(4).to_csv("data/baselines_summary.csv")
