"""Generate every LaTeX table in the paper directly from the experiment CSVs."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments import load
from synthgen import tail_report
from edge_profile import circ_sums, membership

OUT = "paper/tables"
GEN = ["block_bootstrap", "var1_bootstrap", "garch_fhs", "phase_randomization", "iaaft",
       "rung_full", "rung_symmetric", "rung_factor", "rung_diagonal", "rung_pooled"]
GLAB = {"block_bootstrap": "Stationary block bootstrap", "var1_bootstrap": "VAR(1), bootstrapped residuals",
        "garch_fhs": "GARCH(1,1)-FHS", "phase_randomization": "Phase randomization",
        "iaaft": "Multivariate IAAFT", "rung_full": "Rung: full", "rung_symmetric": "Rung: symmetric",
        "rung_factor": "Rung: factor", "rung_diagonal": "Rung: diagonal", "rung_pooled": "Rung: pooled"}
UNI = {"etf": "9 sector ETFs", "all": "97 instruments"}
CLS = ["full", "symmetric", "factor", "diagonal", "pooled"]


def w(name, s):
    open(f"{OUT}/{name}.tex", "w").write(s)


# ---------------- data summary ----------------
px = pd.read_csv("data/prices.csv", parse_dates=["Date"]).set_index("Date")
rows = []
for u in ["etf", "all"]:
    R, names = load(u)
    tr = tail_report(R)
    C = np.corrcoef(R.T)
    rows.append(f"{UNI[u]} & {R.shape[1]} & {R.shape[0]:,} & {tr['excess_kurtosis']:.1f} & "
                f"{tr['acf_abs_lag1to5']:.2f} & {C[np.triu_indices_from(C, 1)].mean():.2f} \\\\")
d0, d1 = px.index[1].strftime("%Y-%m-%d"), px.index[-1].strftime("%Y-%m-%d")
w("tab_data", r"""\begin{tabular}{lrrrrr}
\toprule
Universe & $N$ & $T$ & Excess kurtosis & ACF of $|r|$ & Mean pairwise corr. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}""")
open(f"{OUT}/dates.tex", "w").write(f"\\newcommand{{\\datestart}}{{{d0}}}\\newcommand{{\\dateend}}{{{d1}}}")

# ---------------- realism + own-class tau ----------------
def realism(u):
    d = pd.read_csv(f"data/exp_{u}.csv").groupby("generator").mean(numeric_only=True)
    lines = []
    for g in GEN:
        r = d.loc[g]
        lines.append(f"{GLAB[g]} & {(r['kurt'] + 0.0) if abs(r['kurt']) >= 0.05 else 0.0:.1f} & {r['acf']:.2f} & {r.corr_err:.3f} & {r.path_corr:+.3f} & "
                     f"{r.vol_ratio:.2f} & {r.tau_static_sharpe:.3f} \\\\")
        if g == "iaaft":
            lines.append(r"\midrule")
    return "\n".join(lines)


for u in ["etf", "all"]:
    R, _ = load(u)
    tr = tail_report(R)
    w(f"tab_realism_{u}", r"""\begin{tabular}{lrrrrrr}
\toprule
Generator & Kurtosis & ACF $|r|$ & Max corr.\ error & Path corr. & Vol.\ ratio & $\tau$ static Sharpe \\
\midrule
Original data & """ + f"{tr['excess_kurtosis']:.1f} & {tr['acf_abs_lag1to5']:.2f} & -- & -- & 1.00 & 1 \\\\\n\\midrule\n"
      + realism(u) + r"""
\bottomrule
\end{tabular}""")

# ---------------- tau matrices (circular and ordinary) ----------------
for u in ["etf", "all"]:
    d = pd.read_csv(f"data/exp_{u}.csv").groupby("generator").mean(numeric_only=True)
    lines = []
    for g in GEN:
        cells = []
        for c in CLS:
            a, b = d.loc[g, f"tauc_{c}"], d.loc[g, f"tau_{c}"]
            fa = r"\textbf{1}" if a > 0.99995 else f"{a:.3f}"
            cells.append(f"{fa} / {b:.3f}")
        lines.append(GLAB[g] + " & " + " & ".join(cells) + r" \\")
        if g == "iaaft":
            lines.append(r"\midrule")
    w(f"tab_tau_{u}", r"""\begin{tabular}{l""" + "c" * 5 + r"""}
\toprule
Generator & \multicolumn{5}{c}{Strategy class scored (circular / ordinary backtest)} \\
\cmidrule(lr){2-6}
 & full & symmetric & factor & diagonal & pooled \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---------------- edge profile ----------------
EG = ["rung_factor", "rung_diagonal", "rung_pooled", "block_bootstrap", "garch_fhs"]
EGL = ["Factor", "Diagonal", "Pooled", "Block bootstr.", "GARCH-FHS"]
for u in ["etf", "all"]:
    e = pd.read_csv(f"data/edge_{u}.csv")
    p = pd.read_csv(f"data/edge_profile_{u}.csv").set_index(["strategy", "generator"])
    o = e[e.generator == "original"].set_index("strategy")
    lines = []
    for s in o.index:
        cells = [f"{100*o.loc[s,'circ_ret']:.1f} ({o.loc[s,'circ_t']:.1f})"]
        for g in EG:
            r = p.loc[(s, g)]
            if g.startswith("rung_") and g[5:] in membership(s):
                cells.append(r"\emph{pinned}")
            else:
                cells.append(f"{100*r.circ_ret_null_mean:.1f}$\\pm${100*r.circ_ret_null_sd:.1f} [{r.p_circ_ret:.2f}]")
        lines.append(s + " & " + " & ".join(cells) + r" \\")
    w(f"tab_edge_{u}", r"""\begin{tabular}{l""" + "r" * 6 + r"""}
\toprule
Strategy & Original & """ + " & ".join(EGL) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")

# ---------------- reversal / volatility interaction ----------------
lines = []
conc = {}
for u in ["etf", "all"]:
    R, _ = load(u)
    N = R.shape[1]
    B = -(np.eye(N) - np.ones((N, N)) / N)
    def share(X):
        W_ = circ_sums(X, 1, 5) @ B.T
        pnl, g = np.sum(W_ * X, 1), np.abs(W_).sum(1)
        top = g >= np.quantile(g, 0.9)
        return pnl[top].sum() / pnl.sum()
    P = np.load(f"data/paths_{u}_rung_full.npy")
    sh = np.array([share(np.asarray(P[k])) for k in range(P.shape[0])])
    e = pd.read_csv(f"data/edge_{u}.csv")
    s = "XS reversal 1w"
    og = e[(e.generator == "original") & (e.strategy == s)].g_sharpe.iloc[0]
    cells = [f"{og:.2f}"]
    for g in ["rung_full", "rung_factor", "rung_diagonal", "rung_pooled", "block_bootstrap"]:
        x = e[(e.generator == g) & (e.strategy == s)].g_sharpe
        cells.append(f"{x.mean():.2f}$\\pm${x.std(ddof=0):.2f}")
    lines.append(f"{UNI[u]} & " + " & ".join(cells) + f" & {share(R):.2f} & {sh.mean():.2f}$\\pm${sh.std():.2f} \\\\")
    conc[u] = (share(R), sh.mean(), sh.std(), P.shape[0])
w("tab_reversal", r"""\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{6}{c}{Sharpe ratio, daily gross-normalized weekly reversal} & \multicolumn{2}{c}{Top-decile P\&L share} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-9}
Universe & Original & Full & Factor & Diagonal & Pooled & Block b. & Original & Full rung \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")
with open(f"{OUT}/numbers.tex", "w") as f:
    for u, (a, b, c, n) in conc.items():
        k = "Etf" if u == "etf" else "All"
        f.write(f"\\newcommand{{\\concOrig{k}}}{{{a:.2f}}}\\newcommand{{\\concSyn{k}}}{{{b:.2f}}}"
                f"\\newcommand{{\\concSynSd{k}}}{{{c:.2f}}}\\newcommand{{\\nPaths{k}}}{{{n}}}\n")
print("tables written:", sorted(os.listdir(OUT)))
