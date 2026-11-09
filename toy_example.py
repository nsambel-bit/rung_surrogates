"""Toy two-asset market used in the paper's explanation of the rungs."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthgen import RungGenerator, RUNGS, lag_moments

rng = np.random.default_rng(42)
T = 4000
x = np.zeros(T); y = np.zeros(T)
for t in range(1, T):
    x[t] = 0.10 * x[t - 1] + np.sqrt(1 - 0.10**2) * rng.normal()          # X: mild own momentum
    y[t] = 0.30 * x[t - 1] - 0.05 * y[t - 1] + 0.95 * rng.normal()        # Y follows X, mild reversal
R = 0.01 * np.column_stack([x, y])

def m1(Z):
    return 1e4 * lag_moments(Z - Z.mean(0), 1)[0]           # circular M_1, units of 1e-4

def row(lbl, M, f):
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    print(f"{lbl:10s} a={a:+.3f} b={b:+.3f} c={c:+.3f} d={d:+.3f}  a+d={a+d:+.3f} (b+c)/2={(b+c)/2:+.3f} market={f@M@f:+.3f}")
    return a, b, c, d

f = np.array([1, 1]) / np.sqrt(2)
res = {"original": m1(R)}
row("original", res["original"], f)
for r in RUNGS:
    seg = {"rung": r, "F": f[:, None]} if r == "factor" else {"rung": r}
    g = RungGenerator(R, [seg], tail="rank", vol_beta=0.0, seed=3)
    (Y,), _ = g.generate(1)
    res[r] = m1(Y)
    row(r, res[r], f)

lines = []
names = {"original": "Original", "full": "Full", "symmetric": "Symmetric", "factor": "Factor (market)",
         "diagonal": "Diagonal", "pooled": "Pooled"}
for k in ["original", "full", "symmetric", "factor", "diagonal", "pooled"]:
    M = res[k]; a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    lines.append(f"{names[k]} & {a:+.3f} & {d:+.3f} & {c:+.3f} & {b:+.3f} & {a+d:+.3f} & {(b+c)/2:+.3f} & {f@M@f:+.3f} \\\\")
    if k == "original":
        lines.append(r"\midrule")
open("paper/tables/tab_toy.tex", "w").write(r"""\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{4}{c}{Entries of $M_1$ ($\times 10^{-4}$)} & \multicolumn{3}{c}{Combinations} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
Data & $a$: X$\to$X & $d$: Y$\to$Y & $c$: X$\to$Y & $b$: Y$\to$X & $a+d$ & $(b+c)/2$ & market \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")
