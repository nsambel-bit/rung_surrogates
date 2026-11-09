"""Run the rung generator on the 20y price dataset and score rank preservation."""
import sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude")
from synthgen import (RungGenerator, RUNGS, sample_strategies, mean_pnl, rank_agreement,
                      tail_report, abs_acf, lag_moments)
from scipy import stats

px = pd.read_csv("data/prices.csv", parse_dates=["Date"]).set_index("Date")
drop = ["^GSPC", "^RUT", "^NDX", "^VIX", "GOOG"]
px = px.drop(columns=drop)
R = px.pct_change().iloc[1:].to_numpy()
names = list(px.columns)
T, N = R.shape
L, NS, K = 5, 400, 5
vals, vecs = np.linalg.eigh(np.cov(R, rowvar=False))
F = vecs[:, ::-1][:, :K]
print(f"T={T} N={N}  orig {tail_report(R)}  top-{K} PCs explain "
      f"{vals[::-1][:K].sum()/vals.sum():.1%}", flush=True)

# ---- realistic named strategies (all ||A||=1), lags 1..L ----
etf = [names.index(c) for c in names if c.startswith("XL")]
I = np.eye(N); J = np.ones((N, N)) / N
def norm(A): return A / np.linalg.norm(A)
named = {}
named["TS momentum, lag 1 (diag)"] = norm(np.stack([I] + [0*I]*(L-1)))
named["TS momentum, lags 1-5 (diag)"] = norm(np.stack([I]*L))
named["XS reversal Lo-MacKinlay, lag 1"] = norm(np.stack([-(I-J)] + [0*I]*(L-1)))
named["XS reversal, lags 1-5"] = norm(np.stack([-(I-J)]*L))
named["market timing (1'1/N), lag 1"] = norm(np.stack([J] + [0*I]*(L-1)))
named["factor momentum top-5 PCs, lags 1-5"] = norm(np.stack([F @ F.T]*L))
named["PC1 lead-lag on stocks, lag 1"] = norm(np.stack([np.outer(F[:,1], F[:,0])] + [0*I]*(L-1)))
E = np.zeros((N, N)); E[np.ix_(etf, etf)] = 1.0; E[etf, etf] = 0.0
named["sector-ETF cross lead-lag, lag 1"] = norm(np.stack([E] + [0*I]*(L-1)))
names_s = list(named); A_named = np.stack([named[k] for k in names_s])

# which rung contains which named strategy (projection norm ≈ 1)
def in_rung(A, rung):
    if rung == "full": return True
    if rung == "symmetric": return np.allclose(A, np.swapaxes(A, -1, -2))
    if rung == "diagonal": return np.allclose(A, A * I)
    if rung == "pooled": return all(np.allclose(a, a[0,0]*I) for a in A)
    if rung == "factor": P = F @ F.T; return np.allclose(A, P @ A @ P)

# noise floor of the full class: best in-sample linear-signal "edge" vs noise
M = lag_moments(R - R.mean(0), L)
S = np.cov(R, rowvar=False); w, V = np.linalg.eigh(S); Sih = V @ np.diag(w**-0.5) @ V.T
port = T * sum(np.sum((Sih @ m @ Sih)**2) for m in M)
dof = N * N * L
print(f"portmanteau stat {port:.0f} vs chi2 dof {dof} (z={(port-dof)/np.sqrt(2*dof):.1f})", flush=True)

rows, named_rows = [], []
n_paths = 5
for rung in RUNGS:
    t0 = time.time()
    seg = {"rung": rung, "F": F} if rung == "factor" else {"rung": rung}
    gen = RungGenerator(R, [seg], tail="rank", seed=2026)
    Ys, infos = gen.generate(n_paths)
    A_in = sample_strategies(rung, N, L, NS, F=F, seed=1)
    A_all = sample_strategies("full", N, L, NS, seed=2)
    Wst = np.random.default_rng(3).normal(size=(NS, N))          # static portfolios
    def static_sharpe(X): m, C = X.mean(0), np.cov(X, rowvar=False); return (Wst @ m) / np.sqrt(np.einsum("ni,ij,nj->n", Wst, C, Wst))
    p0 = mean_pnl(A_named, R)
    for j, Y in enumerate(Ys):
        tr = tail_report(Y)
        rows.append(dict(rung=rung, path=j, vol_beta=infos[j]["vol_beta"],
                         pinned_err=max(gen.pinned_error(Y).values()),
                         tau_in_circ=rank_agreement(A_in, R, Y, True),
                         tau_in_std=rank_agreement(A_in, R, Y, False),
                         tau_full=rank_agreement(A_all, R, Y, True),
                         kurt=tr["excess_kurtosis"], acf=tr["acf_abs_lag1to5"],
                         corr=np.mean([np.corrcoef(R[:, i], Y[:, i])[0, 1] for i in range(N)]),
                         vol_ratio=float(np.mean(Y.std(0) / R.std(0))),
                         corr_err=float(np.abs(np.corrcoef(Y.T) - np.corrcoef(R.T)).max()),
                         avg_corr=float(np.corrcoef(Y.T)[np.triu_indices(N, 1)].mean()),
                         tau_static_sharpe=stats.kendalltau(static_sharpe(R), static_sharpe(Y))[0],
                         seconds=None))
        p1 = mean_pnl(A_named, Y)
        for k, nm in enumerate(names_s):
            named_rows.append(dict(rung=rung, path=j, strategy=nm, inside=in_rung(named[nm], rung),
                                   pnl_orig=p0[k], pnl_syn=p1[k]))
    print(f"{rung:9s} done in {time.time()-t0:.0f}s  beta={infos[0]['vol_beta']:.3f}", flush=True)
    np.save(f"data/syn_sigma_{rung}.npy", np.stack(Ys))

df = pd.DataFrame(rows); dn = pd.DataFrame(named_rows)
df.to_csv("data/rung_summary_sigma.csv", index=False); dn.to_csv("data/named_strategies_sigma.csv", index=False)
pd.set_option("display.width", 200)
print(df.groupby("rung", sort=False).agg(['mean','min','max']).round(4)[
      ["pinned_err","corr_err","tau_in_circ","tau_in_std","tau_full","tau_static_sharpe","kurt","acf","avg_corr","corr","vol_ratio","vol_beta"]].T.to_string())
