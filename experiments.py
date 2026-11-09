"""
experiments.py - score rung surrogates and baselines on one universe.

usage:  python experiments.py <universe> <n_paths> [generator,generator,...]
  universe : etf   (the 9 Select Sector SPDR ETFs)
             all   (88 stocks + 9 ETFs; indices, ^VIX and GOOG removed)

Every generator is scored on the same strategy battery:
  * 400 random linear-signal strategies (lags 1..5, ||A||_F = 1) in each subspace
    full / symmetric / factor / diagonal / pooled  -> Kendall tau of mean-P&L rankings,
    original vs synthetic, with ordinary (linear) and circular P&L
  * 400 random static portfolios -> tau of Sharpe-ratio rankings
  * realism: excess kurtosis, |r| autocorrelation, max correlation-matrix error,
    correlation of the synthetic path with the original, volatility ratio
"""
import sys, time, os, zlib
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthgen import RungGenerator, RUNGS, sample_strategies, mean_pnl, tail_report
import baselines as bl

L, NS = 5, 400
BETAS_ALL = {"full": 0.328, "symmetric": 0.352, "factor": 0.028,
             "diagonal": 0.046, "pooled": 0.014}        # calibrated earlier (all universe)


def load(universe):
    px = pd.read_csv("data/prices.csv", parse_dates=["Date"]).set_index("Date")
    if universe == "etf":
        px = px[[c for c in px.columns if c.startswith("XL")]]
    else:
        px = px.drop(columns=["^GSPC", "^RUT", "^NDX", "^VIX", "GOOG"])
    return px.pct_change().iloc[1:].to_numpy(), list(px.columns)


def main(universe, n_paths, which=None):
    R, names = load(universe)
    T, N = R.shape
    K = 3 if universe == "etf" else 5
    F = np.linalg.eigh(np.cov(R, rowvar=False))[1][:, ::-1][:, :K]
    classes = {c: sample_strategies(c, N, L, NS, F=F, seed=10 + i) for i, c in enumerate(RUNGS)}
    W = np.random.default_rng(99).normal(size=(NS, N))
    p0 = {c: (mean_pnl(A, R, False), mean_pnl(A, R, True)) for c, A in classes.items()}
    C0 = np.corrcoef(R.T)

    def sharpe(X):
        m, C = X.mean(0), np.cov(X, rowvar=False)
        return (W @ m) / np.sqrt(np.einsum("ni,ij,nj->n", W, C, W))
    s0 = sharpe(R)

    def score(Y):
        row = {}
        for c, A in classes.items():
            row[f"tau_{c}"] = stats.kendalltau(p0[c][0], mean_pnl(A, Y, False))[0]
            row[f"tauc_{c}"] = stats.kendalltau(p0[c][1], mean_pnl(A, Y, True))[0]
        row["tau_static_sharpe"] = stats.kendalltau(s0, sharpe(Y))[0]
        tr = tail_report(Y)
        row.update(kurt=tr["excess_kurtosis"], acf=tr["acf_abs_lag1to5"],
                   corr_err=float(np.abs(np.corrcoef(Y.T) - C0).max()),
                   path_corr=float(np.mean([np.corrcoef(R[:, i], Y[:, i])[0, 1] for i in range(N)])),
                   vol_ratio=float(np.mean(Y.std(0) / R.std(0))))
        return row

    gens = {
        "block_bootstrap": lambda: (lambda rng: bl.block_bootstrap(R, rng)),
        "var1_bootstrap": lambda: (lambda m: (lambda rng: m.sample(rng)))(bl.VAR1(R)),
        "garch_fhs": lambda: (lambda m: (lambda rng: m.sample(rng)))(bl.GarchFHS(R)),
        "phase_randomization": lambda: (lambda rng: bl.phase_randomization(R, rng)),
        "iaaft": lambda: (lambda rng: bl.iaaft_multivariate(R, rng)),
    }
    for j, r in enumerate(RUNGS):
        def mk(r=r, j=j):
            seg = {"rung": r, "F": F} if r == "factor" else {"rung": r}
            beta = "auto" if universe == "etf" else BETAS_ALL[r]
            g = RungGenerator(R, [seg], tail="rank", vol_beta=beta, seed=1000 + 17 * j)
            return lambda rng: g.generate(1)[0][0]
        gens[f"rung_{r}"] = mk

    out = f"data/exp_{universe}.csv"
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    for name, make in gens.items():
        if which and name not in which:
            continue
        if len(done) and name in set(done.generator):
            print(f"skip {name} (already in {out})", flush=True)
            continue
        t0 = time.time()
        sampler = make()
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        rows = []
        for p in range(n_paths):
            Y = sampler(rng)
            rows.append(dict(generator=name, path=p, **score(Y)))
        df = pd.DataFrame(rows)
        done = pd.concat([done, df], ignore_index=True)
        done.to_csv(out, index=False)
        m = df.mean(numeric_only=True)
        print(f"{name:20s} {time.time()-t0:6.0f}s  " +
              "  ".join(f"{c[:4]}={m['tau_'+c]:.3f}" for c in RUNGS) +
              f"  sharpe={m.tau_static_sharpe:.3f} kurt={m['kurt']:.1f} acf={m['acf']:.2f} "
              f"corr_err={m.corr_err:.3f} pcorr={m.path_corr:.3f}", flush=True)


if __name__ == "__main__":
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    main(sys.argv[1], int(sys.argv[2]), which)
