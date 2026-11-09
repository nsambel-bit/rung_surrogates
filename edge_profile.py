"""
edge_profile.py - where does a strategy's edge live?

For every strategy and every generator, the synthetic paths form a null distribution:
"the strategy's P&L is explained by the statistics this generator pins".
  * rung generators pin mu, Sigma and the rung's spectral statistics exactly
  * block bootstrap / GARCH-FHS are reference nulls (no lagged structure guaranteed)

The p-value of the original P&L against each null, across the rung lattice, is the
strategy's edge profile.

Strategies (linear signal strategies, lags in trading days):
  XS momentum 12-1    A_k = I - 11'/N      k = 22..252   (cross-sectional)
  TS momentum 12-1    A_k = I              k = 22..252   (time-series, per asset)
  Market trend 12-1   A_k = 11'/N          k = 22..252   (all assets follow the market)
  Factor momentum 12-1 A_k = F F'          k = 22..252   (top-K principal components)
  XS reversal 1w      A_k = -(I - 11'/N)   k = 1..5

Three P&L versions. Linear weights are scaled by ONE constant per strategy, chosen so
the original has unit average gross exposure; the same constant is used on every
synthetic path, so the strategy stays linear and "% per year" is readable.
  circ_ret : linear strategy, circular backtest (sample treated as a loop), annualized.
             This is exactly the statistic the rungs pin.
  lin_ret  : linear strategy, ordinary backtest after a 252-day warm-up, annualized.
             Differs from circ_ret by end effects of order lookback / T.
  g_sharpe : weights re-normalized to unit gross EVERY DAY (what practitioners trade),
             ordinary backtest. Nonlinear in returns, so outside the pinned class.

usage: python edge_profile.py <etf|all> <n_paths>
"""
import sys, os, time, zlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthgen import RungGenerator, RUNGS
import baselines as bl
from experiments import load, BETAS_ALL

LAG0, LAG1, WARM = 22, 252, 252


def signal_sums(R, lo, hi):
    """S_t = sum_{k=lo}^{hi} r_{t-k}  (NaN where not available)."""
    C = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R, axis=0)])
    T = R.shape[0]
    S = np.full_like(R, np.nan)
    t = np.arange(hi, T)
    S[t] = C[t - lo + 1] - C[t - hi]
    return S


def strategies(N, F):
    I, J = np.eye(N), np.ones((N, N)) / N
    return {
        "XS momentum 12-1": (I - J, LAG0, LAG1),
        "TS momentum 12-1": (I, LAG0, LAG1),
        "Market trend 12-1": (J, LAG0, LAG1),
        "Factor momentum 12-1": (F @ F.T, LAG0, LAG1),
        "XS reversal 1w": (-(I - J), 1, 5),
    }


def membership(name):
    """Rungs whose strategy subspace contains this strategy (exactly pinned there)."""
    return {"XS momentum 12-1": {"full", "symmetric"},
            "TS momentum 12-1": {"full", "symmetric", "diagonal", "pooled"},
            "Market trend 12-1": {"full", "symmetric"},
            "Factor momentum 12-1": {"full", "symmetric", "factor"},
            "XS reversal 1w": {"full", "symmetric"}}[name]


def circ_sums(R, lo, hi):
    T = R.shape[0]
    Z = np.vstack([R[-hi:], R])
    C = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(Z, axis=0)])
    t = np.arange(T) + hi
    return C[t - lo + 1] - C[t - hi]


def scales(R, strats):
    return {k: 1.0 / np.mean(np.abs(circ_sums(R, lo, hi) @ B.T).sum(1))
            for k, (B, lo, hi) in strats.items()}


def evaluate(R, strats, scale):
    out = {}
    for name, (B, lo, hi) in strats.items():
        Wc = circ_sums(R, lo, hi) @ B.T * scale[name]
        circ = np.sum(Wc * R, axis=1)
        W = (signal_sums(R, lo, hi) @ B.T * scale[name])[WARM:]
        r = R[WARM:]
        lin = np.sum(W * r, axis=1)
        G = W / np.maximum(np.abs(W).sum(1, keepdims=True), 1e-300)
        g = np.sum(G * r, axis=1)
        out[name] = dict(circ_ret=252 * circ.mean(),
                         circ_t=np.sqrt(len(circ)) * circ.mean() / circ.std(),
                         lin_ret=252 * lin.mean(),
                         g_sharpe=np.sqrt(252) * g.mean() / g.std())
    return out


def paths_for(gen, R, F, universe, n):
    f = f"data/paths_{universe}_{gen}.npy"
    if os.path.exists(f):
        P = np.load(f, mmap_mode="r")
        if P.shape[0] >= n:
            return P[:n]
    rng = np.random.default_rng(zlib.crc32(("edge" + gen).encode()))
    if gen == "block_bootstrap":
        P = [bl.block_bootstrap(R, rng) for _ in range(n)]
    elif gen == "garch_fhs":
        m = bl.GarchFHS(R); P = [m.sample(rng) for _ in range(n)]
    else:
        r = gen.replace("rung_", "")
        seg = {"rung": r, "F": F} if r == "factor" else {"rung": r}
        beta = "auto" if universe == "etf" else BETAS_ALL[r]
        g = RungGenerator(R, [seg], tail="rank", vol_beta=beta,
                          seed=zlib.crc32(("edge" + gen).encode()))
        P, _ = g.generate(n)
    P = np.stack(P)
    np.save(f, P)
    return P


def main(universe, n):
    R, names = load(universe)
    T, N = R.shape
    K = 3 if universe == "etf" else 5
    F = np.linalg.eigh(np.cov(R, rowvar=False))[1][:, ::-1][:, :K]
    S = strategies(N, F)
    sc = scales(R, S)
    orig = evaluate(R, S, sc)
    rows = [dict(generator="original", path=-1, strategy=k, **v) for k, v in orig.items()]
    for gen in ["block_bootstrap", "garch_fhs"] + [f"rung_{r}" for r in RUNGS]:
        t0 = time.time()
        P = paths_for(gen, R, F, universe, n)
        for p in range(P.shape[0]):
            for k, v in evaluate(np.asarray(P[p]), S, sc).items():
                rows.append(dict(generator=gen, path=p, strategy=k, **v))
        print(f"{gen:16s} {time.time()-t0:6.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"data/edge_{universe}.csv", index=False)

    # profile table: p-values (two-sided, rank-based) and null means
    summ = []
    for k in S:
        o = df[(df.generator == "original") & (df.strategy == k)].iloc[0]
        for gen in df.generator.unique()[1:]:
            s = df[(df.generator == gen) & (df.strategy == k)]
            inside = gen.startswith("rung_") and gen[5:] in membership(k)
            row = dict(strategy=k, generator=gen, inside=inside)
            for m in ["circ_ret", "lin_ret", "g_sharpe"]:
                x = s[m].to_numpy()
                row[f"{m}_orig"] = o[m]
                row[f"{m}_null_mean"] = x.mean()
                row[f"{m}_null_sd"] = x.std()
                if x.std() <= 1e-9 * max(abs(o[m]), 1e-12):
                    row[f"p_{m}"] = np.nan                       # pinned exactly: no null spread
                else:
                    hi_ = (np.sum(x >= o[m]) + 1) / (len(x) + 1)
                    lo_ = (np.sum(x <= o[m]) + 1) / (len(x) + 1)
                    row[f"p_{m}"] = min(1.0, 2 * min(hi_, lo_))
            summ.append(row)
    sm = pd.DataFrame(summ)
    sm.to_csv(f"data/edge_profile_{universe}.csv", index=False)
    pd.set_option("display.width", 220)
    o = df[df.generator == "original"].set_index("strategy")
    print(o[["circ_ret", "circ_t", "lin_ret", "g_sharpe"]].round(4).to_string())
    print(sm[["strategy", "generator", "inside", "circ_ret_null_mean", "circ_ret_null_sd", "p_circ_ret",
              "lin_ret_null_mean", "p_lin_ret", "g_sharpe_orig", "g_sharpe_null_mean", "p_g_sharpe"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
