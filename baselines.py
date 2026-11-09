"""
baselines.py - reference generators that the rung surrogates are compared against.

All functions take R (T, N) returns and a numpy Generator, and return one (T, N) path.

  block_bootstrap        stationary bootstrap of whole dates (Politis & Romano 1994)
  var1_bootstrap         VAR(1) fitted by OLS, simulated with resampled residual vectors
  garch_fhs              univariate GARCH(1,1)-t per asset + filtered historical simulation
                         (standardized residual vectors resampled by date -> constant
                         conditional correlation, fat tails, vol clustering)
  phase_randomization    multivariate Fourier surrogate with a common random phase per
                         frequency (Prichard & Theiler 1994); Gaussianizes the marginals
  iaaft_multivariate     multivariate IAAFT (Schreiber & Schmitz 2000): alternate
                         cross-spectrum and per-asset rank constraints, ending on the rank
                         step (exact marginals, approximate spectrum)
"""

import numpy as np


def block_bootstrap(R, rng, block_len=20):
    T = R.shape[0]
    idx = np.empty(T, int)
    idx[0] = rng.integers(T)
    p = 1.0 / block_len
    for t in range(1, T):
        idx[t] = rng.integers(T) if rng.random() < p else (idx[t - 1] + 1) % T
    return R[idx]


class VAR1:
    def __init__(self, R):
        X = np.hstack([np.ones((R.shape[0] - 1, 1)), R[:-1]])
        B, *_ = np.linalg.lstsq(X, R[1:], rcond=None)
        self.c, self.A = B[0], B[1:].T
        self.resid = R[1:] - X @ B
        self.R = R

    def sample(self, rng, burn=500):
        T, N = self.R.shape
        e = self.resid[rng.integers(len(self.resid), size=T + burn)]
        y = np.empty((T + burn, N))
        y[0] = self.R.mean(0)
        for t in range(1, T + burn):
            y[t] = self.c + self.A @ y[t - 1] + e[t]
        return y[burn:]


class GarchFHS:
    def __init__(self, R):
        from arch import arch_model
        self.R = R
        T, N = R.shape
        self.params, Z = [], np.empty_like(R)
        for i in range(N):
            m = arch_model(100 * R[:, i], mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            f = m.fit(disp="off")
            mu, om, a, b = (f.params[k] for k in ("mu", "omega", "alpha[1]", "beta[1]"))
            self.params.append((mu, om, a, b))
            Z[:, i] = f.std_resid
        Z = np.nan_to_num(Z)
        self.Z = (Z - Z.mean(0)) / Z.std(0)      # FHS: unit-variance standardized residuals

    def sample(self, rng, burn=500):
        T, N = self.R.shape
        z = self.Z[rng.integers(T, size=T + burn)]
        mu, om, a, b = (np.array(v) for v in zip(*self.params))
        h = om / np.maximum(1 - a - b, 1e-3)
        out = np.empty((T + burn, N))
        for t in range(T + burn):
            e = np.sqrt(h) * z[t]
            out[t] = mu + e
            h = om + a * e ** 2 + b * h
        return out[burn:] / 100.0


def _common_phase(T, K, rng):
    ph = np.exp(2j * np.pi * rng.random(K))
    ph[0] = 1.0
    if T % 2 == 0:
        ph[-1] = rng.choice([-1.0, 1.0])
    return ph


def phase_randomization(R, rng):
    T = R.shape[0]
    X = np.fft.rfft(R, axis=0)
    return np.fft.irfft(X * _common_phase(T, X.shape[0], rng)[:, None], n=T, axis=0)


def iaaft_multivariate(R, rng, n_iter=100):
    T, N = R.shape
    Xo = np.fft.rfft(R, axis=0)
    srt = np.sort(R, axis=0)
    Y = np.column_stack([rng.permutation(R[:, i]) for i in range(N)])
    for _ in range(n_iter):
        Xy = np.fft.rfft(Y, axis=0)
        c = np.sum(np.conj(Xo) * Xy, axis=1)
        a = np.abs(c)
        ph = np.where(a > 1e-12, c / np.where(a > 1e-12, a, 1), 1)
        Xn = Xo * ph[:, None]
        Xn[0] = Xo[0]
        Z = np.fft.irfft(Xn, n=T, axis=0)
        Ynew = np.empty_like(Z)
        order = np.argsort(Z, axis=0)
        for i in range(N):
            Ynew[order[:, i], i] = srt[:, i]
        if np.array_equal(Ynew, Y):
            break
        Y = Ynew
    return Y                                    # ends on the rank step
