"""
synthgen.py - rung-preserving synthetic return generator with fat tails.

Strategy class: linear signal portfolios  w_t = sum_{k>=1} A_k r_{t-k},  ||A|| = 1,
ranked by mean P&L  <A, M>,  M_k = E[r_t r_{t-k}'].

A "rung" is a subspace V = W (x) Phi:
  W   : cross-sectional structure of each A_k  (full / symmetric / factor / diagonal / pooled)
  Phi : frequency band of the lag profile      (cycles per observation, 0 .. 0.5)

Rankings inside V are preserved iff the projection of the cross-spectrum S(w) = X X^H
onto W is preserved at every in-band frequency. The generator alternates:

  R-step : move each in-band Fourier vector to the nearest point of the rung's orbit
           {U X_orig(w) : U in G_W}
  T-step : impose tails (per-asset rank remap / Student-t target / radial Mahalanobis remap)

and always FINISHES on the R-step, so the pinned statistics are exact (circular
cross-moments) while tails are approximate.

With pin_sigma=True (default) every R-step is followed by a Sigma-step that moves only
the rung's free variables until the static covariance equals the original's, so each
signal rung also sits on the static mean-variance level (mu and Sigma pinned).
A T-step reorders values with a volatility template (vol_beta, auto-calibrated) so
volatility clustering is kept.

Rung groups (in-band):
  full      : common phase                       -> keeps S(w)
  symmetric : common phase, optional conjugation -> keeps Re S(w)  (symmetric part)
  factor    : common phase on factor block F,
              free on complement                  -> keeps F' S(w) F
  diagonal  : independent phase per asset         -> keeps |X_i(w)|^2
  pooled    : any unitary                         -> keeps ||X(w)||^2 = tr S(w)
Out-of-band frequencies are free. The DC bin is always kept (preserves the mean).
"""

import numpy as np
from scipy import stats

_EPS = 1e-12
RUNGS = ("full", "symmetric", "factor", "diagonal", "pooled")


def _phase(z):
    a = np.abs(z)
    safe = np.where(a > _EPS, a, 1.0)
    return np.where(a > _EPS, z / safe, 1.0)


def ewma_vol(R, halflife=20):
    """Two-sided (forward+backward averaged) EWMA volatility of each column, normalized
    to geometric mean 1. Two-sided avoids a lag between template and returns."""
    R = np.asarray(R, float)
    if R.ndim == 1:
        R = R[:, None]
    lam = 0.5 ** (1.0 / halflife)
    x = (R - R.mean(0)) ** 2
    def run(z):
        v = np.empty_like(z)
        v[0] = z[: 4 * halflife].mean(0)
        for t in range(1, len(z)):
            v[t] = lam * v[t - 1] + (1 - lam) * z[t]
        return v
    v = 0.5 * (run(x) + run(x[::-1])[::-1])
    s = np.sqrt(np.maximum(v, _EPS))
    return s / np.exp(np.mean(np.log(s), axis=0))


def abs_acf(R, lags=5):
    """Mean autocorrelation of |r - mean| over lags 1..lags and all assets."""
    R = np.asarray(R, float)
    a = np.abs(R - R.mean(0))
    a = a - a.mean(0)
    den = np.sum(a * a, axis=0)
    return float(np.mean([np.mean(np.sum(a[k:] * a[:-k], axis=0) / den)
                          for k in range(1, lags + 1)]))


def _orthonormal(F):
    Q, _ = np.linalg.qr(np.asarray(F, float))
    return Q


class RungGenerator:
    """
    Parameters
    ----------
    R         : (T, N) array of returns.
    segments  : rung spec. Either a rung name ("full", ...) applied to all frequencies,
                or a list of dicts like
                  {"rung": "full",   "band": (0.0, 0.05)},
                  {"rung": "factor", "band": (0.05, 0.5), "K": 2}   # or "F": (N,K) matrix
                Bands are in cycles/observation; first matching segment wins;
                frequencies not covered by any segment are free.
    tail      : "rank"      - per-asset remap to original values (historical tails)
                "student_t" - per-asset remap to a Student-t(nu) target (tail dial)
                "radial"    - remap Mahalanobis norms of daily vectors (joint tails /
                              co-crashes); with nu, target radius is multivariate-t
                "none"      - no tail step (Gaussian-ish surrogates)
    nu        : degrees of freedom for "student_t" / "radial".
    init      : "block" (stationary block bootstrap), "shuffle" (date permutation),
                or "phase" (common random phase of the original).
    block_len : mean block length for "block" init.
    vol_beta  : strength of volatility clustering imposed in the T-step (0 = off), or
                "auto" (default) - calibrated by bisection so the synthetic paths match the
                original's mean autocorrelation of |r| over lags 1..5 (or vol_target_acf).
                The right value depends strongly on the rung (tight rungs ~0.5, pooled ~0.02).
                Values are rank-remapped by the score  y_t * sigma_t**vol_beta, so large
                target magnitudes land where the template volatility sigma_t is high.
                The marginal distribution stays exact; only the ordering changes.
    vol_template : "init" (EWMA vol of the starting path - default; with block init this
                carries the original's clustering), "original" (EWMA vol of R itself, in
                original time order - keeps the historical vol timeline), or a (T,) / (T, N)
                array you supply (e.g. a simulated GARCH / stress vol path).
    vol_halflife : EWMA half-life (observations) used to build sigma_t from a path.
    pin_sigma : also pin the static covariance Sigma (and mean) exactly, so every signal
                rung sits on the static mean-variance level too. Implemented as a
                Riemannian gradient descent that moves ONLY the rung's free variables
                (per-asset phases for diagonal, the unit sphere per bin for pooled and
                out-of-band bins, the factor-complement sphere for factor); full and
                symmetric rungs pin Sigma automatically. The rung stays exact.
    sigma_tol : stop when the whitened covariance error  max|S0^-1/2 (S - S0) S0^-1/2|
                falls below this (1e-6 ~ correlations exact to 6 decimals).
    sigma_inner / sigma_final : gradient iterations per alternation / on the final step.
    n_iter    : max alternations.
    tol       : stop when relative R/T mismatch falls below this.
    seed      : RNG seed.
    """

    def __init__(self, R, segments="full", tail="rank", nu=None, init="block",
                 block_len=20, n_iter=100, tol=1e-4, seed=None,
                 vol_beta="auto", vol_template="init", vol_halflife=20,
                 vol_target_acf=None, pin_sigma=True, sigma_tol=1e-6,
                 sigma_inner=3, sigma_final=3000):
        R = np.asarray(R, float)
        self.R = R
        self.T, self.N = R.shape
        self.rng = np.random.default_rng(seed)
        self.tail, self.nu, self.init = tail, nu, init
        self.block_len, self.n_iter, self.tol = block_len, n_iter, tol
        self.vol_beta, self.vol_template, self.vol_halflife = vol_beta, vol_template, vol_halflife
        self.sig = None
        self.vol_target_acf = (vol_target_acf if vol_target_acf is not None
                               else abs_acf(R))
        self._beta_auto = (vol_beta == "auto")
        if self._beta_auto:
            self.vol_beta = None

        self.Xo = np.fft.rfft(R, axis=0)
        self.freqs = np.fft.rfftfreq(self.T)
        self.has_nyq = (self.T % 2 == 0)
        self.mu = R.mean(0)
        self.Sigma = np.cov(R, rowvar=False)

        if isinstance(segments, str):
            segments = [{"rung": segments}]
        self.segments = []
        for seg in segments:
            seg = dict(seg)
            if seg["rung"] not in RUNGS:
                raise ValueError(f"unknown rung {seg['rung']!r}")
            seg.setdefault("band", (0.0, 0.5))
            if seg["rung"] == "factor":
                if "F" in seg:
                    seg["F"] = _orthonormal(seg["F"])
                else:
                    K = seg.get("K", 3)
                    vals, vecs = np.linalg.eigh(self.Sigma)
                    seg["F"] = vecs[:, ::-1][:, :K]
            self.segments.append(seg)
        self._assign_bins()
        self._prepare_tails()
        self.pin_sigma, self.sigma_tol = pin_sigma, sigma_tol
        self.sigma_inner, self.sigma_final = sigma_inner, sigma_final
        self._prepare_sigma()

    # ------------------------------------------------------------------ setup
    def _assign_bins(self):
        owner = -np.ones(len(self.freqs), int)
        for s, seg in enumerate(self.segments):
            lo, hi = seg["band"]
            m = (self.freqs > 0) & (self.freqs >= lo) & (self.freqs <= hi) & (owner < 0)
            owner[m] = s
        self.owner = owner
        self.bins = [np.where(owner == s)[0] for s in range(len(self.segments))]

    def _prepare_sigma(self):
        """Bin weights for Sigma = sum_k c_k Re(X_k X_k^H) and each bin's free set."""
        K, T = len(self.freqs), self.T
        c = np.full(K, 2.0 / T ** 2)
        c[0] = 0.0                                   # DC = mean, handled separately
        if self.has_nyq:
            c[-1] = 1.0 / T ** 2
        self.c = c
        self.S0 = np.cov(self.R, rowvar=False, ddof=0)
        w, V = np.linalg.eigh(self.S0)
        self.S0_ih = V @ np.diag(w ** -0.5) @ V.T    # S0^{-1/2}
        free = np.ones(K, bool)                      # out-of-band: fully free (Euclidean)
        free[0] = False
        if self.has_nyq:
            free[-1] = False                         # Nyquist must stay real: frozen
        self.idx_sphere = []
        self.idx_phase = []
        self.idx_comp = []                           # list of (bins, P_perp)
        for seg, seg_bins in zip(self.segments, self.bins):
            idx = seg_bins[free[seg_bins]]
            r = seg["rung"]
            if r == "diagonal":
                self.idx_phase.append(idx)
            elif r == "pooled":
                self.idx_sphere.append(idx)          # pooled pins ||X(w)||: sphere
            elif r == "factor":
                F = seg["F"]
                self.idx_comp.append((idx, np.eye(self.N) - F @ F.T))
            free[seg_bins] = False                   # full / symmetric: nothing free
        self.idx_free = np.where(free)[0]
        self.idx_sphere = np.concatenate(self.idx_sphere) if self.idx_sphere else np.array([], int)
        self.idx_phase = np.concatenate(self.idx_phase) if self.idx_phase else np.array([], int)
        self.sigma_active = self.pin_sigma and (len(self.idx_sphere) + len(self.idx_phase)
                                                + len(self.idx_free)
                                                + sum(len(i) for i, _ in self.idx_comp)) > 0

    def _sigma_of(self, X):
        Z = X * np.sqrt(self.c)[:, None]
        return (Z.T @ np.conj(Z)).real

    def _sigma_loss(self, X):
        Dw = self.S0_ih @ (self._sigma_of(X) - self.S0) @ self.S0_ih
        return float(np.sum(Dw ** 2)), Dw

    def _sigma_retract(self, X, G, eta):
        """Move X by -eta*G along each bin's free manifold (G already tangent)."""
        Xn = X.copy()
        i = self.idx_sphere
        if len(i):
            x, g = X[i], G[i]
            nx = np.linalg.norm(x, axis=1, keepdims=True)
            y = x - eta * g
            Xn[i] = y * nx / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), _EPS)
        i = self.idx_phase
        if len(i):
            Xn[i] = X[i] * np.exp(-1j * eta * G[i].real)      # G holds angle gradients
        i = self.idx_free
        if len(i):
            Xn[i] = X[i] - eta * G[i]
        for i, P in self.idx_comp:                             # complement is free (linear)
            if len(i):
                Xn[i] = X[i] - eta * G[i]
        return Xn

    def _sigma_grad(self, X, Dw):
        """Riemannian gradient of ||Dw||^2 on the free manifold; phase part as angles."""
        Dt = self.S0_ih @ Dw @ self.S0_ih
        Ge = 4.0 * self.c[:, None] * (X @ Dt)                  # Euclidean: dL = Re(G^H dX)
        G = np.zeros_like(X)
        i = self.idx_sphere
        if len(i):
            x, g = X[i], Ge[i]
            rad = np.sum((np.conj(x) * g).real, axis=1, keepdims=True)
            G[i] = g - rad * x / np.maximum(np.sum(np.abs(x) ** 2, 1, keepdims=True), _EPS)
        i = self.idx_phase
        if len(i):
            G[i] = (np.conj(Ge[i]) * 1j * X[i]).real           # dL/dtheta
        i = self.idx_free
        if len(i):
            G[i] = Ge[i]
        for i, P in self.idx_comp:
            if len(i):
                G[i] = Ge[i] @ P.T                             # P symmetric projector
        return G

    def _sigma_step(self, X, max_iter):
        if not self.sigma_active or max_iter <= 0:
            return X
        L, Dw = self._sigma_loss(X)
        eta = getattr(self, "_eta", None)
        for _ in range(max_iter):
            if np.max(np.abs(Dw)) < self.sigma_tol:
                break
            G = self._sigma_grad(X, Dw)
            g2 = float(np.sum(np.abs(G) ** 2))
            if g2 < 1e-300:
                break
            if eta is None:
                eta = 0.5 * L / g2
            while True:
                Xn = self._sigma_retract(X, G, eta)
                Ln, Dn = self._sigma_loss(Xn)
                if Ln < L - 1e-4 * eta * g2:
                    X, L, Dw = Xn, Ln, Dn
                    eta *= 1.5
                    break
                eta *= 0.5
                if eta < 1e-30:
                    self._eta = None
                    return X
        self._eta = eta
        self.last_sigma_err = float(np.max(np.abs(Dw)))
        return X

    def _prepare_tails(self):
        T, N, rng = self.T, self.N, self.rng
        if self.tail == "rank":
            self.target_sorted = np.sort(self.R, axis=0)
        elif self.tail == "student_t":
            if self.nu is None or self.nu <= 2:
                raise ValueError("student_t tail needs nu > 2")
            z = rng.standard_t(self.nu, size=(T, N))
            z = (z - z.mean(0)) / z.std(0)
            sd = self.R.std(0)
            self.target_sorted = self.mu + sd * np.sort(z, axis=0)
        elif self.tail == "radial":
            self.Lc = np.linalg.cholesky(self.Sigma)
            self.Lc_inv = np.linalg.inv(self.Lc)
            d_orig = np.linalg.norm((self.R - self.mu) @ self.Lc_inv.T, axis=1)
            if self.nu is None:
                self.target_radius = np.sort(d_orig)
            else:
                d = np.sqrt(rng.chisquare(N, T) / (rng.chisquare(self.nu, T) / self.nu))
                d *= np.sqrt(np.mean(d_orig ** 2) / np.mean(d ** 2))
                self.target_radius = np.sort(d)
        elif self.tail != "none":
            raise ValueError(f"unknown tail {self.tail!r}")

    # ------------------------------------------------------------------ steps
    def _r_step(self, Y, sigma_iter=0):
        Xn = self._r_project(Y)
        Xn = self._sigma_step(Xn, sigma_iter)
        return np.fft.irfft(Xn, n=self.T, axis=0)

    def _r_project(self, Y):
        Xy = np.fft.rfft(Y, axis=0)
        if self.has_nyq:
            Xy[-1] = Xy[-1].real
        Xn = Xy.copy()
        Xn[0] = self.Xo[0]
        for seg, idx in zip(self.segments, self.bins):
            if len(idx) == 0:
                continue
            xo, xy = self.Xo[idx], Xy[idx]
            rung = seg["rung"]
            if rung == "full":
                c = np.sum(np.conj(xo) * xy, axis=1)
                Xn[idx] = xo * _phase(c)[:, None]
            elif rung == "symmetric":
                c1 = np.sum(np.conj(xo) * xy, axis=1)   # vs e^{i phi} xo
                c2 = np.sum(xo * xy, axis=1)            # vs e^{i phi} conj(xo)
                use2 = np.abs(c2) > np.abs(c1)
                base = np.where(use2[:, None], np.conj(xo), xo)
                Xn[idx] = base * _phase(np.where(use2, c2, c1))[:, None]
            elif rung == "factor":
                F = seg["F"]
                zo, zy = xo @ F, xy @ F
                c = np.sum(np.conj(zo) * zy, axis=1)
                Xn[idx] = xy - zy @ F.T + (zo * _phase(c)[:, None]) @ F.T
            elif rung == "diagonal":
                Xn[idx] = np.abs(xo) * _phase(xy)
            elif rung == "pooled":
                ny = np.linalg.norm(xy, axis=1)
                no = np.linalg.norm(xo, axis=1)
                ok = ny > _EPS
                out = xo.copy()
                out[ok] = xy[ok] * (no[ok] / ny[ok])[:, None]
                Xn[idx] = out
        return Xn

    def _vol_weight(self):
        if self.sig is None or not self.vol_beta:
            return None
        return self.sig ** self.vol_beta

    def _t_step(self, Y):
        w = self._vol_weight()
        if self.tail in ("rank", "student_t"):
            out = np.empty_like(Y)
            score = (Y - self.mu) * w if w is not None else Y
            order = np.argsort(score, axis=0)
            for i in range(self.N):
                out[order[:, i], i] = self.target_sorted[:, i]
            return out
        if self.tail == "radial":
            Z = (Y - self.mu) @ self.Lc_inv.T
            d = np.linalg.norm(Z, axis=1)
            score = d * (w.mean(1) if w is not None else 1.0)
            d_new = np.empty_like(d)
            d_new[np.argsort(score)] = self.target_radius
            Z *= (d_new / np.where(d > _EPS, d, 1.0))[:, None]
            return self.mu + Z @ self.Lc.T
        return Y

    def _initial(self):
        T, rng = self.T, self.rng
        if self.init == "shuffle":
            return self.R[rng.permutation(T)]
        if self.init == "block":
            idx = np.empty(T, int)
            idx[0] = rng.integers(T)
            p = 1.0 / self.block_len
            for t in range(1, T):
                idx[t] = rng.integers(T) if rng.random() < p else (idx[t - 1] + 1) % T
            return self.R[idx]
        if self.init == "phase":
            ph = np.exp(2j * np.pi * rng.random(len(self.freqs)))
            ph[0] = 1.0
            if self.has_nyq:
                ph[-1] = rng.choice([-1.0, 1.0])
            return np.fft.irfft(self.Xo * ph[:, None], n=T, axis=0)
        raise ValueError(f"unknown init {self.init!r}")

    def _template(self, Y0):
        if self.vol_beta is not None and self.vol_beta == 0:
            return None
        vt = self.vol_template
        if isinstance(vt, str):
            if vt == "init":
                if self.init == "phase":
                    raise ValueError('vol_template="init" needs init="block" or "shuffle"')
                sig = ewma_vol(Y0, self.vol_halflife)
            elif vt == "original":
                sig = ewma_vol(self.R, self.vol_halflife)
            else:
                raise ValueError(f"unknown vol_template {vt!r}")
        else:
            sig = np.asarray(vt, float)
            sig = sig / np.exp(np.mean(np.log(sig), axis=0))
        if sig.ndim == 1:
            sig = sig[:, None]
        return np.broadcast_to(sig, (self.T, self.N)) if sig.shape[1] == 1 else sig

    # ------------------------------------------------------------------ public
    def _run(self, Y0, beta, n_iter, sigma_final=None):
        saved = self.vol_beta
        self.vol_beta = beta
        self.sig = self._template(Y0) if beta else None
        Y, hist = Y0, []
        self._eta = None
        for _ in range(n_iter):
            Yr = self._r_step(Y, self.sigma_inner)
            Yt = self._t_step(Yr)
            gap = np.linalg.norm(Yt - Yr) / np.linalg.norm(Yr)
            hist.append(gap)
            Y = Yt
            if gap < self.tol:
                break
        sf = self.sigma_final if sigma_final is None else sigma_final
        Y = self._r_step(Y, sf)                 # finish on the rung (+Sigma): exact
        self.vol_beta = saved
        return Y, hist

    def calibrate_vol_beta(self, n_probe=2, n_iter=None, lo=1e-3, hi=8.0, steps=7):
        """Bisection in log(beta) so mean |r|-ACF (lags 1..5) matches the target."""
        if self.tail == "none":
            return 0.0            # no T-step, so the vol weight has nothing to act on
        target = self.vol_target_acf
        n_iter = n_iter or self.n_iter
        inits = [self._initial() for _ in range(n_probe)]
        # probes use only the inner Sigma iterations on the final step (ACF is insensitive
        # to the last bit of Sigma polishing, and full polishing is the slow part)
        f = lambda b: np.mean([abs_acf(self._run(Y0, b, n_iter, self.sigma_inner)[0])
                               for Y0 in inits]) - target
        if f(0.0) >= 0:
            return 0.0
        if f(hi) <= 0:
            return hi
        a, b = np.log(lo), np.log(hi)
        if f(lo) >= 0:
            return lo
        for _ in range(steps):
            m = 0.5 * (a + b)
            if f(np.exp(m)) < 0:
                a = m
            else:
                b = m
        return float(np.exp(0.5 * (a + b)))

    def generate(self, n=1):
        """Return a list of n synthetic (T, N) arrays and a list of info dicts."""
        if self._beta_auto and self.vol_beta is None:
            self.vol_beta = self.calibrate_vol_beta()
        out, infos = [], []
        for _ in range(n):
            Y0 = self._initial()
            Y, hist = self._run(Y0, self.vol_beta, self.n_iter)
            out.append(Y)
            infos.append({"iterations": len(hist), "final_gap": hist[-1] if hist else 0.0,
                          "gap_history": hist, "vol_beta": self.vol_beta})
        return out, infos

    def pinned_error(self, Y):
        """Max relative error of the pinned spectral statistics, per segment."""
        Xs = np.fft.rfft(Y, axis=0)
        errs = {}
        for s, (seg, idx) in enumerate(zip(self.segments, self.bins)):
            if len(idx) == 0:
                continue
            a, b = self.Xo[idx], Xs[idx]
            rung = seg["rung"]
            if rung in ("full", "symmetric"):
                Sa = np.einsum("fi,fj->fij", a, np.conj(a))
                Sb = np.einsum("fi,fj->fij", b, np.conj(b))
                if rung == "symmetric":
                    Sa, Sb = Sa.real, Sb.real
            elif rung == "factor":
                F = seg["F"]
                za, zb = a @ F, b @ F
                Sa = np.einsum("fi,fj->fij", za, np.conj(za))
                Sb = np.einsum("fi,fj->fij", zb, np.conj(zb))
            elif rung == "diagonal":
                Sa, Sb = np.abs(a) ** 2, np.abs(b) ** 2
            else:
                Sa = np.sum(np.abs(a) ** 2, axis=1)
                Sb = np.sum(np.abs(b) ** 2, axis=1)
            errs[f"{s}:{rung}"] = float(np.max(np.abs(Sa - Sb)) / np.max(np.abs(Sa)))
        return errs


# ====================================================================== diagnostics
def lag_moments(R, L, circular=True):
    """M_k = mean_t r_t r_{t-k}',  k = 1..L.  Returns (L, N, N)."""
    T = R.shape[0]
    if circular:
        return np.stack([R.T @ np.roll(R, k, axis=0) / T for k in range(1, L + 1)])
    return np.stack([R[L:].T @ R[L - k:T - k] / (T - L) for k in range(1, L + 1)])


def sample_strategies(kind, N, L, n, F=None, seed=None):
    """n random lag-matrix stacks A (n, L, N, N) inside subspace W, with ||A||_F = 1."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, L, N, N))
    if kind == "symmetric":
        G = 0.5 * (G + np.swapaxes(G, -1, -2))
    elif kind == "diagonal":
        G = G * np.eye(N)
    elif kind == "pooled":
        G = rng.normal(size=(n, L, 1, 1)) * np.eye(N)
    elif kind == "factor":
        K = F.shape[1]
        C = rng.normal(size=(n, L, K, K))
        G = F @ C @ F.T
    elif kind != "full":
        raise ValueError(kind)
    return G / np.linalg.norm(G.reshape(n, -1), axis=1)[:, None, None, None]


def mean_pnl(A, R, circular=True):
    """Mean P&L <A, M> for each strategy in A (n, L, N, N)."""
    M = lag_moments(R, A.shape[1], circular)
    return np.einsum("nkij,kij->n", A, M)


def rank_agreement(A, R0, R1, circular=True):
    return stats.kendalltau(mean_pnl(A, R0, circular), mean_pnl(A, R1, circular))[0]


def tail_report(R):
    return {"excess_kurtosis": float(np.mean(stats.kurtosis(R, axis=0))),
            "acf_abs_lag1to5": abs_acf(R)}


# ====================================================================== demo
def _simulate(T=2520, N=6, seed=0):
    """VAR(1) with lead-lag, common stochastic vol and t(4) shocks."""
    rng = np.random.default_rng(seed)
    B = np.diag(rng.uniform(-0.05, 0.08, N))
    B[1:, 0] = 0.12                      # asset 0 leads the others
    C0 = 0.4 + 0.6 * np.eye(N)
    Lc = np.linalg.cholesky(C0)
    h, r = 0.0, np.zeros(N)
    out = np.empty((T, N))
    for t in range(T):
        h = 0.98 * h + 0.15 * rng.normal()
        e = rng.standard_t(4, N) / np.sqrt(2.0)
        r = 0.0003 + B @ r + 0.01 * np.exp(h) * (Lc @ e)
        out[t] = r
    return out


if __name__ == "__main__":
    R = _simulate()
    T, N = R.shape
    L, n_strat = 5, 400
    F2 = np.linalg.eigh(np.cov(R, rowvar=False))[1][:, ::-1][:, :2]
    A_full = sample_strategies("full", N, L, n_strat, seed=1)

    print(f"data: T={T}, N={N};  original {tail_report(R)}\n")
    header = f"{'rung':10s} {'pinned err':>10s} {'tau in (circ)':>13s} {'tau in (std)':>12s} " \
             f"{'tau full-A':>10s} {'kurt':>6s} {'acf|r|':>7s} {'path corr':>9s} {'vol_beta':>8s}"
    print(header)
    for rung in RUNGS:
        seg = {"rung": rung, "K": 2} if rung == "factor" else {"rung": rung}
        gen = RungGenerator(R, [seg], tail="rank", init="block", seed=7)
        (Y,), (info,) = gen.generate(1)
        A_in = sample_strategies(rung, N, L, n_strat, F=F2, seed=2)
        err = max(gen.pinned_error(Y).values())
        tc = rank_agreement(A_in, R, Y, circular=True)
        ts = rank_agreement(A_in, R, Y, circular=False)
        tf = rank_agreement(A_full, R, Y, circular=True)
        tr = tail_report(Y)
        pc = np.mean([np.corrcoef(R[:, i], Y[:, i])[0, 1] for i in range(N)])
        print(f"{rung:10s} {err:10.1e} {tc:13.3f} {ts:12.3f} {tf:10.3f} "
              f"{tr['excess_kurtosis']:6.2f} {tr['acf_abs_lag1to5']:7.3f} {pc:9.3f} "
              f"{info['vol_beta']:8.3f}")

    print("\ntail x volatility dials (diagonal rung):")
    for tail, nu in [("none", None), ("rank", None), ("student_t", 3.5), ("radial", 5)]:
        for vb in [0.0, "auto"]:
            gen = RungGenerator(R, "diagonal", tail=tail, nu=nu, vol_beta=vb, seed=7)
            (Y,), (info,) = gen.generate(1)
            tr = tail_report(Y)
            print(f"  {tail:10s} nu={str(nu):5s} vol_beta={info['vol_beta']:.3f} "
                  f"pinned err={max(gen.pinned_error(Y).values()):.1e}  "
                  f"kurt={tr['excess_kurtosis']:6.2f}  acf|r|={tr['acf_abs_lag1to5']:.3f}")

    print("\nmixed lattice rung: full below 0.02 cyc/obs, factor(K=2) to 0.2, pooled above:")
    spec = [{"rung": "full", "band": (0, 0.02)},
            {"rung": "factor", "band": (0.02, 0.2), "K": 2},
            {"rung": "pooled", "band": (0.2, 0.5)}]
    gen = RungGenerator(R, spec, tail="rank", seed=7)
    (Y,), (info,) = gen.generate(1)
    print("  pinned errors:", {k: f"{v:.1e}" for k, v in gen.pinned_error(Y).items()})
    print(f"  iterations={info['iterations']}, final R/T gap={info['final_gap']:.2e}, "
          f"{tail_report(Y)}")
