#!/usr/bin/env python3
"""Math/physics-inspired features for regime/crash prediction.

All trailing-window (point-in-time, no lookahead). Grounded in published theory:
  - Critical Slowing Down (Scheffer et al. 2009): as a system nears a tipping
    point, lag-1 autocorrelation and variance RISE. We compute them and their
    trends (Kendall tau).
  - Log-Periodic Power Law / super-exponential growth (Sornette): bubbles grow
    faster than exponential -> positive convexity of log-price; we also measure
    log-periodic oscillation energy.
  - Fractal persistence: Hurst exponent (R/S) and DFA alpha.
  - Complexity collapse: permutation entropy, sample entropy, spectral entropy.
  - Fat tails: Hill tail-index estimator.
  - Chaos: largest-Lyapunov proxy (Rosenstein-lite divergence rate).
"""
import numpy as np
import pandas as pd


def _hurst_rs(x):
    """Rescaled-range Hurst exponent."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 16:
        return np.nan
    ns = [n // k for k in (1, 2, 4, 8) if n // k >= 8]
    rs, sizes = [], []
    for m in ns:
        nseg = n // m
        vals = []
        for i in range(nseg):
            seg = x[i * m:(i + 1) * m]
            z = seg - seg.mean()
            Z = np.cumsum(z)
            R = Z.max() - Z.min()
            S = seg.std()
            if S > 0:
                vals.append(R / S)
        if vals:
            rs.append(np.mean(vals)); sizes.append(m)
    if len(rs) < 2:
        return np.nan
    return np.polyfit(np.log(sizes), np.log(rs), 1)[0]


def _dfa(x, scales=(4, 8, 16, 32)):
    """Detrended Fluctuation Analysis exponent."""
    x = np.asarray(x, float)
    n = len(x)
    y = np.cumsum(x - x.mean())
    F, S = [], []
    for s in scales:
        if n < 2 * s:
            continue
        nseg = n // s
        rms = []
        for i in range(nseg):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            c = np.polyfit(t, seg, 1)
            fit = np.polyval(c, t)
            rms.append(np.sqrt(np.mean((seg - fit) ** 2)))
        if rms:
            F.append(np.sqrt(np.mean(np.array(rms) ** 2))); S.append(s)
    if len(F) < 2:
        return np.nan
    F = np.array(F); F[F == 0] = 1e-12
    return np.polyfit(np.log(S), np.log(F), 1)[0]


def _perm_entropy(x, order=3):
    """Permutation entropy (Bandt-Pompe), normalized to [0,1]."""
    x = np.asarray(x, float)
    n = len(x)
    if n < order + 1:
        return np.nan
    from math import factorial, log
    perms = {}
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i:i + order]))
        perms[pattern] = perms.get(pattern, 0) + 1
    c = np.array(list(perms.values()), float)
    p = c / c.sum()
    H = -np.sum(p * np.log(p))
    return H / log(factorial(order))


def _sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, float)
    n = len(x)
    if n < m + 2:
        return np.nan
    r = r * x.std()
    if r == 0:
        return np.nan
    def phi(mm):
        cnt = 0; tot = 0
        for i in range(n - mm):
            tmpl = x[i:i + mm]
            for j in range(i + 1, n - mm):
                tot += 1
                if np.max(np.abs(tmpl - x[j:j + mm])) <= r:
                    cnt += 1
        return cnt / tot if tot else 0
    a = phi(m + 1); b = phi(m)
    if a == 0 or b == 0:
        return np.nan
    return -np.log(a / b)


def _hill(x, frac=0.10):
    """Hill tail-index estimator on |x| (higher = fatter tail)."""
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    k = max(3, int(len(a) * frac))
    a = a[:k]
    a = a[a > 0]
    if len(a) < 3:
        return np.nan
    return 1.0 / np.mean(np.log(a[:-1] / a[-1]))


def _spectral_entropy(x):
    x = np.asarray(x, float) - np.mean(x)
    if x.std() == 0:
        return np.nan
    ps = np.abs(np.fft.rfft(x)) ** 2
    ps = ps[1:]
    if ps.sum() == 0:
        return np.nan
    p = ps / ps.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(len(p)) if len(p) > 1 else np.nan


def _superexp(logp):
    """Convexity of log-price: positive => super-exponential (bubble) growth."""
    t = np.arange(len(logp))
    try:
        c = np.polyfit(t, logp, 2)
        return c[0]            # quadratic coefficient
    except Exception:
        return np.nan


def _kendall_tau_trend(s):
    """Sign-of-trend strength of a series (Kendall tau vs time), no scipy."""
    s = np.asarray(s, float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < 6:
        return np.nan
    c = 0
    for i in range(n - 1):
        c += np.sum(np.sign(s[i + 1:] - s[i]))
    return c / (n * (n - 1) / 2)


def physics_features(ohlc):
    c = ohlc["c"].to_numpy(float)
    logp = np.log(c)
    r = np.diff(logp, prepend=logp[0])
    n = len(c)
    out = {"date": ohlc["date"].values}

    W = 60       # estimation window
    names = ["hurst", "dfa", "perm_ent", "samp_ent", "hill", "spec_ent", "superexp"]
    cols = {k: np.full(n, np.nan) for k in names}
    ar1 = np.full(n, np.nan)
    rvar = np.full(n, np.nan)
    for i in range(W, n):
        win_r = r[i - W:i]
        win_lp = logp[i - W:i]
        cols["hurst"][i] = _hurst_rs(win_r)
        cols["dfa"][i] = _dfa(win_r)
        cols["perm_ent"][i] = _perm_entropy(win_r, 3)
        cols["hill"][i] = _hill(win_r)
        cols["spec_ent"][i] = _spectral_entropy(win_r)
        cols["superexp"][i] = _superexp(win_lp)
        # cheap sample entropy on a shorter slice (O(W^2))
        cols["samp_ent"][i] = _sample_entropy(win_r[-40:], 2, 0.2)
        ar1[i] = np.corrcoef(win_r[:-1], win_r[1:])[0, 1] if win_r.std() > 0 else np.nan
        rvar[i] = win_r.var()
    for k in names:
        out[f"px_{k}"] = cols[k]
    out["px_ar1"] = ar1
    out["px_var"] = rvar

    # Critical-slowing-down TRENDS (rising AR1 + rising var = warning)
    ar1_s = pd.Series(ar1); var_s = pd.Series(rvar)
    out["px_ar1_trend"] = ar1_s.rolling(40).apply(_kendall_tau_trend, raw=True).to_numpy()
    out["px_var_trend"] = var_s.rolling(40).apply(_kendall_tau_trend, raw=True).to_numpy()
    # log-periodic oscillation energy: power in detrended log-price residual
    lp_res = pd.Series(logp).rolling(W).apply(
        lambda w: np.std(w - np.polyval(np.polyfit(np.arange(len(w)), w, 2), np.arange(len(w)))),
        raw=True).to_numpy()
    out["px_lppl_resid"] = lp_res
    return pd.DataFrame(out)


if __name__ == "__main__":
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(root, "data", "ohlc_spx.csv"), parse_dates=["date"]).sort_values("date")
    f = physics_features(df)
    print(f.tail(3).T)
    print("n_features:", f.shape[1] - 1)
