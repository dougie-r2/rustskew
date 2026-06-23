#!/usr/bin/env python3
"""Creative physics/signal features (frequency waves, extra entropies, inertia,
spring/mean-reversion, fractal, damping) from SPX OHLC. Trailing-window, point-in-time.
Deliberately NON-overlapping with physics_features.py (which has Hurst/DFA/sampen/specent)."""
import numpy as np
import pandas as pd

W = 60


def _detrend(x):
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, 1), t)


# ---- frequency / wave (Hz-like) ----
def _spectrum(x):
    x = _detrend(np.asarray(x, float))
    ps = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    return freqs[1:], ps[1:]            # drop DC


def _dom_period(x):
    f, p = _spectrum(x)
    if p.sum() == 0:
        return np.nan
    return 1.0 / f[np.argmax(p)]        # dominant cycle length (bars)


def _spec_centroid(x):
    f, p = _spectrum(x)
    return np.sum(f * p) / p.sum() if p.sum() > 0 else np.nan


def _peak_ratio(x):
    f, p = _spectrum(x)
    return p.max() / p.sum() if p.sum() > 0 else np.nan   # cycle strength


def _zcr(x):
    x = _detrend(np.asarray(x, float))
    return np.mean(np.abs(np.diff(np.sign(x)))) / 2.0


# ---- extra entropies ----
def _apen(x, m=2, r=0.2):
    x = np.asarray(x, float); n = len(x); r = r * x.std()
    if n < m + 2 or r == 0:
        return np.nan
    def phi(mm):
        cnt = []
        for i in range(n - mm + 1):
            t = x[i:i + mm]
            c = sum(np.max(np.abs(t - x[j:j + mm])) <= r for j in range(n - mm + 1))
            cnt.append(c / (n - mm + 1))
        return np.mean(np.log(cnt))
    return phi(m) - phi(m + 1)


def _renyi2(x, bins=8):
    h, _ = np.histogram(x, bins=bins)
    p = h / h.sum()
    p = p[p > 0]
    return -np.log(np.sum(p ** 2))      # collision (order-2) entropy


def _svd_ent(x, m=4):
    x = np.asarray(x, float); n = len(x) - m + 1
    if n < m:
        return np.nan
    emb = np.column_stack([x[i:i + n] for i in range(m)])
    s = np.linalg.svd(emb, compute_uv=False)
    s = s / s.sum()
    s = s[s > 0]
    return -np.sum(s * np.log(s)) / np.log(len(s))


# ---- inertia / memory ----
def _ac_decay(x):
    x = _detrend(np.asarray(x, float))
    if x.std() == 0:
        return np.nan
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / ac[0]
    below = np.where(ac < 1 / np.e)[0]
    return below[0] if len(below) else len(x)   # lag where ACF<1/e (memory length)


def _persist(x):
    s = np.sign(np.diff(np.asarray(x, float)))
    return abs(s.sum()) / max(1, len(s))         # directional persistence (inertia)


# ---- spring / mean-reversion (Ornstein-Uhlenbeck) ----
def _ou_halflife(x):
    x = np.asarray(x, float)
    dx = np.diff(x); xl = x[:-1]
    if xl.std() == 0:
        return np.nan
    beta = np.polyfit(xl, dx, 1)[0]              # dx = beta*x + c  -> theta=-beta
    if beta >= 0:
        return W * 2.0                            # not mean-reverting -> cap
    return min(W * 2.0, np.log(2) / -beta)


def _var_ratio(x, k=5):
    r = np.diff(np.asarray(x, float))
    if len(r) < k + 2 or r.var() == 0:
        return np.nan
    vk = np.var(np.add.reduceat(r, np.arange(0, len(r) - len(r) % k, k)))
    return vk / (k * r.var())                     # <1 mean-revert, >1 trend


def _restoring(x):
    x = np.asarray(x, float)
    dev = x[-1] - x.mean()                         # displacement from equilibrium
    vel = x[-1] - x[-2] if len(x) > 1 else 0.0
    return -np.sign(dev) * vel                     # restoring if moving back toward mean


# ---- fractal dimension ----
def _higuchi(x, kmax=6):
    x = np.asarray(x, float); n = len(x); L = []
    for k in range(1, kmax + 1):
        Lk = []
        for mi in range(k):
            idx = np.arange(mi, n, k)
            if len(idx) < 2:
                continue
            ll = np.sum(np.abs(np.diff(x[idx]))) * (n - 1) / (len(idx) * k)
            Lk.append(ll)
        if Lk:
            L.append(np.log(np.mean(Lk)))
    if len(L) < 2:
        return np.nan
    return -np.polyfit(np.log(1.0 / np.arange(1, len(L) + 1)), L, 1)[0]


def _katz(x):
    x = np.asarray(x, float)
    d = np.max(np.abs(x - x[0]))
    Lsum = np.sum(np.sqrt(1 + np.diff(x) ** 2))
    if d == 0 or Lsum == 0:
        return np.nan
    n = len(x) - 1
    return np.log(n) / (np.log(n) + np.log(d / Lsum))


def _damping(x):
    """damping ratio proxy: decay rate of |detrended| envelope."""
    x = np.abs(_detrend(np.asarray(x, float))) + 1e-9
    t = np.arange(len(x))
    slope = np.polyfit(t, np.log(x), 1)[0]
    return -slope                                  # >0 = damping (energy bleeding out)


def creative_features(ohlc):
    c = ohlc["c"].to_numpy(float)
    logp = np.log(c)
    r = np.diff(logp, prepend=logp[0])
    n = len(c)
    keys = ["dom_period", "spec_centroid", "peak_ratio", "zcr", "apen", "renyi2",
            "svd_ent", "ac_decay", "persist", "ou_halflife", "var_ratio",
            "restoring", "higuchi", "katz", "damping"]
    cols = {f"cre_{k}": np.full(n, np.nan) for k in keys}
    for i in range(W, n):
        wr = r[i - W:i]; wp = logp[i - W:i]
        cols["cre_dom_period"][i] = _dom_period(wp)
        cols["cre_spec_centroid"][i] = _spec_centroid(wr)
        cols["cre_peak_ratio"][i] = _peak_ratio(wr)
        cols["cre_zcr"][i] = _zcr(wr)
        cols["cre_apen"][i] = _apen(wr[-40:])
        cols["cre_renyi2"][i] = _renyi2(wr)
        cols["cre_svd_ent"][i] = _svd_ent(wr)
        cols["cre_ac_decay"][i] = _ac_decay(wr)
        cols["cre_persist"][i] = _persist(wp)
        cols["cre_ou_halflife"][i] = _ou_halflife(wp)
        cols["cre_var_ratio"][i] = _var_ratio(wp)
        cols["cre_restoring"][i] = _restoring(wp)
        cols["cre_higuchi"][i] = _higuchi(wp)
        cols["cre_katz"][i] = _katz(wp)
        cols["cre_damping"][i] = _damping(wp)
    out = {"date": ohlc["date"].values}; out.update(cols)
    return pd.DataFrame(out)


if __name__ == "__main__":
    import os, time
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(root, "data", "ohlc_spx_full.csv"), parse_dates=["date"]).sort_values("date")
    t0 = time.time(); f = creative_features(df.tail(300).reset_index(drop=True))
    print(f.dropna().tail(2).T)
    print("n_features:", f.shape[1] - 1, "| 300-row:", round(time.time() - t0, 1), "s")
