#!/usr/bin/env python3
"""Advanced math/physics features (checklist 'C', history-computable) from SPX OHLC.
All trailing-window, point-in-time. Cross-asset 'C' features that need the merged
aux series (VIX/rates/USD) are computed in build_features, not here."""
import numpy as np
import pandas as pd

W = 60      # estimation window


def _embed(x, m=3, tau=1):
    n = len(x) - (m - 1) * tau
    if n <= 1:
        return np.empty((0, m))
    return np.column_stack([x[i * tau:i * tau + n] for i in range(m)])


def _lyapunov(x):
    """Rosenstein-lite largest Lyapunov exponent on a window."""
    Y = _embed(np.asarray(x, float), m=3, tau=1)
    n = len(Y)
    if n < 12:
        return np.nan
    D = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    # exclude temporally-close neighbours
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        D[i, lo:hi] = np.inf
    nn = np.argmin(D, axis=1)
    K = 6
    div = []
    for k in range(1, K):
        d = []
        for i in range(n - k):
            j = nn[i]
            if j + k < n:
                dist = np.linalg.norm(Y[i + k] - Y[j + k])
                if dist > 0:
                    d.append(np.log(dist))
        if d:
            div.append(np.mean(d))
    if len(div) < 2:
        return np.nan
    return np.polyfit(np.arange(len(div)), div, 1)[0]


def _rqa(x, rr=0.15):
    """Recurrence quantification: (determinism, laminarity) on a window."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 12:
        return np.nan, np.nan
    D = np.abs(x[:, None] - x[None, :])
    thr = np.quantile(D, rr)
    R = (D <= thr).astype(np.int8)
    np.fill_diagonal(R, 0)
    # determinism: fraction of recurrence points on diagonal lines length>=2
    diag_pts = 0
    for k in range(1, n):
        d = np.diag(R, k)
        # count points in runs of length>=2
        run = 0
        for v in d:
            if v:
                run += 1
            else:
                if run >= 2:
                    diag_pts += run
                run = 0
        if run >= 2:
            diag_pts += run
    total = R.sum() / 2
    det = diag_pts / total if total > 0 else np.nan
    # laminarity: vertical lines length>=2
    vert_pts = 0
    for c in range(n):
        run = 0
        for v in R[:, c]:
            if v:
                run += 1
            else:
                if run >= 2:
                    vert_pts += run
                run = 0
        if run >= 2:
            vert_pts += run
    lam = vert_pts / (R.sum()) if R.sum() > 0 else np.nan
    return det, lam


def _dfa(x, scales):
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    F = []
    for s in scales:
        if len(x) < 2 * s:
            F.append(np.nan); continue
        nseg = len(x) // s
        rms = []
        for i in range(nseg):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            fit = np.polyval(np.polyfit(t, seg, 1), t)
            rms.append(np.sqrt(np.mean((seg - fit) ** 2)))
        F.append(np.sqrt(np.mean(np.square(rms))) if rms else np.nan)
    return np.array(F)


def _mfdfa_width(x):
    """Width of the multifractal singularity spectrum (proxy: spread of h(q))."""
    scales = [4, 8, 16]
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    hq = []
    for q in (-3, 0, 3):
        Fq = []
        for s in scales:
            if len(x) < 2 * s:
                Fq.append(np.nan); continue
            nseg = len(x) // s
            rms2 = []
            for i in range(nseg):
                seg = y[i * s:(i + 1) * s]
                t = np.arange(s)
                fit = np.polyval(np.polyfit(t, seg, 1), t)
                rms2.append(np.mean((seg - fit) ** 2))
            rms2 = np.array(rms2)
            if q == 0:
                Fq.append(np.exp(0.5 * np.mean(np.log(rms2 + 1e-12))))
            else:
                Fq.append(np.mean(rms2 ** (q / 2)) ** (1 / q))
        Fq = np.array(Fq)
        ok = np.isfinite(Fq) & (Fq > 0)
        if ok.sum() >= 2:
            hq.append(np.polyfit(np.log(np.array(scales)[ok]), np.log(Fq[ok]), 1)[0])
    return (max(hq) - min(hq)) if len(hq) >= 2 else np.nan


def _sampen(x, m=2, r=0.2):
    x = np.asarray(x, float); n = len(x)
    if n < m + 2:
        return np.nan
    r = r * x.std()
    if r == 0:
        return np.nan
    def phi(mm):
        c = 0; t = 0
        for i in range(n - mm):
            tmpl = x[i:i + mm]
            for j in range(i + 1, n - mm):
                t += 1
                if np.max(np.abs(tmpl - x[j:j + mm])) <= r:
                    c += 1
        return c / t if t else 0
    a, b = phi(m + 1), phi(m)
    return -np.log(a / b) if a > 0 and b > 0 else np.nan


def _mse(x, scale=2):
    x = np.asarray(x, float)
    cg = np.array([x[i:i + scale].mean() for i in range(0, len(x) - scale + 1, scale)])
    return _sampen(cg, 2, 0.2)


def _cusum(x):
    x = np.asarray(x, float)
    s = (x - x.mean()) / (x.std() + 1e-12)
    c = np.cumsum(s)
    return (c.max() - c.min()) / len(x)


def advanced_features(ohlc, dd_trigger_thresh=0.03, hawkes_only=False):
    c = ohlc["c"].to_numpy(float)
    logp = np.log(c)
    r = np.diff(logp, prepend=logp[0])
    n = len(c)
    out = {"date": ohlc["date"].values}
    if not hawkes_only:                        # the 15 weak C feats (experimental, ADV only)
        keys = ["lyapunov", "rqa_det", "rqa_lam", "mfdfa_w", "mse2", "cusum"]
        cols = {f"adv_{k}": np.full(n, np.nan) for k in keys}
        for i in range(W, n):
            win = r[i - W:i]
            cols["adv_lyapunov"][i] = _lyapunov(win)
            det, lam = _rqa(win)
            cols["adv_rqa_det"][i] = det
            cols["adv_rqa_lam"][i] = lam
            cols["adv_mfdfa_w"][i] = _mfdfa_width(win)
            cols["adv_mse2"][i] = _mse(win[-40:], 2)
            cols["adv_cusum"][i] = _cusum(win)
        out.update(cols)

    # Hawkes self-exciting intensity of past >=3% drawdown triggers (causal).
    roll_max = pd.Series(c).cummax().to_numpy()  # not used; use rolling 20d high
    hi20 = pd.Series(ohlc["h"]).rolling(20, min_periods=1).max().to_numpy()
    dd = c / hi20 - 1.0
    trig = (dd <= -dd_trigger_thresh) & (np.r_[0, dd[:-1]] > -dd_trigger_thresh)  # cross down
    beta = 1.0 / 20.0
    inten = np.zeros(n); acc = 0.0
    for i in range(n):
        acc *= np.exp(-beta)
        inten[i] = acc            # intensity from PAST triggers only (strictly before i via prior accumulation)
        if trig[i]:
            acc += 1.0
    out["adv_hawkes"] = inten
    return pd.DataFrame(out)


def _disc(x, bins=4):
    q = np.quantile(x, np.linspace(0, 1, bins + 1)); q[0] -= 1e-9; q[-1] += 1e-9
    return np.clip(np.digitize(x, q[1:-1]), 0, bins - 1)

def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return -np.sum(p * np.log(p))

def cross_asset_features(ret, vix, bond, usd, W=60, bins=4):
    """C cross-asset/info-theory features from aligned series (point-in-time)."""
    n = len(ret)
    rs = pd.Series(ret)
    vchg = pd.Series(np.r_[0, np.diff(vix)])
    bchg = pd.Series(np.r_[0, np.diff(bond)])
    uchg = pd.Series(np.r_[0, np.diff(usd)])
    out = {}
    out["adv_corr_vix"] = rs.rolling(W).corr(vchg).to_numpy()
    out["adv_corr_bond"] = rs.rolling(W).corr(bchg).to_numpy()
    out["adv_corr_usd"] = rs.rolling(W).corr(uchg).to_numpy()
    # co-skewness of SPX returns w.r.t. VIX changes; downside beta vs VIX-up days
    cosk = np.full(n, np.nan); dbeta = np.full(n, np.nan)
    mi = np.full(n, np.nan); te = np.full(n, np.nan)
    rv = ret; vv = vix
    for i in range(W, n):
        rw = rv[i - W:i]; vw = vchg.values[i - W:i]
        sr, sv = rw.std(), vw.std()
        if sr > 0 and sv > 0:
            cosk[i] = np.mean((rw - rw.mean()) * (vw - vw.mean()) ** 2) / (sr * sv ** 2)
            up = vw > 0                      # downside beta: SPX sensitivity when VIX rises
            if up.sum() > 5:
                dbeta[i] = np.polyfit(vw[up], rw[up], 1)[0]
        # mutual information & transfer entropy (VIX -> SPX), binned
        vlev = vv[i - W:i]
        try:
            dr = _disc(rw, bins); dv = _disc(vlev, bins)
            jc = np.histogram2d(dv, dr, bins=bins)[0]
            mi[i] = _entropy(jc.sum(1)) + _entropy(jc.sum(0)) - _entropy(jc.ravel())
            # TE: I(R_t ; V_{t-1} | R_{t-1})  via binned entropies
            r1, r0 = dr[1:], dr[:-1]; v0 = dv[:-1]
            h_r1r0 = _entropy(np.histogram2d(r1, r0, bins=bins)[0].ravel())
            h_r0 = _entropy(np.histogram(r0, bins=bins)[0])
            h_r1r0v0 = _entropy(np.histogramdd(np.c_[r1, r0, v0], bins=bins)[0].ravel())
            h_r0v0 = _entropy(np.histogram2d(r0, v0, bins=bins)[0].ravel())
            te[i] = (h_r1r0 - h_r0) - (h_r1r0v0 - h_r0v0)
        except Exception:
            pass
    out["adv_coskew_vix"] = cosk
    out["adv_downbeta_vix"] = dbeta
    out["adv_mi_vix"] = mi
    out["adv_te_vix_spx"] = te
    return out


if __name__ == "__main__":
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(root, "data", "ohlc_spx_full.csv"), parse_dates=["date"]).sort_values("date")
    import time
    t0 = time.time()
    f = advanced_features(df.tail(400).reset_index(drop=True))
    print(f.dropna().tail(3).T)
    print("n_features:", f.shape[1] - 1, "| 400-row timing:", round(time.time() - t0, 1), "s")
