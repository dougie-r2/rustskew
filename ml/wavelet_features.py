#!/usr/bin/env python3
"""Wavelet-transform features (multi-scale decomposition), point-in-time.

Uses the STATIONARY (undecimated) wavelet transform `swt` so coefficients are
time-localized and shift-invariant — each day's features use only the trailing
window ending at that day. Captures which time-scale the recent dynamics live on;
energy shifting between scales (and falling wavelet entropy) flags regime turns.
"""
import numpy as np
import pandas as pd
import pywt

W = 64          # trailing window (2^6) -> supports up to level 6
LEVEL = 4
WAVELET = "db4"

def _swt_feats(x):
    # x: length-W array. swt needs length divisible by 2^LEVEL; W=64 ok.
    x = np.asarray(x, float)
    x = x - x.mean()
    try:
        coeffs = pywt.swt(x, WAVELET, level=LEVEL, trim_approx=True, norm=True)
    except Exception:
        return None
    # coeffs = [cAn, cDn, ..., cD1]
    cA = coeffs[0]
    details = coeffs[1:]                      # cDn..cD1  (n=LEVEL)
    energies = np.array([np.sum(d ** 2) for d in details], float)
    tot = energies.sum() + 1e-12
    rel = energies / tot                      # relative energy per scale
    went = -np.sum(rel * np.log(rel + 1e-12)) / np.log(len(rel))  # wavelet entropy [0,1]
    out = {}
    # detail levels are ordered coarse->fine here (cD_LEVEL ... cD_1)
    for i, d in enumerate(details):
        lvl = LEVEL - i                        # map to scale index
        out[f"wv_relE_L{lvl}"] = rel[i]
        out[f"wv_edge_L{lvl}"] = d[-1]         # current (most recent) coefficient
    out["wv_entropy"] = went
    out["wv_hi_lo_ratio"] = (rel[-1] + rel[-2]) / (rel[0] + rel[1] + 1e-12)  # fine/coarse energy
    out["wv_approx_edge"] = cA[-1]
    return out

def wavelet_features(ohlc):
    c = ohlc["c"].to_numpy(float)
    logp = np.log(c)
    r = np.diff(logp, prepend=logp[0])
    n = len(c)
    rows = []
    keys = None
    for i in range(n):
        if i < W:
            rows.append(None); continue
        f = _swt_feats(r[i - W:i])
        if keys is None and f is not None:
            keys = list(f.keys())
        rows.append(f)
    data = {"date": ohlc["date"].values}
    if keys is None:
        return pd.DataFrame(data)
    for k in keys:
        data[k] = [ (row[k] if row else np.nan) for row in rows ]
    return pd.DataFrame(data)

if __name__ == "__main__":
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(root, "data", "ohlc_spx.csv"), parse_dates=["date"]).sort_values("date")
    f = wavelet_features(df)
    print(f.tail(2).T)
    print("n_features:", f.shape[1] - 1)
