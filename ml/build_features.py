#!/usr/bin/env python3
"""Assemble the aligned daily feature matrix + drawdown-onset target labels.

Output: ml/features.csv   (one row per SPX trading day, 2017-01-03 -> latest)

Lookahead discipline:
  - price/vol/technical features use close-of-day t (act next open) -> as-of t
  - external released-next-day series (FRED, DIX/GEX) are shifted +1 trading day
  - target y_h* uses ONLY future drawdown-start dates
"""
import os, glob, math
import numpy as np
import pandas as pd
import swisseph as swe
import sys
sys.path.insert(0, os.path.dirname(__file__))
from ephemeris import features_for
from physics_features import physics_features
from wavelet_features import wavelet_features
from hmm_features import hmm_features
from advanced_features import advanced_features, cross_asset_features
from creative_features import creative_features

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
RAW = os.path.join(HERE, "raw")
PROD = bool(os.environ.get("PROD"))   # PROD=1 -> skip non-auto/non-predictive blocks (astro, season)
LONG = bool(os.environ.get("LONG"))   # LONG=1 -> extended 1990-2026 history + ZigZag-5% auto labels
ADV = bool(os.environ.get("ADV"))     # ADV=1 -> advanced 'C' features (tested: no gain; off by default)
CRE = bool(os.environ.get("CRE"))     # CRE=1 -> creative physics features (freq/entropy/inertia/spring/fractal)

# ---- finalized ground-truth drawdown windows (>=5% peak-to-trough) ----
WINDOWS = [
    ("2018-01-29","2018-02-09"),("2018-03-13","2018-04-02"),("2018-10-03","2018-12-26"),
    ("2019-05-01","2019-06-03"),("2019-07-29","2019-08-07"),("2019-09-19","2019-10-03"),
    ("2020-02-20","2020-03-23"),("2020-06-10","2020-06-15"),("2020-09-02","2020-09-24"),
    ("2020-10-13","2020-10-30"),("2021-02-16","2021-03-04"),("2021-09-03","2021-10-04"),
    ("2021-11-22","2021-12-03"),("2022-01-04","2022-01-24"),("2022-02-10","2022-02-24"),
    ("2022-03-30","2022-06-17"),("2022-08-18","2022-10-13"),("2023-02-02","2023-03-13"),
    ("2023-08-01","2023-08-18"),("2023-09-01","2023-10-27"),("2024-04-01","2024-04-19"),
    ("2024-07-16","2024-08-05"),("2024-12-06","2025-01-13"),("2025-02-20","2025-04-07"),
    ("2025-10-29","2025-11-21"),("2026-02-11","2026-03-30"),
]

def load_ohlc():
    fn = "ohlc_spx_full.csv" if LONG else "ohlc_spx.csv"
    df = pd.read_csv(os.path.join(ROOT, "data", fn), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def zigzag_events(df, th=0.05):
    """ZigZag 5% swing detector -> [(top_date, bottom_date)] for each >=5% peak-to-trough
    down-leg. Reproduces ~85% of the user's manual labels (validated)."""
    h = df["h"].to_numpy(float); l = df["l"].to_numpy(float)
    dt = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    piv = []                       # (idx, 'H'|'L')
    mode = "H"; ext_i = 0; ext = h[0]
    for i in range(1, len(df)):
        if mode == "H":
            if h[i] > ext:
                ext = h[i]; ext_i = i
            elif l[i] <= ext * (1 - th):
                piv.append((ext_i, "H")); mode = "L"; ext = l[i]; ext_i = i
        else:
            if l[i] < ext:
                ext = l[i]; ext_i = i
            elif h[i] >= ext * (1 + th):
                piv.append((ext_i, "L")); mode = "H"; ext = h[i]; ext_i = i
    ev = []
    for a, b in zip(piv, piv[1:]):
        if a[1] == "H" and b[1] == "L" and l[b[0]] <= h[a[0]] * (1 - th):
            ev.append((dt[a[0]], dt[b[0]]))
    return ev

def technicals(df):
    c, h, l, o = df["c"], df["h"], df["l"], df["o"]
    lr = np.log(c / c.shift(1))
    f = pd.DataFrame({"date": df["date"]})
    for n in (1, 2, 3, 5, 10, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 10, 20, 60):
        f[f"rv_{n}"] = lr.rolling(n).std() * math.sqrt(252)
    f["rv_ratio_10_60"] = f["rv_10"] / f["rv_60"]
    f["rv_ratio_5_20"] = f["rv_5"] / f["rv_20"]
    f["rvov_20"] = f["rv_10"].rolling(20).std()                 # vol-of-vol
    for n in (20, 50, 100, 200):
        ma = c.rolling(n).mean()
        f[f"dist_ma{n}"] = c / ma - 1.0
    f["ma50_slope_10"] = c.rolling(50).mean().pct_change(10)
    f["ma200_slope_20"] = c.rolling(200).mean().pct_change(20)
    # RSI 14
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # Bollinger %b 20
    m20 = c.rolling(20).mean(); s20 = c.rolling(20).std()
    f["bb_pctb_20"] = (c - (m20 - 2 * s20)) / (4 * s20)
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (macd - sig) / c
    # drawdown-from-rolling-high
    for n in (20, 60, 252):
        f[f"dd_high_{n}"] = c / c.rolling(n).max() - 1.0
    f["days_since_high_252"] = (c.rolling(252).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True))
    # range vols
    f["parkinson_10"] = (np.log(h / l) ** 2).rolling(10).mean().pipe(lambda x: np.sqrt(x / (4 * math.log(2)) * 252))
    gk = 0.5 * np.log(h / l) ** 2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2
    f["garman_klass_10"] = np.sqrt(gk.rolling(10).mean() * 252)
    # streaks / breadth-ish on price
    sign = np.sign(d.fillna(0))
    f["up_days_5"] = (sign > 0).rolling(5).sum()
    # consecutive down streak
    streak = np.zeros(len(c));
    for i in range(1, len(c)):
        streak[i] = streak[i-1] + 1 if d.iloc[i] < 0 else 0
    f["down_streak"] = streak
    # overnight gap
    gap = o / c.shift(1) - 1.0
    f["gap"] = gap
    f["gap_abs_10"] = gap.abs().rolling(10).mean()
    # return skew/kurt
    f["ret_skew_20"] = lr.rolling(20).skew()
    f["ret_kurt_20"] = lr.rolling(20).kurt()
    f["atr_pct_14"] = (pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
                       .rolling(14).mean() / c)
    return f

def zscore(s, n=252):
    return (s - s.rolling(n).mean()) / s.rolling(n).std()

def load_raw(name):
    p = os.path.join(RAW, f"{name}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    return df

def main():
    ohlc = load_ohlc()
    cal = ohlc[["date"]].copy()                 # master trading calendar
    feats = technicals(ohlc)
    feats = feats.merge(physics_features(ohlc), on="date", how="left")  # econophysics block
    feats = feats.merge(wavelet_features(ohlc), on="date", how="left")  # multi-scale wavelet
    feats = feats.merge(hmm_features(ohlc), on="date", how="left")      # causal HMM regimes
    if ADV:
        feats = feats.merge(advanced_features(ohlc), on="date", how="left")  # full 16 C feats (experimental)
    else:
        feats = feats.merge(advanced_features(ohlc, hawkes_only=True), on="date", how="left")  # Hawkes (borderline, default-on)
    if CRE:
        feats = feats.merge(creative_features(ohlc), on="date", how="left")  # creative physics (experimental)

    # ---- market/macro series aligned to calendar (FRED) ----
    # as-of close (daily) get no lag; weekly conditions get a publish-lag shift.
    def add(name, lag=0):
        raw = load_raw(name)
        if raw is None:
            print(f"  [skip] {name} (no file)")
            return None
        m = pd.merge_asof(cal, raw, on="date")  # as-of merge, ffill stale values
        col = m[name]
        if lag:
            col = col.shift(lag)
        feats[name] = col.values
        return col

    asof = ["vix","vix3m","vxn","hy_oas","ig_oas","ccc_oas","t10y2y","t10y3m",
            "dgs10","dgs2","dollar_broad","wti",
            # newly recovered (CBOE + cooled-down Yahoo)
            "vvix","skew","vix9d","vxtlt","cor1m","cor3m",
            "spy","rsp","hyg","lqd","ief","xlp","xlu","gold","copper","dxy"]
    for nm in asof:
        add(nm, lag=0)
    weekly = ["nfci","anfci","stlfsi"]   # weekly series, ~3-day publish lag
    for nm in weekly:
        add(nm, lag=3)

    # ---- DIX / GEX (released next morning -> lag 1) ----
    dix = pd.read_csv(os.path.join(ROOT, "DIX.csv"), parse_dates=["date"]).sort_values("date")
    dm = pd.merge_asof(cal, dix[["date","dix","gex"]], on="date")
    feats["dix"] = dm["dix"].shift(1).values
    feats["gex"] = dm["gex"].shift(1).values

    # ---- derived cross-series features ----
    def col(n): return feats[n] if n in feats else None
    if col("vix") is not None and col("vix3m") is not None:
        feats["vix_term"] = feats["vix"] / feats["vix3m"]      # >=1 inversion
    if col("vix") is not None and col("vix9d") is not None:
        feats["vix_term_short"] = feats["vix9d"] / feats["vix"]
    if col("vvix") is not None and col("vix") is not None:
        feats["vvix_vix"] = feats["vvix"] / feats["vix"]
    if col("vix") is not None:
        feats["vix_z"] = zscore(pd.Series(feats["vix"].values)).values
        feats["vix_chg_5"] = pd.Series(feats["vix"].values).pct_change(5).values
    if col("skew") is not None:
        feats["skew_z"] = zscore(pd.Series(feats["skew"].values)).values
    if col("hy_oas") is not None:
        s = pd.Series(feats["hy_oas"].values)
        feats["hy_oas_chg_20"] = s.diff(20).values
        feats["hy_oas_z"] = zscore(s).values
    if col("hy_oas") is not None and col("ig_oas") is not None:
        feats["hy_minus_ig"] = feats["hy_oas"] - feats["ig_oas"]
    if col("hyg") is not None and col("lqd") is not None:
        feats["hyg_lqd"] = feats["hyg"] / feats["lqd"]
        feats["hyg_lqd_mom20"] = pd.Series((feats["hyg"]/feats["lqd"]).values).pct_change(20).values
    if col("copper") is not None and col("gold") is not None:
        feats["copper_gold"] = feats["copper"] / feats["gold"]
    if col("xlu") is not None and col("xlk") is not None:
        feats["xlu_xlk_mom20"] = pd.Series((feats["xlu"]/feats["xlk"]).values).pct_change(20).values  # defensive rotation
    if col("dgs10") is not None and col("dgs2") is not None and "t10y2y" not in feats:
        feats["t10y2y"] = feats["dgs10"] - feats["dgs2"]
    # newly-recovered derived features (vol-of-vol, tail, bond-vol, breadth, credit)
    if col("vvix") is not None:
        feats["vvix_z"] = zscore(pd.Series(feats["vvix"].values)).values
    if col("skew") is not None:
        feats["skew_chg20"] = pd.Series(feats["skew"].values).diff(20).values
    if col("vix9d") is not None:
        feats["vix9d_vix"] = feats["vix9d"] / feats["vix"]      # short-end term structure
    if col("vxtlt") is not None:
        s = pd.Series(feats["vxtlt"].values)
        feats["vxtlt_z"] = zscore(s).values
        feats["vxtlt_chg20"] = s.pct_change(20).values
    if col("rsp") is not None and col("spy") is not None:
        feats["rsp_spy"] = feats["rsp"] / feats["spy"]
        feats["rsp_spy_mom20"] = pd.Series((feats["rsp"] / feats["spy"]).values).pct_change(20).values
    if col("hyg") is not None and col("ief") is not None:
        feats["hyg_ief_mom20"] = pd.Series((feats["hyg"] / feats["ief"]).values).pct_change(20).values
    for d in ("xlp", "xlu"):
        if col(d) is not None and col("spy") is not None:
            feats[f"{d}_spy_mom20"] = pd.Series((feats[d] / feats["spy"]).values).pct_change(20).values
    if col("dxy") is not None:
        feats["dxy_mom20"] = pd.Series(feats["dxy"].values).pct_change(20).values
    # implied correlation (systemic risk)
    if col("cor1m") is not None:
        s = pd.Series(feats["cor1m"].values)
        feats["cor1m_z"] = zscore(s).values
        feats["cor1m_chg20"] = s.diff(20).values
    if col("cor1m") is not None and col("cor3m") is not None:
        feats["cor_term"] = feats["cor1m"] / feats["cor3m"]     # >1 = front-loaded stress

    # ---- calendar SEASONALITY (cyclic sin/cos) — non-predictive, PROD skips ----
    if not PROD:
        d = pd.to_datetime(cal["date"])
        doy = d.dt.dayofyear.values; mon = d.dt.month.values; dow = d.dt.dayofweek.values
        dom = d.dt.day.values; dim = d.dt.days_in_month.values
        two_pi = 2 * math.pi
        feats["sea_doy_sin"] = np.sin(two_pi * doy / 365.25)
        feats["sea_doy_cos"] = np.cos(two_pi * doy / 365.25)
        feats["sea_month_sin"] = np.sin(two_pi * mon / 12)
        feats["sea_month_cos"] = np.cos(two_pi * mon / 12)
        feats["sea_dow_sin"] = np.sin(two_pi * dow / 5)
        feats["sea_dow_cos"] = np.cos(two_pi * dow / 5)
        feats["sea_tom"] = ((dom <= 3) | (dom >= dim - 2)).astype(int)   # turn-of-month
        feats["sea_dom_frac"] = dom / dim
        def _days_to_opex(ts):
            import calendar as _cal
            y, m = ts.year, ts.month
            c = _cal.monthcalendar(y, m)
            fridays = [wk[_cal.FRIDAY] for wk in c if wk[_cal.FRIDAY] != 0]
            third = pd.Timestamp(y, m, fridays[2])
            if ts <= third:
                return (third - ts).days
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            c2 = _cal.monthcalendar(ny, nm)
            f2 = [wk[_cal.FRIDAY] for wk in c2 if wk[_cal.FRIDAY] != 0]
            return (pd.Timestamp(ny, nm, f2[2]) - ts).days
        feats["sea_days_to_opex"] = [ _days_to_opex(ts) for ts in d ]

    # FRED-derived (sources actually available)
    if col("vix3m") is not None:
        feats["vix_term_inv"] = (feats["vix"] >= feats["vix3m"]).astype(int)
    if col("vxn") is not None:
        feats["vxn_vix"] = feats["vxn"] / feats["vix"]
    if col("dollar_broad") is not None:
        feats["dollar_mom20"] = pd.Series(feats["dollar_broad"].values).pct_change(20).values
    if col("wti") is not None:
        feats["wti_mom20"] = pd.Series(feats["wti"].values).pct_change(20).values
    for nm in ("nfci","anfci","stlfsi","t10y2y","t10y3m"):
        if col(nm) is not None:
            feats[f"{nm}_chg20"] = pd.Series(feats[nm].values).diff(20).values
    feats["dix_z"] = zscore(pd.Series(feats["dix"].values)).values
    feats["dix_chg_5"] = pd.Series(feats["dix"].values).diff(5).values
    feats["gex_z"] = zscore(pd.Series(feats["gex"].values)).values
    feats["gex_neg"] = (pd.Series(feats["gex"].values) < 0).astype(int).values
    # C cross-asset / info-theory features (rolling corr/coskew/downside-beta/MI/TE) — need merged vix/rates/usd
    if ADV and all(col(x) is not None for x in ("vix", "dgs10", "dollar_broad")):
        lp = np.log(ohlc["c"].to_numpy(float)); ret_arr = np.diff(lp, prepend=lp[0])
        gv = lambda nm: pd.Series(feats[nm].values).ffill().bfill().to_numpy()
        for k, v in cross_asset_features(ret_arr, gv("vix"), gv("dgs10"), gv("dollar_broad")).items():
            feats[k] = v

    # ---- ephemeris (astrology) features — non-predictive, PROD skips ----
    if not PROD:
        eph_rows = [features_for(dd.year, dd.month, dd.day) for dd in cal["date"]]
        eph = pd.DataFrame(eph_rows)
        eph["date"] = cal["date"].values
        feats = feats.merge(eph, on="date", how="left")
        if "ph_bradley_P" in feats:
            feats["ph_bradley_slope5"] = feats["ph_bradley_P"].diff(5)
            feats["ph_bradley_vs_ma21"] = feats["ph_bradley_P"] - feats["ph_bradley_P"].rolling(21).mean()

    # ---- target labels ----
    windows = zigzag_events(ohlc) if LONG else WINDOWS
    dates = cal["date"].reset_index(drop=True)
    pos = {d: i for i, d in enumerate(dates)}
    starts = [pd.Timestamp(s) for s, _ in windows]
    # nearest trading-day index >= start
    start_idx = []
    for s in starts:
        idx = dates.searchsorted(s)
        if idx < len(dates):
            start_idx.append(idx)
    start_idx = sorted(set(start_idx))
    in_dd = np.zeros(len(dates), dtype=int)
    for s, e in windows:
        si = dates.searchsorted(pd.Timestamp(s))
        ei = dates.searchsorted(pd.Timestamp(e), side="right") - 1
        in_dd[si:ei+1] = 1
    for H in (3, 5, 10):
        y = np.zeros(len(dates), dtype=int)
        for i in range(len(dates)):
            for si in start_idx:
                if i < si <= i + H:
                    y[i] = 1
                    break
        feats[f"y_h{H}"] = y
    feats["in_dd"] = in_dd

    # ---- turning-point labels: TOP (drawdown start) / BOTTOM (drawdown end) ----
    # a day is a TOP/BOTTOM if within +/-TOL trading days of a labeled start/end.
    TOL = 2
    y_top = np.zeros(len(dates), dtype=int)
    y_bot = np.zeros(len(dates), dtype=int)
    for s, e in windows:
        si = dates.searchsorted(pd.Timestamp(s))
        ei = dates.searchsorted(pd.Timestamp(e))
        for j in range(max(0, si - TOL), min(len(dates), si + TOL + 1)):
            y_top[j] = 1
        for j in range(max(0, ei - TOL), min(len(dates), ei + TOL + 1)):
            y_bot[j] = 1
    feats["y_top"] = y_top
    feats["y_bottom"] = y_bot

    # trim off the rolling-warmup year (252d). LONG starts 1990 -> keep from 1991.
    trim_start = "1991-01-02" if LONG else "2017-01-03"
    feats = feats[feats["date"] >= trim_start].reset_index(drop=True)
    if LONG:
        name = "features_long_cre.csv" if CRE else ("features_long_adv.csv" if ADV else "features_long.csv")
    else:
        name = "features.csv"
    out = os.path.join(HERE, name)
    feats.to_csv(out, index=False)
    n_feat = sum(1 for c in feats.columns if c not in ("date","y_h3","y_h5","y_h10","in_dd"))
    print(f"wrote {out}: {len(feats)} rows x {len(feats.columns)} cols ({n_feat} features)")
    for H in (3,5,10):
        y = feats[f"y_h{H}"]; tr = feats[feats["in_dd"]==0]
        print(f"  y_h{H}: {int(y.sum())} positives total | base rate (out-of-dd) = {tr[f'y_h{H}'].mean():.3%}")
    print(f"  in_dd days: {int(feats['in_dd'].sum())} / {len(feats)}")
    # coverage of key features
    print("  non-null coverage of key cols:")
    for cc in ["vix","vix3m","vvix","skew","move","hy_oas","t10y2y","nfci","dix","gex","ph_mercury_retro"]:
        if cc in feats:
            print(f"    {cc:16} {feats[cc].notna().mean():.1%}")

if __name__ == "__main__":
    main()
