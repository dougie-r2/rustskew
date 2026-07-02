"""
Replicate & honestly validate the TradingView 'SPX Top' complacency logic on our data:
  fire when  SPX >= 95th pct of trailing 252d  AND  VIX <= 15.5  AND  COR1M <= 9
(thresholds taken AS-IS from the chart, no refitting). Rising edge + 10d cooldown.
Also a de-magic-numbered variant using expanding percentiles (VIX<=35%ile, COR1M<=10%ile).
Metrics identical to ml/corr_*.py: prec4/prec5 = P(-4%/-5% from fire close within 20d),
zigzag-4% episode recall, dd20@fire. Windows: 2007-2012 (incl. 2007 top) and 2013-2026.

Run:  uv run --with pandas --with numpy python ml/complacency_top.py
"""
import numpy as np
import pandas as pd

f = pd.read_csv("ml/features_long.csv", parse_dates=["date"])
ohlc = pd.read_csv("data/ohlc_spx_full.csv", parse_dates=["date"])
df = f.merge(ohlc, on="date", how="inner").reset_index(drop=True)
c = df["c"].values
n = len(df)
dates = df["date"]

H = 20
fwdmin = np.full(n, np.nan)
for i in range(n - 1):
    fwdmin[i] = c[i + 1 : min(n, i + 1 + H)].min() / c[i] - 1
full_win = np.arange(n) < n - H
y4 = np.where(np.isnan(fwdmin), np.nan, (fwdmin <= -0.04).astype(float))
y5 = np.where(np.isnan(fwdmin), np.nan, (fwdmin <= -0.05).astype(float))
y4[~full_win] = np.nan
y5[~full_win] = np.nan
dd20 = np.array([c[i] / c[max(0, i - 20) : i + 1].max() - 1 for i in range(n)])

def zigzag_declines(px, thr=0.04):
    eps, trend, ext, pk = [], 0, 0, None
    for i in range(1, len(px)):
        if trend >= 0:
            if px[i] >= px[ext]:
                ext = i
            elif px[i] / px[ext] - 1 <= -thr:
                pk, trend, ext = ext, -1, i
        else:
            if px[i] <= px[ext]:
                ext = i
            elif px[i] / px[ext] - 1 >= thr:
                eps.append((pk, ext))
                trend, ext = 1, i
    if trend < 0 and pk is not None and px[ext] / px[pk] - 1 <= -thr:
        eps.append((pk, ext))
    return eps

episodes = zigzag_declines(c, 0.04)

def cooldown(sig, k=10):
    out = np.zeros(n, int)
    last = -10**9
    for i in range(n):
        if sig[i] and i - last > k:
            out[i] = 1
            last = i
    return out

def onset(mask):
    mask = np.asarray(mask, bool)
    return cooldown((mask & ~np.concatenate([[False], mask[:-1]])).astype(int))

def eval_sig(fire, name, lo, hi, verbose=False):
    w = (dates >= lo).values & (dates <= hi).values & ~np.isnan(y4)
    idx = np.where((np.asarray(fire) == 1) & w)[0]
    ev_eps = [(pk, tr) for pk, tr in episodes if w[pk]]
    yrs = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
    if len(idx) == 0:
        print(f"{name:34s} [{lo[:4]}-{hi[:4]}]  no fires  ({len(ev_eps)} eps)")
        return
    caught, leads = 0, []
    for pk, tr in ev_eps:
        hits = [i for i in idx if pk - 10 <= i < tr]
        if hits:
            caught += 1
            leads.append(hits[0] - pk)
    print(f"{name:34s} [{lo[:4]}-{hi[:4]}] {len(idx):3d} ({len(idx)/yrs:4.1f}/yr)  "
          f"prec4 {np.nanmean(y4[idx]):.2f}  prec5 {np.nanmean(y5[idx]):.2f}  "
          f"ep-recall {caught}/{len(ev_eps)}={caught/len(ev_eps):.2f}  "
          f"lead pk {np.median(leads) if leads else float('nan'):+.0f}d  "
          f"dd20@fire {np.median(dd20[idx]):+.1%}")
    if verbose:
        for i in idx:
            fm = fwdmin[i]
            tag = "?" if not full_win[i] else ("HIT4" if fm <= -0.04 else ("hit5" if fm <= -0.05 else "miss"))
            print(f"    {dates.iloc[i].date()}  vix {df['vix'].iloc[i]:.1f} cor1m {df['cor1m'].iloc[i]:.1f} "
                  f"spx%ile {spx_pct[i]:.2f}  fwd20 {fm:+.1%}  {tag}")

# ---------------- conditions ----------------
spx_pct = pd.Series(c).rolling(252, min_periods=252).apply(
    lambda a: (a[:-1] <= a[-1]).mean(), raw=True).values
vix = df["vix"].values
cor = df["cor1m"].values

# exact chart thresholds
cond_exact = (spx_pct >= 0.95) & (vix <= 15.5) & (cor <= 9)
print(f"exact-condition days: {int(np.nansum(cond_exact))}")

# de-magic-numbered variant: expanding percentiles (causal)
def expanding_pct(x):
    out = np.full(n, np.nan)
    seen = []
    for i in range(n):
        if np.isnan(x[i]):
            continue
        if len(seen) >= 252:
            out[i] = (np.asarray(seen) <= x[i]).mean()
        seen.append(x[i])
    return out

vix_ep = expanding_pct(vix)
cor_ep = expanding_pct(cor)
cond_pct = (spx_pct >= 0.95) & (vix_ep <= 0.35) & (cor_ep <= 0.10)

print("\n=== exact TV thresholds (95% / VIX<=15.5 / COR1M<=9), onset+cd10 ===")
sig_e = onset(np.nan_to_num(cond_exact.astype(float)).astype(bool))
eval_sig(sig_e, "TV exact", "2007-01-03", "2012-12-31")
eval_sig(sig_e, "TV exact", "2013-01-02", "2026-12-31", verbose=True)

print("\n=== percentile variant (spx>=95% / vix<=35%ile / cor1m<=10%ile) ===")
sig_p = onset(np.nan_to_num(cond_pct.astype(float)).astype(bool))
eval_sig(sig_p, "pct variant", "2007-01-03", "2012-12-31")
eval_sig(sig_p, "pct variant", "2013-01-02", "2026-12-31", verbose=True)

# ablation: which leg does the work?
print("\n=== ablation (2013-2026, onset+cd10) ===")
eval_sig(onset((spx_pct >= 0.95) & (vix <= 15.5)), "spx95 & vix<=15.5 only", "2013-01-02", "2026-12-31")
eval_sig(onset((spx_pct >= 0.95) & (cor <= 9)), "spx95 & cor<=9 only", "2013-01-02", "2026-12-31")
eval_sig(onset((vix <= 15.5) & (cor <= 9)), "vix & cor only (no price)", "2013-01-02", "2026-12-31")
eval_sig(onset(spx_pct >= 0.95), "spx95 only", "2013-01-02", "2026-12-31")

# user's two example windows + the chart's three marks
print("\nfires near key dates (exact | pct variant):")
for lo, hi, tag in [("2025-01-15", "2025-03-05", "user ex1 (2025-02-18)"),
                    ("2026-05-01", "2026-06-18", "user ex2 (2026-06-02)"),
                    ("2025-07-01", "2025-08-31", "chart mark Aug 2025"),
                    ("2026-01-01", "2026-02-15", "chart mark Jan 2026")]:
    w = (dates >= lo) & (dates <= hi)
    fe = [str(d.date()) for d in dates[w & (pd.Series(sig_e) == 1)]]
    fp = [str(d.date()) for d in dates[w & (pd.Series(sig_p) == 1)]]
    print(f"  {tag:26s}: exact {fe}  pct {fp}")
