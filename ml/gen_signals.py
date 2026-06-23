#!/usr/bin/env python3
"""Generate dashboard turning-point signals -> data/signals.csv (date,top,bottom).

Walk-forward OOS probabilities (honest, no lookahead) for the chosen models:
  TOP  = technical+physics+wavelet+hmm   (best top detector, ~AUC 0.67) -> orange
  BOTTOM = everything except astrology    (best bottom detector, ~AUC 0.92) -> teal
Flagging is CAUSAL (real-time usable): a day fires on the RISING EDGE — the first day its
OOS prob crosses above an EXPANDING-WINDOW (past-only) quantile threshold; re-arms after the
prob drops back below. No future data is ever used (matches ml/reflag.py).
"""
import os, sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
sys.path.insert(0, os.path.dirname(__file__))
from train import load, feat_cols, group_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [7, 23, 101]
# production groups = only daily-auto-updatable sources (OHLC+FRED+CBOE+DIX);
# astrology/seasonality/breadth excluded (non-predictive or no daily source).
PROD_GROUPS = {"technical","physics","wavelet","hmm","vol","macro",
               "credit","rates","conditions","flow","corr"}
TOP_GROUPS = {"technical", "physics", "wavelet", "hmm"}
BOT_GROUPS = set(PROD_GROUPS)
TOP_Q, BOT_Q = 0.93, 0.93     # top ~7% of days

def cb(s):
    return CatBoostClassifier(iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=8.0,
        loss_function="Logloss", auto_class_weights="Balanced", random_seed=s,
        verbose=0, allow_writing_files=False)

def wf(X, y, seed, n_splits=8, embargo=5, init_frac=0.25):
    n = len(X); start = int(n * init_frac)
    b = np.linspace(start, n, n_splits + 1).astype(int)
    oos = np.full(n, np.nan)
    for k in range(n_splits):
        te0, te1 = b[k], b[k+1]; tr = te0 - embargo
        if tr < 100 or te1 <= te0 or y.iloc[:tr].sum() < 4:
            continue
        m = cb(seed); m.fit(X.iloc[:tr], y.iloc[:tr])
        oos[te0:te1] = m.predict_proba(X.iloc[te0:te1])[:, 1]
    return oos

def oos_prob(df, want, target):
    cols = [c for c in feat_cols(df) if group_of(c) in want]
    X = df[cols]; y = df[target].astype(int)
    return np.nanmean([wf(X, y, s) for s in SEEDS], axis=0)

def flag(prob, q, minhist=252):
    """CAUSAL flagging: rising edge + EXPANDING-WINDOW quantile (past probabilities only).
    Fires the first day the prob crosses above the threshold; re-arms after it drops back.
    No future data — replaces the old +/-W local-max + full-sample-quantile (hindsight-only)."""
    p = np.asarray(prob, float)
    out = np.zeros(len(p), dtype=int)
    prev_above = False
    seen = []                                  # past valid probs only
    for i in range(len(p)):
        if np.isnan(p[i]):
            prev_above = False
            continue
        if len(seen) >= minhist:
            thr = np.quantile(seen, q)         # expanding window: data up to YESTERDAY only
            above = p[i] >= thr
            if above and not prev_above:
                out[i] = 1
            prev_above = above
        seen.append(p[i])                      # add today AFTER deciding (no peeking)
    return out

def main():
    df = load().reset_index(drop=True)
    top_p = oos_prob(df, TOP_GROUPS, "y_top")
    bot_p = oos_prob(df, BOT_GROUPS, "y_bottom")
    out = pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "top": flag(top_p, TOP_Q),
        "bottom": flag(bot_p, BOT_Q),
    })
    path = os.path.join(ROOT, "data", "signals.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}: {len(out)} rows | top signals={int(out['top'].sum())} "
          f"bottom signals={int(out['bottom'].sum())}")
    print("recent top dates:", list(out[out.top==1].date.tail(6)))
    print("recent bottom dates:", list(out[out.bottom==1].date.tail(6)))

if __name__ == "__main__":
    main()
