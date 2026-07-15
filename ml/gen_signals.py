#!/usr/bin/env python3
"""Generate dashboard turning-point signals -> data/signals.csv (date,top,bottom).

PUBLISHED SIGNALS ARE IMMUTABLE. The pipeline is append-only:

  data/probs.csv    point-in-time OOS probabilities (date,top_prob,bottom_prob).
                    Each daily run trains on data up to (today - EMBARGO) and predicts
                    ONLY the new day(s), then appends. Rows, once written, never change.
  data/signals.csv  pure function of the frozen prob history: rising-edge over an
                    EXPANDING-WINDOW (past-only) quantile threshold, exactly flag().
                    Because past probs are frozen, past flags can never repaint.

Bootstrap: if data/probs.csv is missing, the full prob history is generated once with
walk-forward CV (honest OOS, no lookahead) and frozen from then on. Deleting probs.csv
forces a full rebuild — research only; it rewrites published history.

Models: TOP = technical+physics+wavelet+hmm, BOTTOM = all PROD groups (see CLAUDE.md).
"""
import os, sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
sys.path.insert(0, os.path.dirname(__file__))
from train import load, feat_cols, group_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS_PATH = os.path.join(ROOT, "data", "probs.csv")
SIGNALS_PATH = os.path.join(ROOT, "data", "signals.csv")
SEEDS = [7, 23, 101]
EMBARGO = 5
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

def wf(X, y, seed, n_splits=8, embargo=EMBARGO, init_frac=0.25):
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

def live_prob(df, want, target, i):
    """Point-in-time prob for row i: train on [0, i-EMBARGO), predict row i.
    Same guards and seed-averaging as wf() — the live continuation of the backtest."""
    cols = [c for c in feat_cols(df) if group_of(c) in want]
    X = df[cols]; y = df[target].astype(int)
    tr = i - EMBARGO
    if tr < 100 or y.iloc[:tr].sum() < 4:
        return float("nan")
    ps = []
    for s in SEEDS:
        m = cb(s); m.fit(X.iloc[:tr], y.iloc[:tr])
        ps.append(m.predict_proba(X.iloc[[i]])[:, 1][0])
    return float(np.mean(ps))

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

def bootstrap(df):
    print("bootstrap: no probs.csv — generating full walk-forward prob history (one-time)")
    return pd.DataFrame({
        "date": df["date"].dt.strftime("%Y-%m-%d"),
        "top_prob": oos_prob(df, TOP_GROUPS, "y_top"),
        "bottom_prob": oos_prob(df, BOT_GROUPS, "y_bottom"),
    })

def append_new(df, probs):
    last = probs["date"].iloc[-1]
    dstr = df["date"].dt.strftime("%Y-%m-%d")
    new_pos = np.flatnonzero(dstr.to_numpy() > last)
    if len(new_pos) == 0:
        print(f"no new dates after {last} — signals unchanged")
        return probs
    rows = []
    for i in new_pos:
        rows.append({"date": dstr.iloc[i],
                     "top_prob": live_prob(df, TOP_GROUPS, "y_top", i),
                     "bottom_prob": live_prob(df, BOT_GROUPS, "y_bottom", i)})
        print(f"appended {dstr.iloc[i]}: top={rows[-1]['top_prob']:.4f} "
              f"bottom={rows[-1]['bottom_prob']:.4f}")
    return pd.concat([probs, pd.DataFrame(rows)], ignore_index=True)

def main():
    df = load().reset_index(drop=True)
    if os.path.exists(PROBS_PATH):
        # round_trip parser: read floats to the exact stored value so rewriting the
        # file is byte-stable (git diff of probs.csv shows only appended lines)
        probs = append_new(df, pd.read_csv(PROBS_PATH, dtype={"date": str},
                                           float_precision="round_trip"))
    else:
        probs = bootstrap(df)
    probs.to_csv(PROBS_PATH, index=False)
    out = pd.DataFrame({
        "date": probs["date"],
        "top": flag(probs["top_prob"].to_numpy(float), TOP_Q),
        "bottom": flag(probs["bottom_prob"].to_numpy(float), BOT_Q),
    })
    out.to_csv(SIGNALS_PATH, index=False)
    print(f"wrote {SIGNALS_PATH}: {len(out)} rows | top signals={int(out['top'].sum())} "
          f"bottom signals={int(out['bottom'].sum())}")
    print("recent top dates:", list(out[out.top==1].date.tail(6)))
    print("recent bottom dates:", list(out[out.bottom==1].date.tail(6)))

if __name__ == "__main__":
    main()
