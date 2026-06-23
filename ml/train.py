#!/usr/bin/env python3
"""Train CatBoost to predict drawdown onset, with purged walk-forward CV.

Honest evaluation discipline:
  - rows inside a drawdown window (in_dd==1) are removed from modeling
  - time-ordered expanding-window folds, NO shuffling
  - an embargo gap purges train rows whose H-day label horizon could overlap test
  - all reported metrics are OUT-OF-SAMPLE (concatenated across test folds)
  - compared against the base rate (a skill-less model scores ~base on PR-AUC)

Also runs a group ablation to find the best feature combination and to isolate
whether the astrology (ph_*) block adds anything beyond market/macro features.
"""
import os, sys, json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(__file__)
HORIZON = int(os.environ.get("H", "5"))
META = {"date", "y_h3", "y_h5", "y_h10", "in_dd", "y_top", "y_bottom"}

def load():
    df = pd.read_csv(os.path.join(HERE, "features.csv"), parse_dates=["date"])
    return df

def feat_cols(df):
    return [c for c in df.columns if c not in META]

def group_of(c):
    if c.startswith("ph_"):
        return "astro"
    if c.startswith("px_"):
        return "physics"
    if c.startswith("wv_"):
        return "wavelet"
    if c.startswith("hmm_"):
        return "hmm"
    if c.startswith("sea_"):
        return "season"
    if c.startswith("cor"):
        return "corr"
    if c.startswith(("dix", "gex")):
        return "flow"
    if c.startswith(("vix", "vvix", "skew", "vxn", "vxtlt")):
        return "vol"
    if c.startswith(("hy_", "ig_", "ccc_", "hyg", "lqd", "ief")):
        return "credit"
    if c.startswith(("rsp", "spy")):
        return "breadth"
    if c.startswith(("t10y", "dgs")):
        return "rates"
    if c.startswith(("nfci", "anfci", "stlfsi")):
        return "conditions"
    if c.startswith(("dollar", "wti", "dxy", "gold", "copper", "xlp", "xlu")):
        return "macro"
    return "technical"

def cb():
    return CatBoostClassifier(
        iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=8.0,
        loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=7, verbose=0, allow_writing_files=False)

def purged_walkforward(X, y, n_splits=6, embargo=10, init_frac=0.4):
    n = len(X)
    start = int(n * init_frac)
    bounds = np.linspace(start, n, n_splits + 1).astype(int)
    oos = np.full(n, np.nan)
    importances = np.zeros(X.shape[1])
    nfit = 0
    for k in range(n_splits):
        te0, te1 = bounds[k], bounds[k + 1]
        tr_end = te0 - embargo
        if tr_end < 100 or te1 <= te0:
            continue
        tr = slice(0, tr_end)
        if y.iloc[tr].sum() < 5:          # need some positives to learn
            continue
        m = cb()
        m.fit(X.iloc[tr], y.iloc[tr])
        oos[te0:te1] = m.predict_proba(X.iloc[te0:te1])[:, 1]
        importances += m.get_feature_importance()
        nfit += 1
    if nfit:
        importances /= nfit
    return oos, importances

def evaluate(df, cols, y, label):
    X = df[cols].reset_index(drop=True)
    oos, imp = purged_walkforward(X, y.reset_index(drop=True))
    mask = ~np.isnan(oos)
    yy, pp = y.reset_index(drop=True)[mask], oos[mask]
    base = yy.mean()
    auc = roc_auc_score(yy, pp) if yy.nunique() > 1 else float("nan")
    ap = average_precision_score(yy, pp) if yy.nunique() > 1 else float("nan")
    lift = ap / base if base > 0 else float("nan")
    print(f"  {label:26} feats={len(cols):3d}  AUC={auc:.3f}  PR-AUC={ap:.3f}  "
          f"base={base:.3f}  lift={lift:.2f}x  (n_oos={mask.sum()})")
    return dict(label=label, n_feats=len(cols), auc=auc, pr_auc=ap, base=base,
                lift=lift, n_oos=int(mask.sum()), oos=oos, imp=imp, cols=cols)

def main():
    df = load()
    df = df[df["in_dd"] == 0].reset_index(drop=True)   # model only out-of-drawdown days
    y = df[f"y_h{HORIZON}"].astype(int)
    cols = feat_cols(df)
    groups = {}
    for c in cols:
        groups.setdefault(group_of(c), []).append(c)
    print(f"Horizon H={HORIZON} trading days. modeling rows={len(df)} "
          f"positives={int(y.sum())} base_rate={y.mean():.3%}")
    print("groups:", {g: len(v) for g, v in groups.items()})

    print("\n=== ALL FEATURES ===")
    full = evaluate(df, cols, y, "ALL")

    print("\n=== SINGLE GROUP ===")
    singles = {g: evaluate(df, v, y, f"only:{g}") for g, v in sorted(groups.items())}

    print("\n=== LEAVE-ONE-GROUP-OUT ===")
    for g in sorted(groups):
        rest = [c for c in cols if group_of(c) != g]
        evaluate(df, rest, y, f"drop:{g}")

    print("\n=== MARKET-ONLY vs MARKET+ASTRO ===")
    market = [c for c in cols if group_of(c) != "astro"]
    mkt = evaluate(df, market, y, "market(no-astro)")
    both = evaluate(df, cols, y, "market+astro")
    astro_only = singles["astro"]
    print(f"\n  astro adds: dAUC={both['auc']-mkt['auc']:+.3f}  "
          f"dPR-AUC={both['pr_auc']-mkt['pr_auc']:+.3f}")

    # top features by mean CV importance (full model)
    imp = sorted(zip(full["cols"], full["imp"]), key=lambda t: -t[1])
    print("\n=== TOP 25 FEATURES (full-model mean CV importance) ===")
    for name, v in imp[:25]:
        print(f"  {v:7.3f}  {name:24s} [{group_of(name)}]")

    # persist results
    out = {
        "horizon": HORIZON, "rows": len(df), "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "results": {k: {kk: vv for kk, vv in r.items() if kk not in ("oos", "imp", "cols")}
                    for k, r in [("ALL", full), ("market", mkt), ("market_astro", both),
                                 *[(f"only_{g}", s) for g, s in singles.items()]]},
        "top_features": [(n, float(v), group_of(n)) for n, v in imp[:40]],
    }
    with open(os.path.join(HERE, f"results_h{HORIZON}.json"), "w") as f:
        json.dump(out, f, indent=2)
    # OOS predictions for the full model (for inspection / plotting)
    pred = df[["date", f"y_h{HORIZON}"]].copy()
    pred["p_full"] = full["oos"]
    pred["p_market"] = mkt["oos"]
    pred.to_csv(os.path.join(HERE, f"oos_pred_h{HORIZON}.csv"), index=False)
    print(f"\nwrote results_h{HORIZON}.json and oos_pred_h{HORIZON}.csv")

if __name__ == "__main__":
    main()
