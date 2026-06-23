#!/usr/bin/env python3
"""Hidden-Markov-Model regime features (causal, no lookahead).

A Gaussian HMM on [log-return, realized-vol] gives regime posteriors. To stay
point-in-time we (a) REFIT only on past data, periodically, and (b) read the
FILTERED posterior at day t as the last row of predict_proba over the trailing
window ending at t (smoothed == filtered at the final timestep). States are
re-identified each refit by sorting on the volatility-dimension mean so
"bear" = highest-vol state consistently.

Markov add-ons: regime persistence (self-transition prob of current state) and
a transition-intensity proxy (1 - max posterior = how 'mixed'/unstable the state).
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from hmmlearn.hmm import GaussianHMM

K = 3
REFIT_EVERY = 126
MIN_TRAIN = 300
WIN = 252

def hmm_features(ohlc):
    c = ohlc["c"].to_numpy(float)
    logret = np.diff(np.log(c), prepend=np.log(c[0]))
    rv = pd.Series(logret).rolling(20).std().bfill().to_numpy()
    obs = np.column_stack([logret, rv])
    n = len(obs)

    bear = np.full(n, np.nan); bull = np.full(n, np.nan)
    mid = np.full(n, np.nan); persist = np.full(n, np.nan); instab = np.full(n, np.nan)
    exp_ret = np.full(n, np.nan)

    model = None; order = None; mu = None; sd = None
    for t in range(n):
        need_refit = t >= MIN_TRAIN and (model is None or t % REFIT_EVERY == 0)
        if need_refit:
            X = obs[:t]                            # PAST ONLY
            mu = X.mean(0); sd = X.std(0) + 1e-9
            try:
                m = GaussianHMM(n_components=K, covariance_type="diag",
                                n_iter=60, random_state=0, tol=1e-3)
                m.fit((X - mu) / sd)
                order = np.argsort(m.means_[:, 1])   # asc vol: [bull, mid, bear]
                model = m
            except Exception:
                pass
        if model is not None and t >= MIN_TRAIN:
            w = obs[max(0, t - WIN):t + 1]
            try:
                post = model.predict_proba((w - mu) / sd)[-1]   # filtered at t
            except Exception:
                continue
            bull[t] = post[order[0]]; mid[t] = post[order[1]]; bear[t] = post[order[2]]
            cur = int(np.argmax(post))
            persist[t] = model.transmat_[cur, cur]
            instab[t] = 1.0 - post.max()
            # expected next-day standardized return of current state (de-standardized)
            exp_ret[t] = model.means_[cur, 0] * sd[0] + mu[0]

    return pd.DataFrame({
        "date": ohlc["date"].values,
        "hmm_bear_prob": bear, "hmm_bull_prob": bull, "hmm_mid_prob": mid,
        "hmm_persist": persist, "hmm_instability": instab, "hmm_exp_ret": exp_ret,
        "hmm_bear_chg5": pd.Series(bear).diff(5).to_numpy(),
    })

if __name__ == "__main__":
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(root, "data", "ohlc_spx.csv"), parse_dates=["date"]).sort_values("date")
    f = hmm_features(df)
    print(f.dropna().tail(3).T)
    print("n_features:", f.shape[1] - 1)
