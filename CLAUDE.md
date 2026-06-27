# skew — SPX turning-point research & signals

## Purpose
Detect **S&P 500 turning points** — tops (고점, before a drawdown) and bottoms (저점, the
trough) — early enough to act on, and surface them as live signals on a chart dashboard.
The goal is a usable **BUY-the-bottom / (soft) sell-the-top** indicator, validated honestly
out-of-sample. (It started as a study of 25Δ vol skew vs crashes and grew into a full
turning-point detection system.)

## What works (honest, OOS-validated)
- **BOTTOM signal is the product.** TabPFN AUC ≈ **0.94**; discrete causal signals run
  **~86% precision / ~55% recall (±5 trading days)** ≈ 4× random. Reliable enough to trade.
- **TOP signal is weak** and intrinsically hard — AUC ≈ **0.72**, discrete signals ~63%
  precision / 30% recall (±5d). Treat as a faint risk gauge, **not a trigger**. Confirmed:
  more data (26→185 events) and many model families did NOT break this ceiling — tops look
  like any normal high; bottoms are capitulation EVENTS with a loud signature.
- **`ret_2` (2-day return) is the #1 feature for both targets.** Bottoms also lean on
  drawdown depth, VIX, realized-vol/range-vol, MACD turn, dealer-gamma (GEX) flips.

## Architecture
- **Rust (`src/`)** = data collection + dashboard. `skew update` (CI/daily) fetches SPX OHLC
  (Yahoo ^GSPC), CBOE option snapshots (`data/snap/`), dealer gamma. `skew serve SPY` runs a
  std-only HTTP dashboard on `127.0.0.1:8080` (TradingView candles + signal markers, vol
  surface, dealer gamma, etc.). Serve reads **cached data only — no network at startup.**
- **Python ML (`ml/`)**, run via `uv`. Feature engineering + models. Key files:
  `build_features.py` (→ `features_long.csv`), `perm_importance.py` (feature selection),
  `gen_signals_tabpfn.py` (probabilities → `ml/oos_probs.csv`), `reflag.py` (causal signals
  → `data/signals.csv`), `bakeoff.py`/`tabpfn_run.py` (model comparison).

## Data (all free, no API key)
SPX OHLC 1990–2026 (Yahoo ^GSPC, `data/ohlc_spx_full.csv`); FRED (VIX/VIX3M/VXN, yield curve,
NFCI/ANFCI/STLFSI, dollar, oil, credit OAS 2023+); CBOE index history (VVIX/SKEW/VIX9D/VXTLT/
COR1M/3M) + daily SPX option snapshots; SqueezeMetrics DIX/GEX. Ground-truth turns =
**ZigZag-5% swing labels** (`docs/ground_truth_bear_markets.md` has the original 26 manual ≥5%
windows; the model uses the auto-labeled ~185).

## Models
**CatBoost** (GBDT, CPU, CI-friendly) and **TabPFN v2** (`tabpfn==2.2.1`, GPU, best — top 0.765
/ bottom 0.945). Bayesian-HS, GP, TabICL, transformers, astrology, advanced-physics/CFTC/
creative features were all tested and dropped (no OOS gain). Feature selection is by **OOS
permutation importance (`perm_importance.py`), NOT SHAP** — SHAP overstated in-sample
importance; pruning the 120 feats to the ~72 (top)/~18 (bottom) OOS-useful ones *improved* tops.

## Running
```bash
cargo build --release && ./target/release/skew serve SPY      # dashboard @ :8080/candles
# regenerate signals (needs the local GPU):
uv run --python 3.11 --with "tabpfn==2.2.1" --with torch ... python ml/gen_signals_tabpfn.py
Q=0.95 uv run --with pandas --with numpy python ml/reflag.py   # causal signals.csv (top 5%)
```

## Conventions / gotchas
- **Signals must be CAUSAL.** The TabPFN probability is point-in-time, but flagging must not
  peek the future. Use `reflag.py`'s **rising-edge** (fire the first day prob crosses above
  the threshold) — NOT a ±N local-max window (that needs future days, hindsight-only).
- **TabPFN needs a GPU** (RTX 2080 here); CI has none. So: CI auto-updates price/snapshots/
  panels via CatBoost-free steps; **regenerate TabPFN signals locally and commit** them.
  Always `nvidia-smi` and kill leftover TabPFN procs (`pkill -9 -f`) before a run — a zombie
  can silently hog the GPU.
- **curl in this env**: use `--http1.1` and NO custom `-A` user-agent for FRED/CBOE/Dolt
  (a UA triggers HTTP/2 errors); Yahoo is the exception (needs a browser UA) and 429-throttles
  bulk requests. Yahoo is isolated to the OHLC fetch only (no free full-OHLC alternative).
- Dashboard panels (25Δ skew%/IV%, **3D vol surface**) are computed from the **daily CBOE SPX
  snapshots** (`data/snap/SPX/`). The old Dolt SPY source was frozen and is now fully removed
  from the dashboard — `build_surface` reads snapshots only via `snap_smile()` (no fallback),
  surface dte capped at 180d (drops multi-year LEAPS). Stale/non-auto pages were deleted;
  nav = Candles, Credit, VIX, Vol Surface, Dealer Gamma, Net GEX, GEX+.
- Detailed results live in `docs/leading_indicator_results.md`, `docs/model_selection.md`,
  `docs/feature_checklist.md`.

## Judgment rule
Decide feature/model value by **out-of-sample AUC + discrete signal precision/recall**, never
by SHAP or in-sample fit. Bottoms are the trustworthy side; be skeptical of any "great" top signal.
