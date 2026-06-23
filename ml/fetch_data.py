#!/usr/bin/env python3
"""Fetch raw daily series for the leading-indicator model.

Sources (all free, no key):
  - Yahoo Finance v8 chart API  -> indices/ETFs/vol indices
  - FRED fredgraph.csv          -> credit spreads, yield curve, financial conditions
Writes each series to ml/raw/<name>.csv with columns: date,<name>
Prints a coverage table (first date, last date, rows) so we can see what resolved.
"""
import json, os, sys, time, subprocess, urllib.parse
from datetime import datetime, timezone

RAW = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW, exist_ok=True)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
NOW = int(time.time())
P1 = int(datetime(2010, 1, 1, tzinfo=timezone.utc).timestamp())

def _get(url):
    """curl with retry+backoff. NOTE: this env throws HTTP/2 errors when a custom
    -A user-agent is sent, so use --http1.1 and curl's default UA (matches the
    project's working fetch.rs)."""
    last = None
    for attempt in range(4):
        r = subprocess.run(["curl", "-s", "--http1.1", "-m", "30", url],
                           capture_output=True)
        body = r.stdout
        if r.returncode == 0 and body and b"Too Many Requests" not in body[:200]:
            return body
        last = (body[:120] or r.stderr[:120])
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"curl failed (rc={r.returncode}): {last!r}")

def _get_ua(url):
    """Yahoo needs a browser UA (and tolerates HTTP/2); FRED/CBOE do not."""
    last = None
    for attempt in range(4):
        r = subprocess.run(["curl", "-s", "-m", "30", "-A", UA, url], capture_output=True)
        body = r.stdout
        if r.returncode == 0 and body and b"Too Many Requests" not in body[:200]:
            return body
        last = (body[:120] or r.stderr[:120])
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"yahoo curl failed (rc={r.returncode}): {last!r}")

def cboe(fname):
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{fname}.csv"
    raw = _get(url).decode(errors="ignore").strip().splitlines()
    out = []
    for line in raw[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            d = datetime.strptime(parts[0], "%m/%d/%Y").strftime("%Y-%m-%d")
            v = float(parts[-1])   # close = last column (handles single-val and OHLC)
        except Exception:
            continue
        out.append((d, v))
    return out

def yahoo(symbol):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
           f"?period1={P1}&period2={NOW}&interval=1d")
    data = json.loads(_get_ua(url))
    res = data["chart"]["result"][0]
    ts = res["timestamp"]
    closes = res["indicators"]["quote"][0]["close"]
    out = []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        d = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
        out.append((d, c))
    # de-dupe by date keeping last
    seen = {}
    for d, c in out:
        seen[d] = c
    return sorted(seen.items())

def fred(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    raw = _get(url).decode().strip().splitlines()
    out = []
    for line in raw[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0], parts[1]
        if v in (".", "", "NA"):
            continue
        try:
            out.append((d, float(v)))
        except ValueError:
            pass
    return out

def save(name, rows):
    path = os.path.join(RAW, f"{name}.csv")
    with open(path, "w") as f:
        f.write(f"date,{name}\n")
        for d, v in rows:
            f.write(f"{d},{v}\n")
    return path

# CBOE official index history (rate-limit-free) -> the tail/vol features that
# FRED lacks (VVIX, SKEW, VIX9D, bond-vol proxy VXTLT).
CBOE = {
    "vvix":  "VVIX_History",     # vol-of-vol
    "skew":  "SKEW_History",     # CBOE SKEW (tail risk)
    "vix9d": "VIX9D_History",    # 9-day VIX (short-end term structure)
    "vxtlt": "VXTLT_History",    # TLT implied vol = free MOVE-like bond-vol proxy
    "cor1m": "COR1M_History",    # 1M implied correlation (systemic/dispersion risk)
    "cor3m": "COR3M_History",    # 3M implied correlation
}
# Yahoo ETFs were dropped from the daily pipeline: they re-trip the 429 throttle and
# the only features that depended on them (breadth RSP/SPY) never materialized. Every
# series below (CBOE + FRED + DIX) is reliably curl-fetchable every day with no wall.
YAHOO = {}
FRED = {
    # vol indices (daily, as-of close)
    "vix":     "VIXCLS",
    "vix3m":   "VXVCLS",
    "vxn":     "VXNCLS",
    # credit (ICE -> truncated to ~2023-06; kept, NaN before that)
    "hy_oas":  "BAMLH0A0HYM2",
    "ig_oas":  "BAMLC0A0CM",
    "ccc_oas": "BAMLH0A3HYC",
    # yield curve (daily)
    "t10y2y":  "T10Y2Y",
    "t10y3m":  "T10Y3M",
    "dgs10":   "DGS10",
    "dgs2":    "DGS2",
    # financial conditions / stress (weekly -> publish lag handled downstream)
    "nfci":    "NFCI",
    "anfci":   "ANFCI",
    "stlfsi":  "STLFSI4",
    # macro
    "dollar_broad": "DTWEXBGS",
    "wti":     "DCOILWTICO",
}

def cover(name, rows):
    if not rows:
        return f"{name:14} FAILED (0 rows)"
    return f"{name:14} {rows[0][0]} -> {rows[-1][0]}  ({len(rows)} rows)"

print("=== CBOE ===", flush=True)
for name, fname in CBOE.items():
    try:
        rows = cboe(fname)
        save(name, rows)
        print(cover(name, rows), flush=True)
    except Exception as e:
        print(f"{name:14} ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)

print("=== FRED ===", flush=True)
for name, sid in FRED.items():
    try:
        rows = fred(sid)
        save(name, rows)
        print(cover(name, rows), flush=True)
    except Exception as e:
        print(f"{name:14} ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)

print("=== DIX/GEX (SqueezeMetrics) ===", flush=True)
try:
    body = _get("https://squeezemetrics.com/monitor/static/DIX.csv")
    txt = body.decode(errors="ignore")
    if txt.lower().startswith("date") and "," in txt:
        path = os.path.join(os.path.dirname(RAW), "..", "DIX.csv")
        with open(os.path.normpath(path), "w") as f:
            f.write(txt.strip() + "\n")
        last = txt.strip().splitlines()[-1].split(",")[0]
        print(f"DIX.csv updated -> last {last}", flush=True)
    else:
        print(f"DIX skipped (unexpected body: {txt[:60]!r})", flush=True)
except Exception as e:
    print(f"DIX ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)

print("DONE")
