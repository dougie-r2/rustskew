//! S&P 500 index daily levels from FRED (series `SP500`, no API key required),
//! plus derived metrics: running drawdown and forward returns. Cached to disk.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::dates;
use crate::fetch::{curl_get, curl_get_ua};

#[derive(Clone, Debug)]
pub struct PriceRow {
    pub date: String,
    pub close: f64,
    pub drawdown: f64,    // close / running_peak - 1  (<= 0)
    pub rv_21: f64,       // trailing 21-day annualized realized vol (close-to-close)
    pub fwd_5: f64,       // 5-trading-day forward return
    pub fwd_21: f64,      // ~1-month forward return
    pub fwd_min_21: f64,  // worst close-to-close drop over next 21 days (<= 0)
}

const FRED_URL: &str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500";

/// Fetch the SP500 series (cached at data/price_spx.csv) and parse date->close.
pub fn load_spx(cache_root: &Path) -> Result<Vec<(String, f64)>, String> {
    let cache = cache_root.join("price_spx.csv");
    let bytes = if let Ok(b) = std::fs::read(&cache) {
        b
    } else {
        let b = curl_get(FRED_URL)?;
        std::fs::create_dir_all(cache_root).ok();
        std::fs::write(&cache, &b).ok();
        b
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    for line in text.lines().skip(1) {
        let mut it = line.split(',');
        let (Some(d), Some(v)) = (it.next(), it.next()) else { continue };
        let d = d.trim();
        let v = v.trim();
        if d.is_empty() || v.is_empty() || v == "." {
            continue;
        }
        if let Ok(c) = v.parse::<f64>() {
            out.insert(d.to_string(), c);
        }
    }
    if out.is_empty() {
        return Err("no SP500 rows parsed from FRED".into());
    }
    Ok(out.into_iter().collect())
}

const YAHOO_OHLC: &str =
    "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?range=10y&interval=1d";

type Bars = Vec<(String, f64, f64, f64, f64)>;

fn fetch_ohlc_yahoo() -> Result<Bars, String> {
    let bytes = curl_get_ua(YAHOO_OHLC, "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("yahoo json: {e}"))?;
    let r = &v["chart"]["result"][0];
    let ts = r["timestamp"].as_array().ok_or("yahoo: no timestamp")?;
    let q = &r["indicators"]["quote"][0];
    let arr = |k: &str| q[k].as_array().cloned().unwrap_or_default();
    let (op, hi, lo, cl) = (arr("open"), arr("high"), arr("low"), arr("close"));
    let mut out = Vec::with_capacity(ts.len());
    for i in 0..ts.len() {
        let t = ts[i].as_i64().unwrap_or(0);
        let f = |a: &[Value]| a.get(i).and_then(|x| x.as_f64());
        if let (Some(o), Some(h), Some(l), Some(c)) = (f(&op), f(&hi), f(&lo), f(&cl)) {
            out.push((dates::fmt_ymd(t / 86400), o, h, l, c));
        }
    }
    if out.is_empty() {
        return Err("yahoo: no OHLC bars parsed".into());
    }
    Ok(out)
}

fn write_ohlc_cache(cache_root: &Path, bars: &Bars) {
    let mut csv = String::from("date,o,h,l,c\n");
    for (d, o, h, l, c) in bars {
        csv.push_str(&format!("{d},{:.2},{:.2},{:.2},{:.2}\n", o, h, l, c));
    }
    std::fs::create_dir_all(cache_root).ok();
    std::fs::write(cache_root.join("ohlc_spx.csv"), csv).ok();
}

fn read_ohlc_cache(cache_root: &Path) -> Bars {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string(cache_root.join("ohlc_spx.csv")) {
        for line in t.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 5 {
                continue;
            }
            if let (Ok(o), Ok(h), Ok(l), Ok(c)) =
                (f[1].parse(), f[2].parse(), f[3].parse(), f[4].parse())
            {
                out.push((f[0].to_string(), o, h, l, c));
            }
        }
    }
    out
}

/// Daily OHLC for the S&P 500 (^GSPC), cached at data/ohlc_spx.csv. Uses cache if present.
pub fn load_ohlc(cache_root: &Path) -> Result<Bars, String> {
    let cached = read_ohlc_cache(cache_root);
    if !cached.is_empty() {
        return Ok(cached);
    }
    let bars = fetch_ohlc_yahoo()?;
    write_ohlc_cache(cache_root, &bars);
    Ok(bars)
}

/// Force-refresh OHLC from Yahoo (fills any gap up to the latest session).
/// Returns (bar count, last date).
pub fn update_ohlc(cache_root: &Path) -> Result<(usize, String), String> {
    let bars = fetch_ohlc_yahoo()?;
    write_ohlc_cache(cache_root, &bars);
    let last = bars.last().map(|b| b.0.clone()).unwrap_or_default();
    Ok((bars.len(), last))
}

/// Compute drawdown + forward-return metrics over the (date-sorted) series.
pub fn enrich(series: &[(String, f64)]) -> Vec<PriceRow> {
    let n = series.len();
    // daily log returns (logret[i] is the return into day i; logret[0] = NaN)
    let mut logret = vec![f64::NAN; n];
    for i in 1..n {
        logret[i] = (series[i].1 / series[i - 1].1).ln();
    }
    const H: usize = 21;
    let ann = (252.0_f64).sqrt();

    let mut peak = f64::MIN;
    let mut rows = Vec::with_capacity(n);
    for (i, (date, close)) in series.iter().enumerate() {
        if *close > peak {
            peak = *close;
        }
        let drawdown = close / peak - 1.0;

        // trailing 21-day realized vol (sample std of last H log returns)
        let rv_21 = if i >= H {
            let win = &logret[(i - H + 1)..=i];
            let m = win.iter().sum::<f64>() / win.len() as f64;
            let var = win.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (win.len() as f64 - 1.0);
            var.sqrt() * ann
        } else {
            f64::NAN
        };

        let fwd = |k: usize| -> f64 {
            if i + k < n {
                series[i + k].1 / close - 1.0
            } else {
                f64::NAN
            }
        };
        // worst forward close over next 21 days
        let mut worst = 0.0_f64;
        for j in (i + 1)..(i + 22).min(n) {
            let r = series[j].1 / close - 1.0;
            if r < worst {
                worst = r;
            }
        }
        rows.push(PriceRow {
            date: date.clone(),
            close: *close,
            drawdown,
            rv_21,
            fwd_5: fwd(5),
            fwd_21: fwd(21),
            fwd_min_21: worst,
        });
    }
    rows
}
