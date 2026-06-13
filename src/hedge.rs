//! Put-hedge simulation.
//!
//! Entry (all three must hold): skew %ile >= 90  AND  ts_ratio > 1.05  AND
//! "spot up + vol up" (SPX higher and ATM IV higher than 5 trading days ago).
//! Instrument: 30D ATM put. Exit a tranche when term structure returns to
//! contango (ts_ratio < 1.0) OR spot is >3% below the tranche's entry, else at
//! expiry (21 td). Capital: $10,000, deploy $5,000 per trigger, max 2 tranches.

use std::collections::BTreeMap;
use std::path::Path;

use crate::{bs, fetch, price};

const T_ENTRY: f64 = 21.0 / 252.0; // 30 calendar ~ 21 trading days
const HOLD_TD: usize = 21;
const PCT_WIN: usize = 252;
const SKEW_PCT_MIN: f64 = 90.0;
const TS_MIN: f64 = 1.05;
const SUVU_LB: usize = 5; // spot-up-vol-up lookback (trading days)
const STOP_DROP: f64 = 0.03; // take profit if spot >3% below entry
const TRAIL: f64 = 0.30; // trailing stop: exit if put value gives back 30% of its peak (once in profit)
const TRANCHE: f64 = 5000.0;
const BASE: f64 = 10000.0;

struct Pos {
    entry_i: usize,
    entry_date: String,
    k: f64,
    entry_spot: f64,
    p0: f64,
    cost: f64,
    peak: f64, // running peak position value (for trailing stop)
}

fn rolling_pct(vals: &[f64], win: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; vals.len()];
    for i in 0..vals.len() {
        if !vals[i].is_finite() {
            continue;
        }
        let lo = i.saturating_sub(win - 1);
        let (mut c, mut t) = (0.0, 0.0);
        for j in lo..=i {
            if vals[j].is_finite() {
                t += 1.0;
                if vals[j] <= vals[i] {
                    c += 1.0;
                }
            }
        }
        if t > 1.0 {
            out[i] = c / t * 100.0;
        }
    }
    out
}

pub fn run(symbol: &str) {
    // ---- skew series: date -> (iv_atm, ts_ratio, skew_pct) ----
    let text = match std::fs::read_to_string(format!("out/skew_{symbol}.csv")) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("run `skew backfill {symbol}` first");
            return;
        }
    };
    let mut sdates = Vec::new();
    let mut iv = Vec::new();
    let mut ts = Vec::new();
    let mut snorm = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 10 {
            continue;
        }
        let p = |i: usize| f[i].parse::<f64>().unwrap_or(f64::NAN);
        sdates.push(f[0].to_string());
        iv.push(p(3));
        snorm.push(p(5));
        ts.push(p(9));
    }
    let skpct = rolling_pct(&snorm, PCT_WIN);
    // date -> (iv, ts, skpct)
    let smap: BTreeMap<&str, (f64, f64, f64)> = sdates
        .iter()
        .enumerate()
        .map(|(i, d)| (d.as_str(), (iv[i], ts[i], skpct[i])))
        .collect();

    // ---- daily SPX (FRED) ----
    let cache = fetch::default_cache_root();
    let spx = match price::load_spx(Path::new(&cache)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("price error: {e}");
            return;
        }
    };
    let n = spx.len();
    let close: Vec<f64> = spx.iter().map(|(_, c)| *c).collect();
    let dates: Vec<&str> = spx.iter().map(|(d, _)| d.as_str()).collect();

    // forward-fill skew metrics onto the daily grid
    let (mut ivff, mut tsff, mut skff) = (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    let mut last = (f64::NAN, f64::NAN, f64::NAN);
    // walk skew dates in order, assign to each daily date the latest skew obs <= it
    let mut sd_sorted: Vec<(&str, (f64, f64, f64))> =
        smap.iter().map(|(d, v)| (*d, *v)).collect();
    sd_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut j = 0;
    for i in 0..n {
        while j < sd_sorted.len() && sd_sorted[j].0 <= dates[i] {
            last = sd_sorted[j].1;
            j += 1;
        }
        ivff[i] = last.0;
        tsff[i] = last.1;
        skff[i] = last.2;
    }

    // ---- simulate ----
    let mut cash = BASE;
    let mut open: Vec<Pos> = Vec::new();
    let mut equity_curve: Vec<(String, f64)> = Vec::with_capacity(n);
    let mut trades: Vec<String> = Vec::new();
    let mut n_entries = 0usize;
    let put = |s: f64, k: f64, t: f64, v: f64| bs::bs_price(false, s, k, t, v);

    for i in 0..n {
        // exits
        let mut still_open: Vec<Pos> = Vec::new();
        for mut pos in open.drain(..) {
            let held = i - pos.entry_i;
            let rem_t = ((HOLD_TD.saturating_sub(held)) as f64) / 252.0;
            let value = pos.cost * put(close[i], pos.k, rem_t, ivff[i]) / pos.p0;
            if value > pos.peak {
                pos.peak = value;
            }
            if held == 0 {
                still_open.push(pos);
                continue;
            }
            // Exits: (a) -3% take-profit, (c) trailing stop (only once in profit), expiry.
            let reason = if close[i] <= pos.entry_spot * (1.0 - STOP_DROP) {
                Some("stop-3%")
            } else if pos.peak > pos.cost && value <= pos.peak * (1.0 - TRAIL) {
                Some("trail")
            } else if held >= HOLD_TD {
                Some("expiry")
            } else {
                None
            };
            if let Some(reason) = reason {
                cash += value;
                trades.push(format!(
                    "{} -> {}  {:<8}  spot {:.0}->{:.0} ({:+.1}%)  P&L {:+.0} (cost {:.0}, val {:.0})",
                    pos.entry_date, dates[i], reason, pos.entry_spot, close[i],
                    (close[i] / pos.entry_spot - 1.0) * 100.0, value - pos.cost, pos.cost, value
                ));
            } else {
                still_open.push(pos);
            }
        }
        open = still_open;

        // entry
        let trigger = skff[i].is_finite()
            && skff[i] >= SKEW_PCT_MIN
            && tsff[i].is_finite()
            && tsff[i] > TS_MIN
            && i >= SUVU_LB
            && ivff[i].is_finite()
            && ivff[i - SUVU_LB].is_finite()
            && close[i] > close[i - SUVU_LB]
            && ivff[i] > ivff[i - SUVU_LB];
        if trigger && open.len() < 2 && cash >= 1.0 {
            let p0 = put(close[i], close[i], T_ENTRY, ivff[i]);
            if p0 > 1e-9 {
                let cost = TRANCHE.min(cash);
                cash -= cost;
                n_entries += 1;
                open.push(Pos {
                    entry_i: i,
                    entry_date: dates[i].to_string(),
                    k: close[i],
                    entry_spot: close[i],
                    p0,
                    cost,
                    peak: cost,
                });
            }
        }

        // mark-to-market equity
        let mut mtm = cash;
        for pos in &open {
            let held = i - pos.entry_i;
            let rem_t = ((HOLD_TD.saturating_sub(held)) as f64) / 252.0;
            let pnow = put(close[i], pos.k, rem_t, ivff[i]);
            mtm += pos.cost * pnow / pos.p0;
        }
        equity_curve.push((dates[i].to_string(), mtm));
    }
    // force-close any still open at last mark (already in equity); realize into cash for final
    let final_eq = equity_curve.last().map(|x| x.1).unwrap_or(BASE);

    // ---- output ----
    std::fs::create_dir_all("out").ok();
    let mut eq_csv = String::from("date,equity\n");
    for (d, e) in &equity_curve {
        eq_csv.push_str(&format!("{d},{:.2}\n", e));
    }
    std::fs::write("out/hedge_equity.csv", eq_csv).ok();
    std::fs::write("out/hedge_trades.txt", trades.join("\n")).ok();

    println!("=== {symbol} put-hedge sim  (entry: skew%ile>=90 & ts>1.05 & spot-up-vol-up; 30D ATM put) ===");
    println!("(exit: -3% take-profit | trailing stop {:.0}% off peak once in profit | expiry; no contango exit)", TRAIL * 100.0);
    println!("entries: {n_entries}  | closed trades: {}", trades.len());
    let pnls: Vec<f64> = trades.iter().filter_map(|t| {
        t.split("P&L ").nth(1)?.split_whitespace().next()?.parse::<f64>().ok()
    }).collect();
    let total_pnl: f64 = pnls.iter().sum();
    let wins = pnls.iter().filter(|p| **p > 0.0).count();
    println!("realized P&L sum: {:+.0}  | winners {}/{}", total_pnl, wins, pnls.len());
    println!("start equity: ${:.0}  ->  final equity: ${:.0}  ({:+.1}%)", BASE, final_eq, (final_eq / BASE - 1.0) * 100.0);

    // equity at key milestones
    let at = |target: &str| {
        equity_curve
            .iter()
            .filter(|(d, _)| d.as_str() <= target)
            .last()
            .map(|(d, e)| format!("{d}: ${:.0}", e))
            .unwrap_or_default()
    };
    println!("\nequity path around the Feb-2026 crash:");
    for t in ["2026-01-15", "2026-02-04", "2026-02-26", "2026-03-09", "2026-03-30", "2026-04-15"] {
        println!("  {}", at(t));
    }
    println!("\nfirst / last trades:");
    for t in trades.iter().take(4) {
        println!("  {t}");
    }
    if trades.len() > 4 {
        println!("  ...");
        for t in trades.iter().rev().take(4).rev() {
            println!("  {t}");
        }
    }
    println!("\nwrote out/hedge_equity.csv, out/hedge_trades.txt");
}
