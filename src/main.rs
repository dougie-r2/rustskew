mod bs;
mod bt;
mod cboe;
mod chart;
mod hedge;
mod dates;
mod fetch;
mod fred;
mod price;
mod serve;
mod skew;
mod svi;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "day" => cmd_day(&args),
        "backfill" => cmd_backfill(&args),
        "report" => cmd_report(&args),
        "backtest" => bt::run(args.get(2).map(|s| s.as_str()).unwrap_or("SPY")),
        "hedge" => hedge::run(args.get(2).map(|s| s.as_str()).unwrap_or("SPY")),
        "update" => run_update(),
        "serve" => serve::run(args.get(2).map(|s| s.as_str()).unwrap_or("SPY")),
        _ => {
            eprintln!("usage:");
            eprintln!("  skew day <SYMBOL> <YYYY-MM-DD>            one day's 25d 30DTE skew");
            eprintln!("  skew backfill <SYMBOL> [from] [to]       daily skew series -> out/skew_<SYMBOL>.csv");
            eprintln!("  skew report <SYMBOL>                     join with SPX, stats + HTML chart");
            eprintln!("  skew backtest <SYMBOL>                   long-option directional backtest");
            eprintln!("  skew update                              refresh price + today's CBOE snapshot + dealer gamma");
        }
    }
}

/// Incremental data update: fill the index-price gap (Yahoo) and take today's
/// CBOE option snapshot (SPX, VIX) + dealer gamma. Run on `update` and at serve start.
fn run_update() {
    println!("== data update ==");
    let cache = fetch::default_cache_root();
    match price::update_ohlc(Path::new(&cache)) {
        Ok((n, last)) => println!("  price: {n} bars, last {last}"),
        Err(e) => eprintln!("  price: {e}"),
    }
    cboe::update_all();
    fred::update_all();
}

#[derive(Clone)]
struct SkewRow {
    iv_25p: f64,
    iv_25c: f64,
    iv_atm: f64,
    skew_abs: f64,
    skew_norm: f64,
}

fn read_skew_csv(path: &str) -> BTreeMap<String, SkewRow> {
    let mut m = BTreeMap::new();
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return m,
    };
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 6 {
            continue;
        }
        let p = |i: usize| f[i].parse::<f64>().unwrap_or(f64::NAN);
        m.insert(
            f[0].to_string(),
            SkewRow {
                iv_25p: p(1),
                iv_25c: p(2),
                iv_atm: p(3),
                skew_abs: p(4),
                skew_norm: p(5),
            },
        );
    }
    m
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();
    let n = pairs.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (x, y) in &pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn mean(v: &[f64]) -> f64 {
    let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        f64::NAN
    } else {
        f.iter().sum::<f64>() / f.len() as f64
    }
}

/// Join skew + price, write combined CSV, print crash-precursor stats, write HTML chart.
fn cmd_report(args: &[String]) {
    let symbol = args.get(2).cloned().unwrap_or_else(|| "SPY".into());
    let skew_map = read_skew_csv(&format!("out/skew_{symbol}.csv"));
    if skew_map.is_empty() {
        eprintln!("no skew data at out/skew_{symbol}.csv — run `skew backfill {symbol}` first");
        return;
    }
    let cache = fetch::default_cache_root();
    let spx = match price::load_spx(&cache) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("price load error: {e}");
            return;
        }
    };
    let price_rows = price::enrich(&spx);
    let price_map: BTreeMap<&str, &price::PriceRow> =
        price_rows.iter().map(|r| (r.date.as_str(), r)).collect();

    // Join on common dates.
    let mut joined: Vec<chart::Joined> = Vec::new();
    let mut fwd5 = Vec::new();
    let mut fwd21 = Vec::new();
    let mut fwdmin = Vec::new();
    let mut sn = Vec::new();
    let mut sa = Vec::new();
    let mut atm = Vec::new();
    for (date, s) in &skew_map {
        if let Some(pr) = price_map.get(date.as_str()) {
            joined.push(chart::Joined {
                date: date.clone(),
                close: pr.close,
                drawdown: pr.drawdown,
                skew_abs: s.skew_abs,
                skew_norm: s.skew_norm,
                atm_iv: s.iv_atm,
                vrp: s.iv_atm - pr.rv_21,
            });
            fwd5.push(pr.fwd_5);
            fwd21.push(pr.fwd_21);
            fwdmin.push(pr.fwd_min_21);
            sn.push(s.skew_norm);
            sa.push(s.skew_abs);
            atm.push(s.iv_atm);
        }
    }
    let n = joined.len();
    if n == 0 {
        eprintln!("no overlapping dates between skew and price");
        return;
    }

    // Write joined CSV.
    std::fs::create_dir_all("out").ok();
    let mut csv = String::from("date,close,drawdown,skew_abs,skew_norm,atm_iv,fwd_5,fwd_21,fwd_min_21\n");
    for (i, j) in joined.iter().enumerate() {
        csv.push_str(&format!(
            "{},{:.2},{:.4},{:.6},{:.6},{:.6},{:.4},{:.4},{:.4}\n",
            j.date, j.close, j.drawdown, j.skew_abs, j.skew_norm, j.atm_iv, fwd5[i], fwd21[i], fwdmin[i]
        ));
    }
    let jpath = format!("out/joined_{symbol}.csv");
    std::fs::write(&jpath, csv).ok();

    // ---- Stats ----
    println!("=== {symbol} 25Δ-30DTE skew vs SPX  ({} days, {}..{}) ===",
        n, joined.first().unwrap().date, joined.last().unwrap().date);
    println!("skew_norm: mean {:.3}  | skew_abs: mean {:.2}pt | ATM IV: mean {:.1}%",
        mean(&sn), mean(&sa) * 100.0, mean(&atm) * 100.0);
    println!();
    println!("Forward-return predictiveness (Pearson corr):");
    println!("  skew_norm vs fwd_21 : {:+.3}", pearson(&sn, &fwd21));
    println!("  skew_abs  vs fwd_21 : {:+.3}", pearson(&sa, &fwd21));
    println!("  ATM IV    vs fwd_21 : {:+.3}", pearson(&atm, &fwd21));
    println!("  skew_norm vs fwd_min_21 (worst drop): {:+.3}", pearson(&sn, &fwdmin));
    println!();

    // Quintiles of skew_norm -> forward outcomes.
    print_quintiles("skew_norm", &sn, &fwd5, &fwd21, &fwdmin);
    print_quintiles("ATM_IV", &atm, &fwd5, &fwd21, &fwdmin);

    // "Right before a crash": days where a >=10% drop occurs within next 21d.
    let mut pre_sn = Vec::new();
    let mut pre_atm = Vec::new();
    let mut norm_sn = Vec::new();
    let mut norm_atm = Vec::new();
    for i in 0..n {
        if fwdmin[i] <= -0.10 {
            pre_sn.push(sn[i]);
            pre_atm.push(atm[i]);
        } else if fwdmin[i].is_finite() {
            norm_sn.push(sn[i]);
            norm_atm.push(atm[i]);
        }
    }
    println!("Pre-drop state (a >=10% drop within next 21 trading days):");
    println!("  pre-drop days: {}  | other days: {}", pre_sn.len(), norm_sn.len());
    println!("  skew_norm  pre-drop {:.3}  vs other {:.3}", mean(&pre_sn), mean(&norm_sn));
    println!("  ATM IV     pre-drop {:.1}% vs other {:.1}%", mean(&pre_atm) * 100.0, mean(&norm_atm) * 100.0);

    // Interactive dashboard (primary) + simple static chart (fallback).
    let title = format!("{symbol} — 25Δ 30DTE skew / IV / VRP vs S&P 500 (2019–2026)");
    let dpath = format!("out/dashboard_{symbol}.html");
    let hpath = format!("out/skew_{symbol}.html");
    chart::write_dashboard(&dpath, &title, &joined).ok();
    chart::write_html(&hpath, &title, &joined).ok();
    println!("\nwrote {jpath}");
    println!("wrote {dpath}   <- interactive (zoom/pan/hover) — open this");
    println!("wrote {hpath}   (static fallback)");
}

fn print_quintiles(name: &str, key: &[f64], fwd5: &[f64], fwd21: &[f64], fwdmin: &[f64]) {
    // rank days by key, split into 5 buckets, average forward outcomes.
    let mut idx: Vec<usize> = (0..key.len()).filter(|&i| key[i].is_finite()).collect();
    idx.sort_by(|&a, &b| key[a].partial_cmp(&key[b]).unwrap());
    let m = idx.len();
    if m < 5 {
        return;
    }
    println!("{name} quintile -> avg forward outcome (Q1 low .. Q5 high):");
    println!("  {:>3} {:>10} {:>10} {:>10} {:>12}", "Q", "key", "fwd_5", "fwd_21", "fwd_min_21");
    for q in 0..5 {
        let lo = q * m / 5;
        let hi = (q + 1) * m / 5;
        let bucket = &idx[lo..hi];
        let k: Vec<f64> = bucket.iter().map(|&i| key[i]).collect();
        let f5: Vec<f64> = bucket.iter().map(|&i| fwd5[i]).collect();
        let f21: Vec<f64> = bucket.iter().map(|&i| fwd21[i]).collect();
        let fm: Vec<f64> = bucket.iter().map(|&i| fwdmin[i]).collect();
        println!(
            "  Q{}  {:>9.3} {:>9.2}% {:>9.2}% {:>11.2}%",
            q + 1, mean(&k), mean(&f5) * 100.0, mean(&f21) * 100.0, mean(&fm) * 100.0
        );
    }
    println!();
}

/// Backfill the full daily 25d-30DTE skew series and write a CSV.
fn cmd_backfill(args: &[String]) {
    let symbol = args.get(2).cloned().unwrap_or_else(|| "SPY".into());
    let from = args.get(3).cloned().unwrap_or_else(|| "2019-02-09".into());
    let to = args.get(4).cloned().unwrap_or_else(|| "2026-06-12".into());

    let (d0, d1) = match (dates::parse_ymd(&from), dates::parse_ymd(&to)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("bad date range");
            return;
        }
    };

    // All weekdays in range (holidays/gaps just return empty chains).
    let mut all_dates: Vec<String> = Vec::new();
    let mut d = d0;
    while d <= d1 {
        if !dates::is_weekend(d) {
            all_dates.push(dates::fmt_ymd(d));
        }
        d += 1;
    }
    let total = all_dates.len();
    eprintln!("backfill {symbol}: {total} weekdays {from}..{to}");

    let cache = fetch::default_cache_root();
    let cache_root = Arc::new(cache);
    let dates_arc = Arc::new(all_dates);
    let next = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<skew::DaySkew>>> = Arc::new(Mutex::new(Vec::with_capacity(total)));

    let n_workers = 8;
    let mut handles = Vec::new();
    for _ in 0..n_workers {
        let dates_arc = Arc::clone(&dates_arc);
        let next = Arc::clone(&next);
        let done = Arc::clone(&done);
        let results = Arc::clone(&results);
        let cache_root = Arc::clone(&cache_root);
        let symbol = symbol.clone();
        handles.push(std::thread::spawn(move || {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= dates_arc.len() {
                    break;
                }
                let date = &dates_arc[i];
                if let Ok(rows) = fetch::dolt_day_rows(&symbol, date, Path::new(cache_root.as_path())) {
                    if !rows.is_empty() {
                        if let Some(s) = skew::compute_day_skew(date, &rows) {
                            results.lock().unwrap().push(s);
                        }
                    }
                }
                let c = done.fetch_add(1, Ordering::Relaxed) + 1;
                if c % 100 == 0 || c == dates_arc.len() {
                    eprintln!("  {c}/{} done", dates_arc.len());
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }

    let mut rows = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    rows.sort_by(|a, b| a.date.cmp(&b.date));

    std::fs::create_dir_all("out").ok();
    let path = format!("out/skew_{symbol}.csv");
    let mut csv = String::from(
        "date,iv_25p,iv_25c,iv_atm,skew_abs,skew_norm,dte_lo,dte_hi,ts_slope,ts_ratio\n",
    );
    for s in &rows {
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.0},{:.0},{:.6},{:.6}\n",
            s.date, s.iv_25p, s.iv_25c, s.iv_atm, s.skew_abs, s.skew_norm, s.dte_lo, s.dte_hi,
            s.ts_slope, s.ts_ratio
        ));
    }
    std::fs::write(&path, csv).expect("write csv");
    eprintln!("wrote {} rows to {path}", rows.len());
}

/// Debug/validation: compute and print one day's skew, with the per-expiry detail.
fn cmd_day(args: &[String]) {
    let symbol = args.get(2).map(|s| s.as_str()).unwrap_or("SPY");
    let date = match args.get(3) {
        Some(d) => d.as_str(),
        None => {
            eprintln!("need a date: skew day {symbol} 2026-05-15");
            return;
        }
    };
    let cache = fetch::default_cache_root();
    let rows = match fetch::dolt_day_rows(symbol, date, Path::new(&cache)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("fetch error: {e}");
            return;
        }
    };
    println!("{symbol} {date}: {} raw rows", rows.len());

    match skew::compute_day_skew(date, &rows) {
        Some(s) => {
            println!("  bracket DTE: {:.0} -> {:.0}  (synthetic {})", s.dte_lo, s.dte_hi, 30);
            println!("  IV 25d put : {:.4}  ({:.2}%)", s.iv_25p, s.iv_25p * 100.0);
            println!("  IV 25d call: {:.4}  ({:.2}%)", s.iv_25c, s.iv_25c * 100.0);
            println!("  IV ATM     : {:.4}  ({:.2}%)", s.iv_atm, s.iv_atm * 100.0);
            println!("  -------------------------------------------");
            println!("  25d skew (abs) : {:.4}  ({:.2} vol pts)", s.skew_abs, s.skew_abs * 100.0);
            println!("  25d skew (norm): {:.4}", s.skew_norm);
        }
        None => println!("  (no skew: chain insufficient that day)"),
    }
}
