//! CBOE delayed-quote option-chain ingestion + dealer gamma (GEX).
//!
//! CBOE serves only TODAY's full chain (with open interest + gamma), so this is
//! a forward-only collector: each run stores one daily snapshot per underlying
//! and appends a dealer-gamma summary. There is no free way to backfill past
//! daily chains, so the archive grows from the day you start running it.

use std::io::Write;
use std::path::Path;

use serde_json::Value;

use crate::bs;
use crate::dates;
use crate::fetch::curl_get;

const CBOE: &str = "https://cdn.cboe.com/api/global/delayed_quotes/options/";

pub struct Opt {
    pub strike: f64,
    pub dte: f64,
    pub is_call: bool,
    pub oi: f64,
    pub gamma: f64,
    pub delta: f64,
    pub iv: f64,
    pub bid: f64,
    pub ask: f64,
}

pub struct Chain {
    pub date: String,
    pub timestamp: String,
    pub spot: f64,
    pub opts: Vec<Opt>,
}

pub struct Gamma {
    pub date: String,
    pub spot: f64,
    pub call_gex: f64, // $ gamma per 1% move (magnitude, call side)
    pub put_gex: f64,
    pub net_gex: f64, // dealer net GEX (sign convention per underlying)
    pub net_vex: f64, // dealer net VEX / vanna ($ per 1 vol-pt)
    pub total_oi: f64,
}

/// Parse an OCC symbol like `SPX260618C06525000` -> (strike, is_call, expiry_days).
fn parse_occ(sym: &str) -> Option<(f64, bool, i64)> {
    let n = sym.len();
    if n < 15 || !sym.is_ascii() {
        return None;
    }
    let strike = sym[n - 8..n].parse::<f64>().ok()? / 1000.0;
    let cp = sym.as_bytes()[n - 9] as char;
    let is_call = cp == 'C' || cp == 'c';
    if !is_call && cp != 'P' && cp != 'p' {
        return None;
    }
    let yy: i64 = sym[n - 15..n - 13].parse().ok()?;
    let mm: i64 = sym[n - 13..n - 11].parse().ok()?;
    let dd: i64 = sym[n - 11..n - 9].parse().ok()?;
    Some((strike, is_call, dates::days_from_civil(2000 + yy, mm, dd)))
}

/// CBOE timestamps with the fetch time, but the data is the last completed
/// session. Roll weekends back to the prior Friday so snapshots carry the real
/// trading date (e.g. a Saturday fetch is Friday's EOD data).
fn session_date(raw: &str) -> String {
    match dates::parse_ymd(raw) {
        Some(mut d) => {
            while dates::is_weekend(d) {
                d -= 1;
            }
            dates::fmt_ymd(d)
        }
        None => raw.to_string(),
    }
}

/// Fetch the current chain for a CBOE symbol (e.g. `_SPX`, `_VIX`).
pub fn fetch_chain(symbol: &str) -> Result<Chain, String> {
    let url = format!("{CBOE}{symbol}.json");
    let bytes = curl_get(&url)?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("cboe json: {e}"))?;
    let ts = v["timestamp"].as_str().unwrap_or("").to_string();
    let date = session_date(ts.get(0..10).unwrap_or(""));
    let d0 = dates::parse_ymd(&date).ok_or("cboe: bad timestamp date")?;
    let data = &v["data"];
    let spot = data["current_price"].as_f64().ok_or("cboe: no spot")?;
    let arr = data["options"].as_array().ok_or("cboe: no options")?;
    let mut opts = Vec::new();
    for o in arr {
        let oi = o["open_interest"].as_f64().unwrap_or(0.0);
        let gamma = o["gamma"].as_f64().unwrap_or(0.0);
        if oi <= 0.0 {
            continue; // only contracts with open interest matter for dealer gamma
        }
        let sym = o["option"].as_str().unwrap_or("");
        let Some((strike, is_call, exp)) = parse_occ(sym) else { continue };
        let dte = (exp - d0) as f64;
        if dte < 0.0 {
            continue;
        }
        opts.push(Opt {
            strike,
            dte,
            is_call,
            oi,
            gamma,
            delta: o["delta"].as_f64().unwrap_or(0.0),
            iv: o["iv"].as_f64().unwrap_or(0.0),
            bid: o["bid"].as_f64().unwrap_or(0.0),
            ask: o["ask"].as_f64().unwrap_or(0.0),
        });
    }
    Ok(Chain { date, timestamp: ts, spot, opts })
}

/// Naive dealer gamma exposure (SqueezeMetrics-style): per option
/// `gamma * OI * 100 * spot^2 * 0.01` = $ per 1% move. Sign convention:
/// equity index -> dealers long calls / short puts; VIX -> dealers short calls.
pub fn dealer_gamma(ch: &Chain, vix: bool) -> Gamma {
    let factor = ch.spot * ch.spot * 0.01; // GEX: $ per 1% spot move
    let vfac = ch.spot * 0.01; // VEX: $ delta per 1 vol-pt move
    let sgn = |is_call: bool| -> f64 {
        if vix {
            if is_call { -1.0 } else { 1.0 }
        } else if is_call {
            1.0
        } else {
            -1.0
        }
    };
    let (mut call, mut put, mut oi, mut net_vex) = (0.0, 0.0, 0.0, 0.0);
    for o in &ch.opts {
        let g = o.gamma * o.oi * 100.0 * factor;
        if o.is_call {
            call += g;
        } else {
            put += g;
        }
        let vanna = if (0.01..=2.0).contains(&o.iv) {
            bs::bs_vanna(ch.spot, o.strike, o.dte / 365.0, o.iv)
        } else {
            0.0
        };
        net_vex += sgn(o.is_call) * vanna * o.oi * 100.0 * vfac;
        oi += o.oi;
    }
    let net = if vix { -call + put } else { call - put };
    Gamma {
        date: ch.date.clone(),
        spot: ch.spot,
        call_gex: call,
        put_gex: put,
        net_gex: net,
        net_vex,
        total_oi: oi,
    }
}

/// Fetch, store a compact snapshot (skip if today's already stored), and append
/// the dealer-gamma summary. Returns a one-line status.
pub fn update_underlying(label: &str, symbol: &str, vix: bool) -> Result<String, String> {
    let ch = fetch_chain(symbol)?;
    let dir = format!("data/snap/{label}");
    let snap = format!("{dir}/{}.csv", ch.date);
    if Path::new(&snap).exists() {
        return Ok(format!("{label} {}: already stored ({} contracts)", ch.date, ch.opts.len()));
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut s = format!(
        "# spot,{:.4},ts,{}\nstrike,dte,cp,oi,gamma,delta,iv,bid,ask\n",
        ch.spot, ch.timestamp
    );
    for o in &ch.opts {
        s.push_str(&format!(
            "{:.2},{:.0},{},{:.0},{:.6},{:.4},{:.4},{:.2},{:.2}\n",
            o.strike, o.dte, if o.is_call { "C" } else { "P" }, o.oi, o.gamma, o.delta, o.iv, o.bid, o.ask
        ));
    }
    std::fs::write(&snap, s).map_err(|e| e.to_string())?;

    let g = dealer_gamma(&ch, vix);
    let gpath = format!("data/gamma_{label}.csv");
    let new = !Path::new(&gpath).exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&gpath)
        .map_err(|e| e.to_string())?;
    if new {
        let _ = f.write_all(b"date,spot,call_gex_mm,put_gex_mm,net_gex_mm,net_vex_mm,gexplus_mm,total_oi\n");
    }
    let _ = f.write_all(
        format!(
            "{},{:.2},{:.1},{:.1},{:.1},{:.1},{:.1},{:.0}\n",
            g.date, g.spot, g.call_gex / 1e6, g.put_gex / 1e6, g.net_gex / 1e6,
            g.net_vex / 1e6, (g.net_gex + g.net_vex) / 1e6, g.total_oi
        )
        .as_bytes(),
    );
    Ok(format!(
        "{label} {}: spot {:.0} · GEX {:+.0} · VEX {:+.0} · GEX+ {:+.0} $mm · {} contracts",
        g.date, g.spot, g.net_gex / 1e6, g.net_vex / 1e6, (g.net_gex + g.net_vex) / 1e6, ch.opts.len()
    ))
}

/// Update all collected underlyings (forward-only snapshots).
pub fn update_all() {
    for (label, sym, vix) in [("SPX", "_SPX", false), ("VIX", "_VIX", true)] {
        match update_underlying(label, sym, vix) {
            Ok(msg) => println!("  {msg}"),
            Err(e) => eprintln!("  {label}: {e}"),
        }
    }
}
