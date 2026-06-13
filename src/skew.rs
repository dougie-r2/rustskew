//! 25-delta synthetic-30DTE volatility skew.
//!
//! Methodology (matches the standard desk approach / the volatility-trading
//! reference repo):
//!   1. Per expiry, pick the OTM put with delta closest to -0.25 and the OTM
//!      call closest to +0.25 (within a delta tolerance), plus an ATM IV
//!      (average of the ~50-delta call and put).
//!   2. Build a *synthetic* 30-DTE point by interpolating the two expiries that
//!      bracket 30 days, in total-variance space (V = IV^2 * t).
//!   3. Skew_abs   = IV(25d put) - IV(25d call)
//!      Skew_norm  = Skew_abs / IV(ATM)        (scale-invariant)

use serde_json::Value;

use crate::dates::parse_ymd;

const TARGET_DTE: f64 = 30.0;
const YEAR: f64 = 252.0; // trading-day annualization (per the reference repo)
const PUT_TARGET: f64 = -0.25;
const CALL_TARGET: f64 = 0.25;
const WING_TOL: f64 = 0.10; // max |delta - target| for the 25d wings
const ATM_TOL: f64 = 0.15; // max |delta - 0.50| for the ATM estimate
const MAX_REL_SPREAD: f64 = 0.25; // drop quotes wider than 25% relative spread

// Plausibility guards against corrupt vendor rows (some Dolt days have garbage
// IV/delta, e.g. ATM IV = 148% or deltas pinned near 0).
const IV_MIN: f64 = 0.01; // 1% vol floor for a single quote
const IV_MAX: f64 = 2.5; // 250% vol ceiling for a single quote
const ATM_IV_MAX: f64 = 1.5; // 30D ATM IV never exceeded ~85% even in Mar-2020
const MAX_IV_RATIO: f64 = 4.0; // the 3 extracted IVs shouldn't differ by >4x
const SKEW_ABS_LO: f64 = -0.05; // equity skew is ~always positive
const SKEW_ABS_HI: f64 = 0.80;

/// One cleaned option quote.
#[derive(Clone, Copy, Debug)]
struct Quote {
    dte: f64,
    is_call: bool,
    iv: f64,
    delta: f64,
}

/// The three IVs we extract from a single expiry.
#[derive(Clone, Copy, Debug)]
struct ExpiryIv {
    dte: f64,
    iv_25p: Option<f64>,
    iv_25c: Option<f64>,
    iv_atm: Option<f64>,
}

/// Final per-day result.
#[derive(Clone, Debug)]
pub struct DaySkew {
    pub date: String,
    pub iv_25p: f64,
    pub iv_25c: f64,
    pub iv_atm: f64,
    pub skew_abs: f64,  // IV25p - IV25c, in vol points (e.g. 0.04 = 4 vol pts)
    pub skew_norm: f64, // skew_abs / IV_atm
    pub dte_lo: f64,
    pub dte_hi: f64,
    // ATM-IV term structure: front (shortest) vs back (longest) expiry that day.
    pub ts_slope: f64, // front_atm_iv - back_atm_iv (vol pts); >0 = inverted/backwardation
    pub ts_ratio: f64, // front_atm_iv / back_atm_iv;          >1 = inverted (stress)
}

fn parse_f(v: &Value, key: &str) -> Option<f64> {
    v.get(key)?.as_str()?.trim().parse::<f64>().ok()
}

/// Parse + clean a day's rows into usable quotes.
fn clean_quotes(date_days: i64, rows: &[Value]) -> Vec<Quote> {
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let exp = match r.get("expiration").and_then(|s| s.as_str()).and_then(parse_ymd) {
            Some(e) => e,
            None => continue,
        };
        let dte = (exp - date_days) as f64;
        if dte <= 0.0 {
            continue;
        }
        let cp = match r.get("call_put").and_then(|s| s.as_str()) {
            Some(s) => s,
            None => continue,
        };
        let is_call = cp.eq_ignore_ascii_case("Call") || cp == "C";
        let iv = match parse_f(r, "vol") {
            Some(x) if x.is_finite() && (IV_MIN..=IV_MAX).contains(&x) => x,
            _ => continue,
        };
        let delta = match parse_f(r, "delta") {
            Some(x) if x.is_finite() => x,
            _ => continue,
        };
        // Relative bid-ask spread filter (drop wide/illiquid quotes).
        if let (Some(bid), Some(ask)) = (parse_f(r, "bid"), parse_f(r, "ask")) {
            let mid = 0.5 * (bid + ask);
            if mid > 0.0 {
                let rel = (ask - bid) / mid;
                if rel > MAX_REL_SPREAD {
                    continue;
                }
            }
        }
        out.push(Quote { dte, is_call, iv, delta });
    }
    out
}

/// From quotes at a single expiry, pull the 25d put, 25d call, and ATM IVs.
fn expiry_iv(dte: f64, quotes: &[Quote]) -> ExpiryIv {
    // nearest-delta selection within tolerance
    let pick = |want: f64, is_call: bool, tol: f64| -> Option<f64> {
        quotes
            .iter()
            .filter(|q| q.is_call == is_call)
            .map(|q| (q, (q.delta - want).abs()))
            .filter(|(_, err)| *err <= tol)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(q, _)| q.iv)
    };

    let iv_25p = pick(PUT_TARGET, false, WING_TOL);
    let iv_25c = pick(CALL_TARGET, true, WING_TOL);

    // ATM = average of ~+0.50 call and ~-0.50 put; fall back to whichever exists.
    let atm_c = pick(0.50, true, ATM_TOL);
    let atm_p = pick(-0.50, false, ATM_TOL);
    let iv_atm = match (atm_c, atm_p) {
        (Some(c), Some(p)) => Some(0.5 * (c + p)),
        (Some(c), None) => Some(c),
        (None, Some(p)) => Some(p),
        (None, None) => None,
    };

    ExpiryIv { dte, iv_25p, iv_25c, iv_atm }
}

/// Interpolate an IV to TARGET_DTE in total-variance space between two expiries.
/// `lo`/`hi` are (dte, iv). If only one is usable, returns it unchanged.
fn interp_variance(lo: Option<(f64, f64)>, hi: Option<(f64, f64)>) -> Option<f64> {
    match (lo, hi) {
        (Some((d1, iv1)), Some((d2, iv2))) if (d2 - d1).abs() > 1e-9 => {
            let t1 = d1 / YEAR;
            let t2 = d2 / YEAR;
            let tt = TARGET_DTE / YEAR;
            let v1 = iv1 * iv1 * t1;
            let v2 = iv2 * iv2 * t2;
            let w = (tt - t1) / (t2 - t1);
            let vt = v1 + w * (v2 - v1);
            if vt > 0.0 && tt > 0.0 {
                Some((vt / tt).sqrt())
            } else {
                None
            }
        }
        (Some((_, iv)), _) => Some(iv),
        (_, Some((_, iv))) => Some(iv),
        _ => None,
    }
}

/// Compute the synthetic 30-DTE 25-delta skew for one trading day.
/// Returns None if the chain that day can't support the calculation.
pub fn compute_day_skew(date: &str, rows: &[Value]) -> Option<DaySkew> {
    let date_days = parse_ymd(date)?;
    let quotes = clean_quotes(date_days, rows);
    if quotes.is_empty() {
        return None;
    }

    // Group quotes by integer DTE and reduce each expiry to its 3 IVs.
    let mut dtes: Vec<f64> = quotes.iter().map(|q| q.dte).collect();
    dtes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dtes.dedup();

    let mut expiries: Vec<ExpiryIv> = dtes
        .iter()
        .map(|&d| {
            let q: Vec<Quote> = quotes.iter().copied().filter(|q| q.dte == d).collect();
            expiry_iv(d, &q)
        })
        .collect();
    // keep only expiries that yield at least the two wings
    expiries.retain(|e| e.iv_25p.is_some() && e.iv_25c.is_some());
    if expiries.is_empty() {
        return None;
    }

    // Choose the two expiries bracketing 30 DTE.
    let below = expiries.iter().filter(|e| e.dte <= TARGET_DTE).max_by(|a, b| {
        a.dte.partial_cmp(&b.dte).unwrap()
    });
    let above = expiries.iter().filter(|e| e.dte > TARGET_DTE).min_by(|a, b| {
        a.dte.partial_cmp(&b.dte).unwrap()
    });

    let (lo, hi): (&ExpiryIv, Option<&ExpiryIv>) = match (below, above) {
        (Some(b), Some(a)) => (b, Some(a)),
        (Some(b), None) => (b, None),    // all expiries <= 30: use the nearest below
        (None, Some(a)) => (a, None),    // all expiries > 30: use the nearest above
        (None, None) => return None,
    };

    let lo_pair = |f: fn(&ExpiryIv) -> Option<f64>| f(lo).map(|iv| (lo.dte, iv));
    let hi_pair = |f: fn(&ExpiryIv) -> Option<f64>| hi.and_then(|h| f(h).map(|iv| (h.dte, iv)));

    let iv_25p = interp_variance(lo_pair(|e| e.iv_25p), hi_pair(|e| e.iv_25p))?;
    let iv_25c = interp_variance(lo_pair(|e| e.iv_25c), hi_pair(|e| e.iv_25c))?;
    let iv_atm = interp_variance(lo_pair(|e| e.iv_atm), hi_pair(|e| e.iv_atm))
        .filter(|x| *x > 0.0);

    let skew_abs = iv_25p - iv_25c;
    let iv_atm_val = iv_atm.unwrap_or(f64::NAN);

    // ---- Day-level plausibility filter (drop corrupt vendor days) ----
    if !iv_atm_val.is_finite() || iv_atm_val <= 0.03 || iv_atm_val > ATM_IV_MAX {
        return None;
    }
    let trio = [iv_25p, iv_25c, iv_atm_val];
    let iv_lo = trio.iter().cloned().fold(f64::INFINITY, f64::min);
    let iv_hi = trio.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if iv_lo < 0.02 || iv_hi > IV_MAX || (iv_lo > 0.0 && iv_hi / iv_lo > MAX_IV_RATIO) {
        return None;
    }
    if !(SKEW_ABS_LO..=SKEW_ABS_HI).contains(&skew_abs) {
        return None;
    }

    let skew_norm = skew_abs / iv_atm_val;

    // ---- ATM-IV term structure: front (shortest) vs back (longest) expiry ----
    // ATM IV per expiry across ALL expiries (not just wing-complete ones).
    let atm_by_exp: Vec<(f64, f64)> = dtes
        .iter()
        .filter(|&&d| d >= 5.0)
        .filter_map(|&d| {
            let q: Vec<Quote> = quotes.iter().copied().filter(|x| x.dte == d).collect();
            expiry_iv(d, &q).iv_atm.map(|iv| (d, iv))
        })
        .collect();
    let (mut ts_slope, mut ts_ratio) = (f64::NAN, f64::NAN);
    if let (Some(front), Some(back)) = (atm_by_exp.first(), atm_by_exp.last()) {
        // need a real spread between the two tenors
        if back.0 - front.0 >= 10.0 && front.1 > 0.0 && back.1 > 0.0 {
            ts_slope = front.1 - back.1;
            ts_ratio = front.1 / back.1;
        }
    }

    Some(DaySkew {
        date: date.to_string(),
        iv_25p,
        iv_25c,
        iv_atm: iv_atm_val,
        skew_abs,
        skew_norm,
        dte_lo: lo.dte,
        dte_hi: hi.map(|h| h.dte).unwrap_or(lo.dte),
        ts_slope,
        ts_ratio,
    })
}
