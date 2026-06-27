//! Minimal std-only HTTP server. Two pages:
//!   /candles  -> TradingView-style candles (3 panes) + spot-up-vol-up markers + hover legend
//!   /surface  -> MenthorQ-style 3D IV surface (Moneyness × Expiration × IV%), date animation
//! Rust computes; browser renders with Lightweight Charts and Three.js.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::Arc;

use serde::Serialize;

use crate::{bs, fetch, fred, price, svi};

const NX: usize = 40; // moneyness grid points
const NY: usize = 30; // dte grid points
const MNY0: f64 = 0.80;
const MNY1: f64 = 1.20;
const MAX_DTE: i64 = 180; // surface term-structure horizon (days); drops multi-year LEAPS
const PCT_WIN: usize = 252;

#[derive(Serialize)]
struct Pt {
    date: String,
    skew_norm: f64,
    atm_iv: f64,
    skew_pct: f64,
    iv_pct: f64,
}

#[derive(Serialize)]
struct Ohlc {
    date: String,
    o: f64,
    h: f64,
    l: f64,
    c: f64,
    skew_pct: f64,
    iv_pct: f64,
    iv: f64,   // forward-filled ATM IV (browser recomputes spot-up-vol-up at any lookback)
    sig: bool,
    top: bool, // ML top (고점) signal  — orange
    bot: bool, // ML bottom (저점) signal — teal
}

/// Load the ML turning-point signals (data/signals.csv: date,top,bottom).
fn load_signals() -> std::collections::HashMap<String, (bool, bool)> {
    let mut m = std::collections::HashMap::new();
    if let Ok(txt) = std::fs::read_to_string("data/signals.csv") {
        for line in txt.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            if f.len() >= 3 {
                m.insert(f[0].to_string(), (f[1].trim() == "1", f[2].trim() == "1"));
            }
        }
    }
    m
}

#[derive(Serialize)]
struct Surface {
    date: String,
    moneyness: Vec<f64>,
    dtes: Vec<f64>,
    z: Vec<Vec<f64>>,
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

fn build_series(symbol: &str) -> Result<Vec<Pt>, String> {
    let text = std::fs::read_to_string(format!("out/skew_{symbol}.csv"))
        .map_err(|e| format!("missing out/skew_{symbol}.csv: {e}"))?;
    let mut pts = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 10 {
            continue;
        }
        let g = |i: usize| f[i].parse::<f64>().unwrap_or(f64::NAN);
        pts.push(Pt {
            date: f[0].to_string(),
            skew_norm: g(5),
            atm_iv: g(3),
            skew_pct: f64::NAN,
            iv_pct: f64::NAN,
        });
    }
    let skp = rolling_pct(&pts.iter().map(|p| p.skew_norm).collect::<Vec<_>>(), PCT_WIN);
    let ivp = rolling_pct(&pts.iter().map(|p| p.atm_iv).collect::<Vec<_>>(), PCT_WIN);
    for (i, p) in pts.iter_mut().enumerate() {
        p.skew_pct = skp[i];
        p.iv_pct = ivp[i];
    }
    Ok(pts)
}

/// (25Δ skew_norm, ATM IV) ~30DTE from a daily CBOE SPX snapshot.
/// ATM = ~50Δ call; skew_norm = (IV@25Δ-put − IV@25Δ-call) / ATM-IV.
fn snapshot_skew_iv(date: &str) -> Option<(f64, f64)> {
    let text = std::fs::read_to_string(format!("data/snap/SPX/{date}.csv")).ok()?;
    let (mut atm, mut natm, mut atmw, mut natmw) = (0.0f64, 0usize, 0.0f64, 0usize);
    let (mut c25, mut nc25, mut p25, mut np25) = (0.0f64, 0usize, 0.0f64, 0usize);
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("strike") {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 7 {
            continue;
        }
        let is_call = f[2] == "C";
        let g = |i: usize| f[i].parse::<f64>().ok();
        if let (Some(dte), Some(delta), Some(iv)) = (g(1), g(5), g(6)) {
            if !(0.01..=2.0).contains(&iv) || !(20.0..=45.0).contains(&dte) {
                continue;
            }
            if is_call {
                if (0.45..=0.55).contains(&delta) { atm += iv; natm += 1; }
                if (0.40..=0.60).contains(&delta) { atmw += iv; natmw += 1; }
                if (0.20..=0.30).contains(&delta) { c25 += iv; nc25 += 1; }
            } else if (-0.30..=-0.20).contains(&delta) {
                p25 += iv; np25 += 1;
            }
        }
    }
    let atm_iv = if natm > 0 { atm / natm as f64 }
        else if natmw > 0 { atmw / natmw as f64 }
        else { return None };
    let skew_norm = if nc25 > 0 && np25 > 0 && atm_iv > 0.0 {
        (p25 / np25 as f64 - c25 / nc25 as f64) / atm_iv
    } else {
        f64::NAN
    };
    Some((skew_norm, atm_iv))
}

/// date -> (25Δ skew_norm, ATM IV) for every available daily SPX snapshot (CI-updated).
fn snapshot_map() -> std::collections::HashMap<String, (f64, f64)> {
    let mut m = std::collections::HashMap::new();
    if let Ok(rd) = std::fs::read_dir("data/snap/SPX") {
        for e in rd.flatten() {
            if let Some(stem) = e.path().file_stem().and_then(|x| x.to_str()) {
                if let Some(v) = snapshot_skew_iv(stem) {
                    m.insert(stem.to_string(), v);
                }
            }
        }
    }
    m
}

fn build_ohlc(symbol: &str) -> Result<Vec<Ohlc>, String> {
    let cache = fetch::default_cache_root();
    let bars = price::load_ohlc(Path::new(&cache))?;
    let pts = build_series(symbol)?;
    let start = pts.first().map(|p| p.date.clone()).unwrap_or_default();
    // RAW (skew_norm, atm_iv) from the Dolt history series (frozen at its last date).
    let sorted: Vec<(String, (f64, f64))> = pts
        .iter()
        .map(|p| (p.date.clone(), (p.skew_norm, p.atm_iv)))
        .collect();
    let bars: Vec<_> = bars.into_iter().filter(|b| b.0 >= start).collect();
    let n = bars.len();
    let (mut rskew, mut ivff) = (vec![f64::NAN; n], vec![f64::NAN; n]);
    let mut j = 0;
    let mut last = (f64::NAN, f64::NAN);
    for i in 0..n {
        while j < sorted.len() && sorted[j].0 <= bars[i].0 {
            last = sorted[j].1;
            j += 1;
        }
        rskew[i] = last.0;
        ivff[i] = last.1;
    }
    // Splice daily SPX-snapshot raw values (25Δ skew_norm, ATM IV) over the Dolt-frozen
    // forward-fill, so the skew%/IV% panels AND the spot-up-vol-up marker all keep
    // updating from the option data CI collects every day.
    let snap = snapshot_map();
    for i in 0..n {
        if let Some(&(sk, iv)) = snap.get(&bars[i].0) {
            if sk.is_finite() {
                rskew[i] = sk;
            }
            if iv.is_finite() {
                ivff[i] = iv;
            }
        }
    }
    // bar-level rolling percentiles (now live through the latest snapshot)
    let skp = rolling_pct(&rskew, PCT_WIN);
    let ivp = rolling_pct(&ivff, PCT_WIN);
    let lb = 1;
    let signals = load_signals();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &bars[i];
        // default spot-up-vol-up (1d); the browser recomputes at any lookback from `iv`.
        let sig = i >= lb
            && ivff[i].is_finite()
            && ivff[i - lb].is_finite()
            && b.4 > bars[i - lb].4
            && ivff[i] > ivff[i - lb];
        let (top, bot) = signals.get(&b.0).copied().unwrap_or((false, false));
        out.push(Ohlc {
            date: b.0.clone(), o: b.1, h: b.2, l: b.3, c: b.4,
            skew_pct: skp[i], iv_pct: ivp[i], iv: ivff[i], sig, top, bot,
        });
    }
    Ok(out)
}

fn interp(samples: &[(f64, f64)], x: f64) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    if x <= samples[0].0 {
        return samples[0].1;
    }
    if x >= samples[samples.len() - 1].0 {
        return samples[samples.len() - 1].1;
    }
    for w in samples.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x >= x0 && x <= x1 {
            let t = if (x1 - x0).abs() < 1e-12 { 0.0 } else { (x - x0) / (x1 - x0) };
            return y0 + t * (y1 - y0);
        }
    }
    samples[samples.len() - 1].1
}

/// Per-expiry option quotes: dte -> (strike, delta, iv, is_put).
type Smile = BTreeMap<i64, Vec<(f64, f64, f64, bool)>>;

/// Live SPX option smile from a stored daily CBOE snapshot CSV
/// (strike,dte,cp,oi,gamma,delta,iv,...; spot from the `# spot,` header).
fn snap_smile(date: &str) -> Option<(Smile, f64)> {
    let text = std::fs::read_to_string(format!("data/snap/SPX/{date}.csv")).ok()?;
    let mut spot = f64::NAN;
    let mut by: Smile = BTreeMap::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# spot,") {
            spot = rest.split(',').next().and_then(|x| x.parse().ok()).unwrap_or(f64::NAN);
            continue;
        }
        if line.starts_with("strike") {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 7 {
            continue;
        }
        let put = f[2].eq_ignore_ascii_case("P");
        if !put && !f[2].eq_ignore_ascii_case("C") {
            continue;
        }
        let (Ok(k), Ok(dte), Ok(d), Ok(iv)) =
            (f[0].parse::<f64>(), f[1].parse::<f64>(), f[5].parse::<f64>(), f[6].parse::<f64>())
        else {
            continue;
        };
        let dte = dte.round() as i64;
        if iv > 0.01 && iv < 2.5 && d.abs() >= 0.02 && d.abs() <= 0.98 && (3..=MAX_DTE).contains(&dte)
            && k > 0.0
        {
            by.entry(dte).or_default().push((k, d, iv, put));
        }
    }
    if !spot.is_finite() || spot <= 0.0 || by.len() < 2 {
        return None;
    }
    Some((by, spot))
}

/// Build a Moneyness × Expiration IV surface (OTM blend: OTM puts below spot,
/// OTM calls above) from the live SPX CBOE snapshot.
fn build_surface(_symbol: &str, date: &str) -> Surface {
    let mut empty = Surface { date: date.to_string(), moneyness: vec![], dtes: vec![], z: vec![] };
    let Some((by, spot)) = snap_smile(date) else {
        return empty;
    };
    let dtes_keys: Vec<i64> = by.keys().copied().collect();
    if dtes_keys.len() < 2 {
        return empty;
    }

    let moneyness: Vec<f64> =
        (0..NX).map(|i| MNY0 + (MNY1 - MNY0) * (i as f64) / ((NX - 1) as f64)).collect();

    // per-expiry OTM-blend smile -> IV at each moneyness grid point
    let mut exp_rows: Vec<(f64, Vec<f64>)> = Vec::new();
    for &dte in &dtes_keys {
        let mut smile: Vec<(f64, f64)> = by[&dte]
            .iter()
            .filter(|q| {
                if q.3 {
                    q.1 >= -0.5 && q.1 <= -0.03 // OTM put
                } else {
                    q.1 <= 0.5 && q.1 >= 0.03 // OTM call
                }
            })
            .map(|q| (q.0 / spot, q.2))
            .collect();
        smile.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if smile.len() < 4 {
            continue;
        }
        // SVI fit per expiry in (log-moneyness, total variance); fall back to linear interp.
        let t = (dte as f64) / 365.0;
        let pts: Vec<(f64, f64)> = smile.iter().map(|&(m, iv)| (m.ln(), iv * iv * t)).collect();
        // clamp evaluation to the quoted log-moneyness range (flat-extrapolate beyond,
        // so SVI wings don't blow up where there are no real strikes).
        let kmin = pts.first().map(|p| p.0).unwrap_or(f64::NEG_INFINITY);
        let kmax = pts.last().map(|p| p.0).unwrap_or(f64::INFINITY);
        let row: Vec<f64> = match svi::fit(&pts) {
            Some(s) => moneyness
                .iter()
                .map(|&m| {
                    let k = m.ln().clamp(kmin, kmax);
                    let w = s.w(k);
                    if w > 0.0 && t > 0.0 { (w / t).sqrt() } else { f64::NAN }
                })
                .collect(),
            None => moneyness.iter().map(|&m| interp(&smile, m)).collect(),
        };
        exp_rows.push((dte as f64, row));
    }
    if exp_rows.len() < 2 {
        return empty;
    }
    let dmin = exp_rows.first().unwrap().0;
    let dmax = exp_rows.last().unwrap().0;
    let dtes: Vec<f64> =
        (0..NY).map(|j| dmin + (dmax - dmin) * (j as f64) / ((NY - 1) as f64)).collect();
    let mut z = Vec::with_capacity(NY);
    for &dy in &dtes {
        let mut r = vec![f64::NAN; NX];
        for k in 0..NX {
            let s: Vec<(f64, f64)> = exp_rows.iter().map(|(dte, row)| (*dte, row[k])).collect();
            r[k] = interp(&s, dy);
        }
        z.push(r);
    }
    empty.moneyness = moneyness;
    empty.dtes = dtes;
    empty.z = z;
    empty
}

#[derive(Serialize)]
struct GammaProfile {
    und: String,
    date: String,
    spot: f64,
    net_mm: f64,
    call_mm: f64,
    put_mm: f64,
    net_vex_mm: f64,
    flip: f64,
    strikes: Vec<f64>,
    gex: Vec<f64>, // per-strike dealer GEX, $mm per 1% move
    vex: Vec<f64>, // per-strike dealer VEX (vanna), $mm per 1 vol-pt move
}

#[derive(Serialize)]
struct GexPoint {
    date: String,
    price: f64,
    dix: f64,
    gex: f64,
}

/// SqueezeMetrics free DIX/GEX history (DIX.csv: date,price,dix,gex). ~2011→now, SPX net GEX in $.
fn read_dix() -> Vec<GexPoint> {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string("DIX.csv") {
        for line in t.lines().skip(1) {
            let f: Vec<&str> = line.trim().split(',').collect();
            if f.len() < 4 {
                continue;
            }
            if let (Ok(p), Ok(d), Ok(g)) =
                (f[1].trim().parse(), f[2].trim().parse(), f[3].trim().parse())
            {
                out.push(GexPoint { date: f[0].trim().to_string(), price: p, dix: d, gex: g });
            }
        }
    }
    out
}

#[derive(Serialize)]
struct GexPlusPt {
    date: String,
    spot: f64,
    gex: f64,
    vex: f64,
    gexplus: f64,
}

/// Our forward GEX/VEX/GEX+ time series from collected CBOE snapshots (data/gamma_<UND>.csv).
fn read_gexplus(und: &str) -> Vec<GexPlusPt> {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string(format!("data/gamma_{und}.csv")) {
        for line in t.lines().skip(1) {
            let f: Vec<&str> = line.trim().split(',').collect();
            if f.len() < 7 {
                continue;
            }
            let p = |i: usize| f[i].trim().parse::<f64>().unwrap_or(f64::NAN);
            out.push(GexPlusPt { date: f[0].to_string(), spot: p(1), gex: p(4), vex: p(5), gexplus: p(6) });
        }
    }
    out
}

// ---- McElligott skew / VIX vol-of-vol panels ----

#[derive(Serialize)]
struct VixSkewPt {
    date: String,
    ratio: f64, // IV(25Δ call) / IV(ATM call) at ~3M  (cm3)
    iv_atm: f64,
    iv_25c: f64,
    spot: f64,
}

/// cm3: VIX 3M (≈90 DTE) call skew = IV(25Δ call)/IV(ATM call), per stored VIX snapshot.
fn build_vix_call_skew() -> Vec<VixSkewPt> {
    let mut out = Vec::new();
    for date in snap_dates("VIX") {
        let Ok(text) = std::fs::read_to_string(format!("data/snap/VIX/{date}.csv")) else { continue };
        let mut spot = f64::NAN;
        let mut by: BTreeMap<i64, Vec<(f64, f64)>> = BTreeMap::new(); // dte -> (call delta, iv)
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("# spot,") {
                spot = rest.split(',').next().and_then(|x| x.parse().ok()).unwrap_or(f64::NAN);
                continue;
            }
            if line.starts_with("strike") {
                continue;
            }
            let f: Vec<&str> = line.split(',').collect();
            if f.len() < 7 || f[2] != "C" {
                continue;
            }
            // strike,dte,cp,oi,gamma,delta,iv,...
            let (Ok(dte), Ok(delta), Ok(iv)) =
                (f[1].parse::<f64>(), f[5].parse::<f64>(), f[6].parse::<f64>())
            else {
                continue;
            };
            if iv > 0.01 && iv < 10.0 && delta > 0.0 && delta < 1.0 {
                by.entry(dte.round() as i64).or_default().push((delta, iv));
            }
        }
        if !spot.is_finite() {
            continue;
        }
        // expiry nearest 90 DTE with enough strikes
        let mut chosen: Option<Vec<(f64, f64)>> = None;
        let mut bestd = i64::MAX;
        for (&dte, v) in &by {
            if v.len() < 4 {
                continue;
            }
            let d = (dte - 90).abs();
            if d < bestd {
                bestd = d;
                chosen = Some(v.clone());
            }
        }
        let Some(mut smile) = chosen else { continue };
        smile.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // by delta ascending
        let iv_atm = interp(&smile, 0.5);
        let iv_25c = interp(&smile, 0.25);
        if iv_atm > 0.0 && iv_25c > 0.0 {
            out.push(VixSkewPt { date, ratio: iv_25c / iv_atm, iv_atm, iv_25c, spot });
        }
    }
    out
}

#[derive(Serialize)]
struct VixGammaPt {
    date: String,
    total: f64, // net dealer gamma imbalance ($mm, signed) — cm4 top
    call: f64,  // signed dealer call gamma ($mm; dealers short VIX calls ⇒ negative) — cm4 bottom
    spot: f64,
}

/// cm4: VIX dealer total gamma imbalance + call gamma, from the forward gamma_VIX.csv series.
fn read_vix_gamma() -> Vec<VixGammaPt> {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string("data/gamma_VIX.csv") {
        for line in t.lines().skip(1) {
            let f: Vec<&str> = line.trim().split(',').collect();
            if f.len() < 5 {
                continue;
            }
            // date,spot,call_gex_mm,put_gex_mm,net_gex_mm,...
            let p = |i: usize| f[i].trim().parse::<f64>().unwrap_or(f64::NAN);
            out.push(VixGammaPt { date: f[0].to_string(), total: p(4), call: -p(2), spot: p(1) });
        }
    }
    out
}

// ---- (1) Credit spreads (cross-asset leading indicator, FRED) ----

#[derive(Serialize)]
struct CreditPt {
    date: String,
    hy: f64,       // ICE BofA US High Yield OAS (%)
    ig: f64,       // ICE BofA US IG OAS (%)
    hy_chg20: f64, // 20-business-day change in HY OAS (widening rate = stress building)
    close: f64,    // SPX (for the lead-vs-equity overlay)
}

/// HY/IG credit spreads (cached FRED series) joined with SPX close. Credit tends to
/// crack before equities, so the 20-day widening rate is the leading signal here.
fn read_credit() -> Vec<CreditPt> {
    let hy = fred::load("credit_hy");
    if hy.is_empty() {
        return Vec::new();
    }
    let ig: BTreeMap<String, f64> = fred::load("credit_ig").into_iter().collect();
    let spx: BTreeMap<String, f64> = read_dix().into_iter().map(|p| (p.date, p.price)).collect();
    let hyv: Vec<f64> = hy.iter().map(|(_, v)| *v).collect();
    let mut out = Vec::with_capacity(hy.len());
    for (i, (d, v)) in hy.iter().enumerate() {
        out.push(CreditPt {
            date: d.clone(),
            hy: *v,
            ig: ig.get(d).copied().unwrap_or(f64::NAN),
            hy_chg20: if i >= 20 { v - hyv[i - 20] } else { f64::NAN },
            close: spx.get(d).copied().unwrap_or(f64::NAN),
        });
    }
    out
}

// ---- (2) DIX / GEX regime (SqueezeMetrics DIX.csv, 2011→now) ----

fn snap_dates(und: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(format!("data/snap/{und}")) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("csv") {
                if let Some(s) = p.file_stem().and_then(|x| x.to_str()) {
                    out.push(s.to_string());
                }
            }
        }
    }
    out.sort();
    out
}

/// Dealer-gamma profile by strike + zero-gamma flip, from a stored CBOE snapshot.
fn build_gamma(und: &str, date: &str) -> Option<GammaProfile> {
    let vix = und.eq_ignore_ascii_case("VIX");
    let text = std::fs::read_to_string(format!("data/snap/{und}/{date}.csv")).ok()?;
    let mut spot = f64::NAN;
    // (strike, dte, is_call, oi, gamma, iv)
    let mut opts: Vec<(f64, f64, bool, f64, f64, f64)> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# spot,") {
            spot = rest.split(',').next().and_then(|x| x.parse().ok()).unwrap_or(f64::NAN);
            continue;
        }
        if line.starts_with("strike") {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 7 {
            continue;
        }
        let p = |i: usize| f[i].parse::<f64>().ok();
        if let (Some(k), Some(dte), Some(oi), Some(g), Some(iv)) = (p(0), p(1), p(3), p(4), p(6)) {
            opts.push((k, dte, f[2] == "C", oi, g, iv));
        }
    }
    if !spot.is_finite() || opts.is_empty() {
        return None;
    }
    let sgn = |is_call: bool| -> f64 {
        if vix {
            if is_call { -1.0 } else { 1.0 }
        } else if is_call {
            1.0
        } else {
            -1.0
        }
    };
    let factor = spot * spot * 0.01; // GEX: $ per 1% spot move
    let vfac = spot * 0.01; // VEX: $ delta per 1 vol-point move
    let (lo, hi) = (spot * 0.80, spot * 1.20);
    let mut by: BTreeMap<i64, f64> = BTreeMap::new(); // strike*100 -> net $ GEX
    let mut vby: BTreeMap<i64, f64> = BTreeMap::new(); // strike*100 -> net $ VEX
    let (mut call, mut put, mut net_vex) = (0.0, 0.0, 0.0);
    for &(k, dte, is_call, oi, g, iv) in &opts {
        let gd = g * oi * 100.0 * factor;
        let vanna = if (0.01..=2.0).contains(&iv) {
            bs::bs_vanna(spot, k, dte / 365.0, iv)
        } else {
            0.0
        };
        let vd = vanna * oi * 100.0 * vfac;
        if is_call {
            call += gd;
        } else {
            put += gd;
        }
        net_vex += sgn(is_call) * vd;
        if k >= lo && k <= hi {
            *by.entry((k * 100.0) as i64).or_default() += sgn(is_call) * gd;
            *vby.entry((k * 100.0) as i64).or_default() += sgn(is_call) * vd;
        }
    }
    let net = if vix { -call + put } else { call - put };

    // zero-gamma flip: recompute net GEX at hypothetical spots via BS gamma, find the crossing nearest spot.
    let net_at = |s: f64| -> f64 {
        let f2 = s * s * 0.01;
        let mut n = 0.0;
        for &(k, dte, is_call, oi, _g, iv) in &opts {
            if iv <= 0.01 || iv > 2.0 {
                continue;
            }
            n += sgn(is_call) * bs::bs_gamma(s, k, dte / 365.0, iv) * oi * 100.0 * f2;
        }
        n
    };
    let (s0, s1, steps) = (spot * 0.85, spot * 1.15, 160usize);
    let mut flip = f64::NAN;
    let mut best = f64::INFINITY;
    let mut prev_s = s0;
    let mut prev_v = net_at(s0);
    for i in 1..=steps {
        let s = s0 + (s1 - s0) * (i as f64) / (steps as f64);
        let v = net_at(s);
        if prev_v.is_finite() && v.is_finite() && prev_v.signum() != v.signum() && (v - prev_v).abs() > 0.0 {
            let cross = prev_s + (s - prev_s) * (prev_v / (prev_v - v));
            if (cross - spot).abs() < best {
                best = (cross - spot).abs();
                flip = cross;
            }
        }
        prev_s = s;
        prev_v = v;
    }

    Some(GammaProfile {
        und: und.to_string(),
        date: date.to_string(),
        spot,
        net_mm: net / 1e6,
        call_mm: call / 1e6,
        put_mm: put / 1e6,
        net_vex_mm: net_vex / 1e6,
        flip,
        strikes: by.keys().map(|k| *k as f64 / 100.0).collect(),
        gex: by.values().map(|v| v / 1e6).collect(),
        vex: vby.values().map(|v| v / 1e6).collect(),
    })
}

fn respond(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-cache, no-store, must-revalidate\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn handle(mut stream: TcpStream, ohlc: &str, dates_json: &str, symbol: &str) {
    let mut buf = [0u8; 8192];
    let nr = stream.read(&mut buf).unwrap_or(0);
    let req = String::from_utf8_lossy(&buf[..nr]);
    let path = req
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("/");
    let (route, query) = path.split_once('?').unwrap_or((path, ""));
    let page = |html: &str| html.replace("__STYLE__", STYLE).replace("__NAV__", NAV);
    let q = |k: &str| {
        let pre = format!("{k}=");
        query.split('&').find_map(|kv| kv.strip_prefix(pre.as_str())).unwrap_or("")
    };
    match route {
        "/" | "/candles" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(CANDLE_HTML).as_bytes()),
        "/vix" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(VIX_HTML).as_bytes()),
        "/credit" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(CREDIT_HTML).as_bytes()),
        "/surface" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(SURFACE_HTML).as_bytes()),
        "/gamma" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(GAMMA_HTML).as_bytes()),
        "/gex" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(GEX_HTML).as_bytes()),
        "/gexplus" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(GEXPLUS_HTML).as_bytes()),
        "/api/ohlc" => respond(&mut stream, "200 OK", "application/json", ohlc.as_bytes()),
        "/api/dates" => respond(&mut stream, "200 OK", "application/json", dates_json.as_bytes()),
        "/api/vix_skew" => {
            let body = serde_json::to_string(&build_vix_call_skew()).unwrap_or_else(|_| "[]".into());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/api/vix_gamma" => {
            let body = serde_json::to_string(&read_vix_gamma()).unwrap_or_else(|_| "[]".into());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/api/credit" => {
            let body = serde_json::to_string(&read_credit()).unwrap_or_else(|_| "[]".into());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/api/surface" => {
            let body = serde_json::to_string(&build_surface(symbol, q("date"))).unwrap_or_else(|_| "{}".into());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/api/gex_history" => {
            let body = serde_json::to_string(&read_dix()).unwrap_or_else(|_| "[]".into());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/api/gexplus" => {
            let und = if q("und").is_empty() { "SPX" } else { q("und") };
            let body = serde_json::to_string(&read_gexplus(und)).unwrap_or_else(|_| "[]".into());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/api/gamma_dates" => {
            let und = if q("und").is_empty() { "SPX" } else { q("und") };
            let body = serde_json::to_string(&snap_dates(und)).unwrap_or_else(|_| "[]".into());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        "/api/gamma" => {
            let und = if q("und").is_empty() { "SPX" } else { q("und") };
            let body = serde_json::to_string(&build_gamma(und, q("date"))).unwrap_or_else(|_| "null".into());
            respond(&mut stream, "200 OK", "application/json", body.as_bytes());
        }
        _ => respond(&mut stream, "404 Not Found", "text/plain", b"not found"),
    }
}

pub fn run(symbol: &str) {
    // Serve reads committed/cached data ONLY — no network fetch at startup (no Yahoo).
    // The daily GitHub Action keeps data/ current; run `skew update` or `git pull` to
    // refresh locally. This keeps the dashboard from blocking on any data source.
    let ohlc = match build_ohlc(symbol) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}\n(run `skew backfill {symbol}` first)");
            return;
        }
    };
    let n_sig = ohlc.iter().filter(|b| b.sig).count();
    let ohlc_json = Arc::new(serde_json::to_string(&ohlc).unwrap());
    let chain_dates = snap_dates("SPX");
    let n_dates = chain_dates.len();
    let dates_json = Arc::new(serde_json::to_string(&chain_dates).unwrap());
    let symbol = Arc::new(symbol.to_string());

    let addr = "127.0.0.1:8080";
    let listener = match TcpListener::bind(addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind {addr} failed: {e}");
            return;
        }
    };
    println!("skew analytics  ->  http://{addr}/candles   and   http://{addr}/surface");
    println!("  {} daily bars, {} spot-up-vol-up signals, {} surface dates", ohlc.len(), n_sig, n_dates);
    for stream in listener.incoming().flatten() {
        let (o, d, s) = (Arc::clone(&ohlc_json), Arc::clone(&dates_json), Arc::clone(&symbol));
        std::thread::spawn(move || handle(stream, &o, &d, &s));
    }
}

const NAV: &str = r#"<header><h1>Skew Analytics</h1>
<nav><a href="/candles" id="nav-c">Candles</a><a href="/credit" id="nav-r">Credit</a><a href="/vix" id="nav-v">VIX</a><a href="/surface" id="nav-s">Vol Surface</a><a href="/gamma" id="nav-g">Dealer Gamma</a><a href="/gex" id="nav-x">Net GEX</a><a href="/gexplus" id="nav-p">GEX+</a></nav></header>"#;

const STYLE: &str = r#"
  :root{--bg:#06080d;--panel:#0d1420;--line:#1c2738;--txt:#cdd6e5;--dim:#7c8aa0;--accent:#5ccfe6}
  *{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--txt);font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif}
  header{padding:14px 24px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:22px}
  header h1{font-size:16px;margin:0;font-weight:700;letter-spacing:.3px}
  nav{display:flex;gap:6px} nav a{color:var(--dim);text-decoration:none;font-size:13px;padding:6px 12px;border-radius:7px}
  nav a:hover{color:var(--txt);background:#131c2b} nav a.active{color:#06080d;background:var(--accent);font-weight:600}
  .wrap{padding:18px 24px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
  .card .sub{font-size:11.5px;color:var(--dim);padding:10px 16px 0}
"#;

const CANDLE_HTML: &str = r###"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skew Analytics — Candles</title>
<script src="https://unpkg.com/lightweight-charts@5.0.8/dist/lightweight-charts.standalone.production.js"></script>
<style>__STYLE__
  .wrap{padding:0}
  .sub{padding:8px 20px}
  #candle{height:calc(100vh - 226px);width:100%} .chartwrap{position:relative}
  .tgbar{gap:14px;padding-top:0;flex-wrap:wrap} .tgbar label{display:flex;align-items:center;gap:5px;cursor:pointer;color:var(--txt)} .tgbar input{accent-color:var(--accent)}
  #legend{position:absolute;left:16px;top:10px;z-index:5;font-size:12.5px;background:#0b1220e6;border:1px solid var(--line);border-radius:7px;padding:7px 12px;opacity:0;pointer-events:none;white-space:nowrap}
  .ctrlbar{display:flex;gap:20px;align-items:center;padding:11px 20px;font-size:12.5px;color:var(--dim)}
  .ctrlbar input[type=number]{background:#0b1220;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:4px 8px;width:60px;font-size:13px}
  .leg{display:flex;align-items:center;gap:6px}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%}
  .dot.red{background:#ff1744;box-shadow:0 0 6px #ff1744} .dot.green{background:#26ff8a;box-shadow:0 0 6px #26ff8a}
  .dot.orange{background:#ff9800;box-shadow:0 0 6px #ff9800} .dot.teal{background:#14b8a6;box-shadow:0 0 6px #14b8a6}
  .ma-swatch{display:inline-block;width:16px;height:2px;background:#4fc3f7;vertical-align:middle}
</style></head><body>__NAV__
<div class="wrap">
  <div class="ctrlbar">
    <span class="leg"><span class="ma-swatch"></span>MA <input type="number" id="maPeriod" value="100" min="2" max="400" step="1"> d</span>
    <span class="leg"><span class="dot red"></span>spot↑vol↑ lookback <input type="number" id="suvuLb" value="1" min="1" max="60" step="1"> d</span>
    <span style="margin-left:auto">25Δ skew %ile · ATM IV %ile (daily SPX snapshots) · hover for values</span>
  </div>
  <div class="ctrlbar tgbar">
    <span>show:</span>
    <label><input type="checkbox" id="tgMA" checked> MA</label>
    <label><input type="checkbox" id="tgSkew" checked> 25Δ skew %ile</label>
    <label><input type="checkbox" id="tgIV" checked> ATM IV %ile</label>
    <label><input type="checkbox" id="tgSuvu" checked> spot↑vol↑</label>
    <label><input type="checkbox" id="tgTop" checked> <span class="dot orange"></span> top 고점</label>
    <label><input type="checkbox" id="tgBot" checked> <span class="dot teal"></span> bottom 저점</label>
  </div>
  <div class="chartwrap"><div id="legend"></div><div id="candle"></div></div>
</div>
<script>
const LC=window.LightweightCharts;
document.getElementById('nav-c').classList.add('active');
(async function(){
  const bars=await (await fetch('/api/ohlc')).json();
  const el=document.getElementById('candle');
  const chart=LC.createChart(el,{autoSize:true,height:600,
    layout:{background:{color:'#0d1420'},textColor:'#9fb0c8',panes:{separatorColor:'#1c2738',separatorHoverColor:'#2a3850'}},
    grid:{vertLines:{color:'#101826'},horzLines:{color:'#101826'}},
    rightPriceScale:{borderColor:'#1c2738'},timeScale:{borderColor:'#1c2738'},
    crosshair:{mode:LC.CrosshairMode.Normal,vertLine:{color:'#3a4a63',labelBackgroundColor:'#1c2738'},horzLine:{color:'#3a4a63',labelBackgroundColor:'#1c2738'}}});
  const candle=chart.addSeries(LC.CandlestickSeries,{upColor:'#26a69a',downColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350'},0);
  candle.setData(bars.map(b=>({time:b.date,open:b.o,high:b.h,low:b.l,close:b.c})));
  // moving-average overlay on the candle pane (adjustable period)
  const maSeries=chart.addSeries(LC.LineSeries,{color:'#4fc3f7',lineWidth:2,priceLineVisible:false,lastValueVisible:false},0);
  function sma(p){const o=[];let s=0;for(let i=0;i<bars.length;i++){s+=bars[i].c;if(i>=p)s-=bars[i-p].c;if(i>=p-1)o.push({time:bars[i].date,value:s/p});}return o;}
  const maInput=document.getElementById('maPeriod');
  function updateMA(){const p=Math.max(2,Math.min(400,+maInput.value||100));maSeries.setData(sma(p));}
  updateMA();maInput.addEventListener('input',updateMA);
  const skew=chart.addSeries(LC.LineSeries,{color:'#ff5c7a',lineWidth:1,priceLineVisible:false,lastValueVisible:false},1);
  skew.setData(bars.filter(b=>b.skew_pct!=null).map(b=>({time:b.date,value:b.skew_pct})));
  const ivS=chart.addSeries(LC.LineSeries,{color:'#ffb454',lineWidth:1,priceLineVisible:false,lastValueVisible:false},2);
  ivS.setData(bars.filter(b=>b.iv_pct!=null).map(b=>({time:b.date,value:b.iv_pct})));
  try{const p=chart.panes();if(p[0])p[0].setHeight(400);if(p[1])p[1].setHeight(120);if(p[2])p[2].setHeight(120);}catch(e){}
  // red: spot-up & vol-up over an adjustable lookback (close[i]>close[i-lb] AND IV[i]>IV[i-lb])
  function redMarkers(lb){const o=[];for(let i=lb;i<bars.length;i++){const a=bars[i],p=bars[i-lb];if(a.iv!=null&&p.iv!=null&&a.c>p.c&&a.iv>p.iv)o.push({time:a.date,position:'aboveBar',color:'#ff1744',shape:'circle'});}return o;}
  // ML turning-point signals: top (고점) orange above bar, bottom (저점) teal below bar
  function topMarkers(){return bars.filter(b=>b.top).map(b=>({time:b.date,position:'aboveBar',color:'#ff9800',shape:'circle',size:2}));}
  function botMarkers(){return bars.filter(b=>b.bot).map(b=>({time:b.date,position:'belowBar',color:'#14b8a6',shape:'circle',size:2}));}
  const markersPrim=LC.createSeriesMarkers(candle,[]);
  const suvuInput=document.getElementById('suvuLb');
  const isOn=id=>{const el=document.getElementById(id);return !el||el.checked;};
  function refreshMarkers(){
    const lb=Math.max(1,Math.min(60,+suvuInput.value||1));
    let all=[];
    if(isOn('tgSuvu')) all=all.concat(redMarkers(lb));
    if(isOn('tgTop')) all=all.concat(topMarkers());
    if(isOn('tgBot')) all=all.concat(botMarkers());
    all.sort((a,b)=> a.time<b.time?-1 : a.time>b.time?1 : 0);
    markersPrim.setMarkers(all);
  }
  refreshMarkers();
  suvuInput.addEventListener('input',refreshMarkers);
  // indicator on/off toggles
  const bind=(id,fn)=>{const e=document.getElementById(id); if(e) e.addEventListener('change',()=>fn(e.checked));};
  bind('tgMA',v=>maSeries.applyOptions({visible:v}));
  bind('tgSkew',v=>skew.applyOptions({visible:v}));
  bind('tgIV',v=>ivS.applyOptions({visible:v}));
  bind('tgSuvu',()=>refreshMarkers());
  bind('tgTop',()=>refreshMarkers());
  bind('tgBot',()=>refreshMarkers());
  chart.timeScale().fitContent();

  const legend=document.getElementById('legend');
  const f=(v,d)=>v==null?'–':(+v).toFixed(d);
  chart.subscribeCrosshairMove(p=>{
    if(!p.time||!p.point){legend.style.opacity=0;return;}
    const c=p.seriesData.get(candle),sk=p.seriesData.get(skew),iv=p.seriesData.get(ivS);
    legend.style.opacity=1;
    legend.innerHTML=`<b>${p.time}</b> &nbsp; O ${f(c&&c.open,1)} H ${f(c&&c.high,1)} L ${f(c&&c.low,1)} C ${f(c&&c.close,1)} &nbsp; `+
      `<span style="color:#ff5c7a">skew ${f(sk&&sk.value,0)}%ile</span> &nbsp; <span style="color:#ffb454">IV ${f(iv&&iv.value,0)}%ile</span>`;
  });
})();
</script></body></html>"###;

const SURFACE_HTML: &str = r###"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skew Analytics — Vol Surface</title>
<style>__STYLE__
  .wrap{padding:0}
  .canvas-host{position:relative;height:calc(100vh - 108px);background:#000}
  .ttl{position:absolute;left:24px;top:18px;z-index:4;pointer-events:none}
  .ttl h2{margin:0;font-size:22px;font-weight:700;letter-spacing:.2px}
  .ttl .ts{font-size:13px;color:var(--dim);margin-top:3px}
  .cbar{position:absolute;right:26px;top:90px;width:18px;height:340px;border-radius:3px;border:1px solid #222}
  .cbtxt{position:absolute;right:50px;font-size:11px;color:#aab4c4;font-variant-numeric:tabular-nums}
  .toolbar{display:flex;gap:14px;align-items:center;padding:12px 20px;border-top:1px solid var(--line);font-size:12px;color:var(--dim);background:var(--panel)}
  .toolbar input[type=range]{flex:1;accent-color:var(--accent)}
  button{background:transparent;color:var(--txt);border:1px solid var(--line);border-radius:7px;padding:6px 14px;cursor:pointer;font-size:13px}
  button:hover{border-color:var(--accent)} #dateLabel{min-width:96px;font-variant-numeric:tabular-nums;color:var(--txt);font-size:13px}
  .help{color:var(--dim);font-size:11px}
</style></head><body>__NAV__
<div class="wrap">
  <div class="canvas-host" id="host">
    <div class="ttl"><h2>Volatility Surface — SPY</h2><div class="ts" id="ts">Timestamp: —</div></div>
    <div class="cbar" id="cbar"></div>
    <div class="cbtxt" id="cb-hi" style="top:84px">60%</div>
    <div class="cbtxt" style="top:255px">30%</div>
    <div class="cbtxt" style="top:426px">0%</div>
    <div class="cbtxt" style="top:62px;color:#dfe6f0">IV</div>
  </div>
  <div class="toolbar">
    <button id="playBtn">▶ play</button>
    <button id="prevBtn" title="prev day">‹</button>
    <input type="range" id="slider" min="0" max="0" value="0">
    <button id="nextBtn" title="next day">›</button>
    <span id="dateLabel"></span>
    <label><input type="checkbox" id="wire" checked> mesh</label>
    <span class="help">Moneyness = strike ÷ spot &nbsp;(1.0 = ATM, &lt;1 = downside puts)</span>
    <span id="info"></span>
  </div>
</div>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
document.getElementById('nav-s').classList.add('active');

const SX=58, SZ=58, SY=30, IV_CAP=0.60, DTE_MAX=70, MN0=0.80, MN1=1.20;
const V=THREE.Vector3;
const xOf=m=>((m-MN0)/(MN1-MN0)-0.5)*SX;
const zOf=d=>(0.5-Math.min(d,DTE_MAX)/DTE_MAX)*SZ;   // dte 0 = front, DTE_MAX = back
const yOf=iv=>Math.max(0,Math.min(1,iv/IV_CAP))*SY;
const STOPS=[[0.28,0.0,0.45],[0.0,0.35,0.92],[0.0,0.82,0.92],[0.15,0.82,0.22],[0.97,0.92,0.10],[0.93,0.12,0.10]];
function turbo(t){t=Math.max(0,Math.min(1,t));const n=STOPS.length-1;const s=t*n;const i=Math.min(n-1,Math.floor(s));const f=s-i;const a=STOPS[i],b=STOPS[i+1];return new THREE.Color(a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f);}

const host=document.getElementById('host');
const scene=new THREE.Scene();scene.background=new THREE.Color('#000000');
const cam=new THREE.PerspectiveCamera(42,host.clientWidth/host.clientHeight,0.1,3000);cam.position.set(74,48,86);
const ren=new THREE.WebGLRenderer({antialias:true});ren.setPixelRatio(window.devicePixelRatio);ren.setSize(host.clientWidth,host.clientHeight);host.appendChild(ren.domElement);
const ctr=new OrbitControls(cam,ren.domElement);ctr.enableDamping=true;ctr.target.set(0,SY*0.42,0);
scene.add(new THREE.AmbientLight(0xffffff,0.85));
const dl=new THREE.DirectionalLight(0xffffff,0.8);dl.position.set(50,90,40);scene.add(dl);
const dl2=new THREE.DirectionalLight(0x99bbff,0.35);dl2.position.set(-60,30,-50);scene.add(dl2);
(function loop(){ctr.update();ren.render(scene,cam);requestAnimationFrame(loop);})();
window.addEventListener('resize',()=>{cam.aspect=host.clientWidth/host.clientHeight;cam.updateProjectionMatrix();ren.setSize(host.clientWidth,host.clientHeight);});

function label(text,color,scale){const cv=document.createElement('canvas');cv.width=300;cv.height=80;const x=cv.getContext('2d');x.fillStyle=color||'#aab4c4';x.font='bold 34px Inter,system-ui,sans-serif';x.textAlign='center';x.textBaseline='middle';x.fillText(text,150,40);const tx=new THREE.CanvasTexture(cv);tx.minFilter=THREE.LinearFilter;const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tx,transparent:true,depthTest:false,depthWrite:false}));const s=scale||1;sp.scale.set(12*s,3.2*s,1);return sp;}
function gl(a,b,c,o){return new THREE.Line(new THREE.BufferGeometry().setFromPoints([a,b]),new THREE.LineBasicMaterial({color:c,transparent:true,opacity:o==null?0.55:o}));}

// static framed grid box (built once)
const MN_TICKS=[0.80,0.90,1.00,1.10,1.20], DTE_TICKS=[0,15,30,45,60], IV_TICKS=[0,0.1,0.2,0.3,0.4,0.5,0.6];
(function buildFrame(){
  const g=new THREE.Group(); const hx=SX/2,hz=SZ/2, zb=zOf(DTE_MAX), zf=zOf(0), FLOOR=0x21314a, WALL=0x16202e;
  DTE_TICKS.forEach(d=>g.add(gl(new V(-hx,0,zOf(d)),new V(hx,0,zOf(d)),FLOOR)));
  MN_TICKS.forEach(m=>g.add(gl(new V(xOf(m),0,zf),new V(xOf(m),0,zb),FLOOR)));
  IV_TICKS.forEach(iv=>g.add(gl(new V(-hx,yOf(iv),zb),new V(hx,yOf(iv),zb),WALL)));
  MN_TICKS.forEach(m=>g.add(gl(new V(xOf(m),0,zb),new V(xOf(m),SY,zb),WALL)));
  IV_TICKS.forEach(iv=>g.add(gl(new V(-hx,yOf(iv),zf),new V(-hx,yOf(iv),zb),WALL)));
  DTE_TICKS.forEach(d=>g.add(gl(new V(-hx,0,zOf(d)),new V(-hx,SY,zOf(d)),WALL)));
  // tick labels
  MN_TICKS.forEach(m=>{const s=label(m.toFixed(2),'#9aa6ba',0.85);s.position.set(xOf(m),-2.4,zf+3.5);g.add(s);});
  DTE_TICKS.forEach(d=>{const s=label(d+'d','#9aa6ba',0.85);s.position.set(hx+3.5,-2.4,zOf(d));g.add(s);});
  [0,0.2,0.4,0.6].forEach(iv=>{const s=label((iv*100).toFixed(0)+'%','#9aa6ba',0.8);s.position.set(-hx-4.5,yOf(iv),zf);g.add(s);});
  // titles
  const a=label('Moneyness','#e6ecf5',1.25);a.position.set(0,-6.2,zf+6);g.add(a);
  const b=label('Expiration (days)','#e6ecf5',1.25);b.position.set(hx+12,-6.2,0);g.add(b);
  const c=label('Implied Vol (%)','#e6ecf5',1.25);c.position.set(-hx-12,SY+3.5,zf);g.add(c);
  const sk=label('Skew','#7f8da3',1.0);sk.position.set(0,SY*0.62,zb-1.5);g.add(sk);
  const ts=label('Term Structure','#7f8da3',1.0);ts.position.set(-hx-1.5,SY*0.62,0);g.add(ts);
  scene.add(g);
})();

let mesh=null,wire=null;
function build(d){
  const nx=(d.moneyness||[]).length,ny=(d.dtes||[]).length;
  if(!nx||!ny){document.getElementById('info').textContent='no chain';return;}
  if(mesh){scene.remove(mesh);mesh.geometry.dispose();}
  if(wire){scene.remove(wire);wire.geometry.dispose();}
  const pos=[],col=[];
  for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){
    const iv=d.z[j][i];
    pos.push(xOf(d.moneyness[i]), yOf(iv), zOf(d.dtes[j]));
    const c=turbo(iv/IV_CAP);col.push(c.r,c.g,c.b);
  }
  const idx=[];for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++){const a=j*nx+i,b=a+1,c=a+nx,e=c+1;idx.push(a,c,b,b,c,e);}
  const geo=new THREE.BufferGeometry();
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setAttribute('color',new THREE.Float32BufferAttribute(col,3));
  geo.setIndex(idx);geo.computeVertexNormals();
  mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:0.45,metalness:0.0}));
  scene.add(mesh);
  wire=new THREE.LineSegments(new THREE.WireframeGeometry(geo),new THREE.LineBasicMaterial({color:0x05080d,transparent:true,opacity:0.28}));
  wire.visible=document.getElementById('wire').checked;scene.add(wire);
  let zmin=9,zmax=-9;for(const r of d.z)for(const v of r){if(v<zmin)zmin=v;if(v>zmax)zmax=v;}
  document.getElementById('info').textContent=`IV ${(zmin*100).toFixed(0)}–${(zmax*100).toFixed(0)}%`;
  document.getElementById('ts').textContent='Timestamp: '+d.date+' EOD';
}
(function(){const cb=document.getElementById('cbar');let s='linear-gradient(to top';for(let k=0;k<=10;k++){const c=turbo(k/10);s+=`,rgb(${c.r*255|0},${c.g*255|0},${c.b*255|0})`;}cb.style.background=s+')';})();
document.getElementById('wire').addEventListener('change',e=>{if(wire)wire.visible=e.target.checked;});

(async function(){
  const DATES=await (await fetch('/api/dates')).json();
  const slider=document.getElementById('slider'),playBtn=document.getElementById('playBtn'),lab=document.getElementById('dateLabel');
  const cache=new Map();
  async function getS(date){if(cache.has(date))return cache.get(date);const d=await (await fetch('/api/surface?date='+date)).json();cache.set(date,d);return d;}
  let cur=0,playing=false,timer=null;
  async function show(i){cur=Math.max(0,Math.min(DATES.length-1,i));slider.value=cur;lab.textContent=DATES[cur];build(await getS(DATES[cur]));}
  function stop(){playing=false;playBtn.textContent='▶ play';if(timer)clearInterval(timer);}
  slider.max=Math.max(0,DATES.length-1);
  slider.addEventListener('input',()=>show(+slider.value));
  document.getElementById('prevBtn').addEventListener('click',()=>{stop();show(cur-1);});
  document.getElementById('nextBtn').addEventListener('click',()=>{stop();show(cur+1);});
  playBtn.addEventListener('click',()=>{playing=!playing;playBtn.textContent=playing?'⏸ pause':'▶ play';if(playing)timer=setInterval(()=>show(cur>=DATES.length-1?0:cur+1),150);else clearInterval(timer);});
  if(DATES.length)await show(DATES.length-1);
})();
</script></body></html>"###;

const GAMMA_HTML: &str = r###"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skew Analytics — Dealer Gamma</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>__STYLE__
  .wrap{padding:0}
  .toolbar{display:flex;gap:18px;align-items:center;padding:11px 20px;font-size:12.5px;color:var(--dim);border-bottom:1px solid var(--line);flex-wrap:wrap}
  select{background:#0b1220;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:5px 9px;font-size:13px}
  #chart{height:calc(100vh - 158px);width:100%}
  .stat{color:var(--txt)} .stat b{font-variant-numeric:tabular-nums}
  .pos{color:#26a69a} .neg{color:#ef5350} .spotc{color:#4fc3f7} .flipc{color:#ffb454}
</style></head><body>__NAV__
<div class="wrap">
  <div class="toolbar">
    <span>underlying <select id="undSel"><option>SPX</option><option>VIX</option></select></span>
    <span>date <select id="dateSel"></select></span>
    <span>metric <select id="metricSel"><option>GEX</option><option>VEX</option></select></span>
    <span id="info"></span>
    <span style="margin-left:auto"><span class="pos">■</span> long gamma (vol-suppressing) · <span class="neg">■</span> short gamma (accelerant) · <span class="spotc">│</span> spot · <span class="flipc">┊</span> γ-flip</span>
  </div>
  <div id="chart"></div>
</div>
<script>
document.getElementById('nav-g').classList.add('active');
const chart=echarts.init(document.getElementById('chart'),null,{renderer:'canvas'});
window.addEventListener('resize',()=>chart.resize());
const undSel=document.getElementById('undSel'),dateSel=document.getElementById('dateSel'),info=document.getElementById('info');
const AX={axisLine:{lineStyle:{color:'#33415a'}},axisLabel:{color:'#8595ad'},nameTextStyle:{color:'#8595ad'}};
function fmt(v){return (v>=0?'+':'')+v.toLocaleString(undefined,{maximumFractionDigits:0});}
let lastG=null;
function render(g){ lastG=g; draw(); }
function draw(){
  const g=lastG;
  if(!g){info.textContent='no snapshot for this date';chart.clear();return;}
  const metric=document.getElementById('metricSel').value;
  const arr=metric==='VEX'?g.vex:g.gex;
  const yName=metric==='VEX'?'Dealer VEX / vanna  ($mm / 1 vol-pt)':'Dealer GEX  ($mm / 1% move)';
  const unit=metric==='VEX'?'$mm/volpt':'$mm/1%';
  info.innerHTML=`<span class="stat">net GEX <b class="${g.net_mm>=0?'pos':'neg'}">${fmt(g.net_mm)}</b> · net VEX <b class="${g.net_vex_mm>=0?'pos':'neg'}">${fmt(g.net_vex_mm)}</b> $mm  ·  spot <b class="spotc">${g.spot.toFixed(0)}</b>  ·  γ-flip <b class="flipc">${g.flip&&isFinite(g.flip)?g.flip.toFixed(0):'—'}</b></span>`;
  const data=g.strikes.map((k,i)=>[k,arr[i]]);
  const ml=[{xAxis:g.spot,lineStyle:{color:'#4fc3f7',width:1.5},label:{formatter:'spot',color:'#4fc3f7',position:'insideEndTop'}}];
  if(g.flip&&isFinite(g.flip)) ml.push({xAxis:g.flip,lineStyle:{color:'#ffb454',width:1.5,type:'dashed'},label:{formatter:'γ-flip',color:'#ffb454',position:'insideEndBottom'}});
  chart.setOption({
    backgroundColor:'transparent',
    grid:{left:78,right:28,top:24,bottom:64},
    tooltip:{trigger:'axis',axisPointer:{type:'shadow'},backgroundColor:'#0b1220',borderColor:'#1c2738',textStyle:{color:'#cdd6e5'},
      formatter:p=>{const d=p[0].data;return `strike ${d[0]}<br/>${metric} ${fmt(d[1])} ${unit}`;}},
    xAxis:Object.assign({type:'value',name:'Strike',scale:true,splitLine:{show:false}},AX),
    yAxis:Object.assign({type:'value',name:yName,splitLine:{lineStyle:{color:'#141d2b'}}},AX),
    dataZoom:[{type:'inside'},{type:'slider',height:16,bottom:8,borderColor:'#1c2738',textStyle:{color:'#8595ad'}}],
    series:[{type:'bar',data,barWidth:'70%',
      itemStyle:{color:p=>p.value[1]>=0?'#26a69a':'#ef5350'},
      markLine:{symbol:'none',silent:true,data:ml}}]
  },true);
}
async function loadGamma(){
  const g=await (await fetch('/api/gamma?und='+undSel.value+'&date='+dateSel.value)).json();
  render(g);
}
async function loadDates(){
  const ds=await (await fetch('/api/gamma_dates?und='+undSel.value)).json();
  dateSel.innerHTML='';
  for(const d of ds){const o=document.createElement('option');o.value=d;o.textContent=d;dateSel.appendChild(o);}
  if(ds.length){dateSel.value=ds[ds.length-1];await loadGamma();}
  else{render(null);}
}
undSel.addEventListener('change',loadDates);
dateSel.addEventListener('change',loadGamma);
document.getElementById('metricSel').addEventListener('change',draw);
loadDates();
</script></body></html>"###;

const GEX_HTML: &str = r###"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skew Analytics — Net GEX</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>__STYLE__
  .wrap{padding:0}
  .sub{padding:10px 20px;font-size:12.5px;color:var(--dim)}
  #gex{height:calc(100vh - 130px);width:100%}
  .neg{color:#ef5350} .pos{color:#26a69a}
</style></head><body>__NAV__
<div class="wrap">
  <div class="sub"><b style="color:var(--txt)">Net Dealer Gamma (GEX) — SPX, 2011→now</b> (SqueezeMetrics) · <span class="neg">red = negative</span> (short gamma / accelerant) · <span class="pos">green = positive</span> (long gamma, vol-suppressing) · grey = SPX (log) · <span id="info"></span></div>
  <div id="gex"></div>
</div>
<script>
document.getElementById('nav-x').classList.add('active');
const AX={axisLine:{lineStyle:{color:'#33415a'}},axisLabel:{color:'#8595ad'},nameTextStyle:{color:'#8595ad'}};
const chart=echarts.init(document.getElementById('gex'),null,{renderer:'canvas'});
window.addEventListener('resize',()=>chart.resize());
(async function(){
  const d=await (await fetch('/api/gex_history')).json();
  if(!d.length){document.getElementById('info').textContent='no DIX.csv';return;}
  const xs=d.map(p=>p.date);
  const gex=d.map(p=>+(p.gex/1e9).toFixed(3));
  const spx=d.map(p=>p.price);
  const neg=d.filter(p=>p.gex<0).length;
  document.getElementById('info').textContent=`${d.length} days · ${neg} negative (${(neg/d.length*100).toFixed(1)}%)`;
  chart.setOption({
    backgroundColor:'transparent',
    grid:{left:74,right:66,top:18,bottom:72},
    tooltip:{trigger:'axis',backgroundColor:'#0b1220',borderColor:'#1c2738',textStyle:{color:'#cdd6e5'}},
    dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:12,borderColor:'#1c2738',textStyle:{color:'#8595ad'}}],
    xAxis:Object.assign({type:'category',data:xs,boundaryGap:false},AX),
    yAxis:[
      Object.assign({type:'value',name:'GEX  $bn / 1%',splitLine:{lineStyle:{color:'#141d2b'}}},AX),
      Object.assign({type:'value',name:'SPX',position:'right',splitLine:{show:false}},AX)
    ],
    series:[
      {name:'GEX',type:'line',data:gex,showSymbol:false,sampling:'lttb',lineStyle:{color:'#26c6da',width:1.4},areaStyle:{color:'rgba(38,198,218,0.10)'},z:3,
        markLine:{symbol:'none',silent:true,data:[{yAxis:0,lineStyle:{color:'#7a8aa0',type:'dashed',width:1}}]}},
      {name:'SPX',type:'line',yAxisIndex:1,data:spx,showSymbol:false,sampling:'lttb',lineStyle:{color:'#8593a8',width:1},z:1}
    ]
  });
})();
</script></body></html>"###;

const GEXPLUS_HTML: &str = r###"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skew Analytics — GEX+</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>__STYLE__
  .wrap{padding:0}
  .sub{padding:10px 20px;font-size:12.5px;color:var(--dim)}
  #gp{height:calc(100vh - 130px);width:100%}
</style></head><body>__NAV__
<div class="wrap">
  <div class="sub"><b style="color:var(--txt)">GEX+ = GEX + VEX</b> — forward series from our CBOE snapshots (grows daily; naive sign) · <span style="color:#26c6da">GEX</span> · <span style="color:#ffb454">VEX</span> · <span style="color:#e6ecf5">GEX+</span> · grey = SPX · <span id="info"></span></div>
  <div id="gp"></div>
</div>
<script>
document.getElementById('nav-p').classList.add('active');
const AX={axisLine:{lineStyle:{color:'#33415a'}},axisLabel:{color:'#8595ad'},nameTextStyle:{color:'#8595ad'}};
const chart=echarts.init(document.getElementById('gp'),null,{renderer:'canvas'});
window.addEventListener('resize',()=>chart.resize());
(async function(){
  const d=await (await fetch('/api/gexplus?und=SPX')).json();
  if(!d.length){document.getElementById('info').textContent='no snapshots yet — run skew update';return;}
  const xs=d.map(p=>p.date), last=d[d.length-1];
  document.getElementById('info').textContent=`${d.length} day(s) · latest GEX ${last.gex.toFixed(0)} · VEX ${last.vex.toFixed(0)} · GEX+ ${last.gexplus.toFixed(0)} $mm`;
  const line=(name,key,color,w)=>({name,type:'line',data:d.map(p=>p[key]),showSymbol:true,symbolSize:5,lineStyle:{color,width:w||1.5},itemStyle:{color}});
  chart.setOption({
    backgroundColor:'transparent',
    legend:{data:['GEX','VEX','GEX+'],textStyle:{color:'#aeb6c4'},top:6},
    grid:{left:74,right:66,top:38,bottom:64},
    tooltip:{trigger:'axis',backgroundColor:'#0b1220',borderColor:'#1c2738',textStyle:{color:'#cdd6e5'}},
    dataZoom:[{type:'inside'},{type:'slider',height:16,bottom:8,borderColor:'#1c2738',textStyle:{color:'#8595ad'}}],
    xAxis:Object.assign({type:'category',data:xs,boundaryGap:false},AX),
    yAxis:[
      Object.assign({type:'value',name:'$mm',splitLine:{lineStyle:{color:'#141d2b'}}},AX),
      Object.assign({type:'value',name:'SPX',position:'right',splitLine:{show:false}},AX)
    ],
    series:[
      line('GEX','gex','#26c6da'),
      line('VEX','vex','#ffb454'),
      Object.assign(line('GEX+','gexplus','#e6ecf5',2.2),{markLine:{symbol:'none',silent:true,data:[{yAxis:0,lineStyle:{color:'#7a8aa0',type:'dashed',width:1}}]}}),
      {name:'SPX',type:'line',yAxisIndex:1,data:d.map(p=>p.spot),showSymbol:false,lineStyle:{color:'#5b6678',width:1},z:1}
    ]
  });
})();
</script></body></html>"###;

const VIX_HTML: &str = r###"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skew Analytics — VIX</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>__STYLE__
  .wrap{padding:0}
  .sub{padding:10px 20px;font-size:12.5px;color:var(--dim)}
  #vx{height:calc(100vh - 130px);width:100%}
</style></head><body>__NAV__
<div class="wrap">
  <div class="sub"><b style="color:var(--txt)">VIX vol-of-vol (Nomura / McElligott style)</b> — forward series from our CBOE VIX snapshots (grows daily). Top: <span style="color:#e34a4a">3M call skew 25dC/ATM</span> (cm3). Middle/bottom: VIX dealer <span style="color:#cdd6e5">total gamma imbalance</span> & <span style="color:#cdd6e5">call gamma</span> (cm4). · <span id="info"></span></div>
  <div id="vx"></div>
</div>
<script>
document.getElementById('nav-v').classList.add('active');
const AX={axisLine:{lineStyle:{color:'#33415a'}},axisLabel:{color:'#8595ad'},nameTextStyle:{color:'#8595ad'}};
const chart=echarts.init(document.getElementById('vx'),null,{renderer:'canvas'});
window.addEventListener('resize',()=>chart.resize());
(async function(){
  const [sk,gm]=await Promise.all([
    (await fetch('/api/vix_skew')).json(),
    (await fetch('/api/vix_gamma')).json()
  ]);
  const dset=new Set([...sk.map(p=>p.date),...gm.map(p=>p.date)]);
  const xs=[...dset].sort();
  const mapv=(arr,key)=>{const m=new Map(arr.map(p=>[p.date,p[key]]));return xs.map(x=>m.has(x)?+(+m.get(x)).toFixed(4):null);};
  const ratio=mapv(sk,'ratio'), total=mapv(gm,'total'), call=mapv(gm,'call');
  const ls=sk[sk.length-1], lg=gm[gm.length-1];
  document.getElementById('info').textContent=
    `${xs.length} day(s)`+(sk.length?` · 3M call skew ${ls.ratio.toFixed(3)}`:' · no VIX snapshots')+
    (gm.length?` · total γ ${lg.total.toFixed(1)} · call γ ${lg.call.toFixed(1)} $mm`:'');
  const G={left:80,right:28};
  const ln=(name,data,color,i)=>({name,type:'line',data,xAxisIndex:i,yAxisIndex:i,showSymbol:true,symbolSize:5,connectNulls:true,lineStyle:{color,width:1.5},itemStyle:{color}});
  const zero={markLine:{symbol:'none',silent:true,data:[{yAxis:0,lineStyle:{color:'#445268',type:'dashed',width:1}}]}};
  chart.setOption({
    backgroundColor:'transparent',
    tooltip:{trigger:'axis',backgroundColor:'#0b1220',borderColor:'#1c2738',textStyle:{color:'#cdd6e5'}},
    axisPointer:{link:[{xAxisIndex:'all'}]},
    grid:[
      Object.assign({top:'6%',height:'23%'},G),
      Object.assign({top:'39%',height:'23%'},G),
      Object.assign({top:'71%',height:'23%'},G)
    ],
    xAxis:[
      Object.assign({type:'category',data:xs,gridIndex:0,boundaryGap:false,axisLabel:{show:false}},AX),
      Object.assign({type:'category',data:xs,gridIndex:1,boundaryGap:false,axisLabel:{show:false}},AX),
      Object.assign({type:'category',data:xs,gridIndex:2,boundaryGap:false},AX)
    ],
    yAxis:[
      Object.assign({type:'value',name:'3M 25dC/ATM',gridIndex:0,scale:true,splitLine:{lineStyle:{color:'#141d2b'}}},AX),
      Object.assign({type:'value',name:'Total γ imb. $mm',gridIndex:1,splitLine:{lineStyle:{color:'#141d2b'}}},AX),
      Object.assign({type:'value',name:'Call γ $mm',gridIndex:2,splitLine:{lineStyle:{color:'#141d2b'}}},AX)
    ],
    dataZoom:[{type:'inside',xAxisIndex:[0,1,2]},{type:'slider',xAxisIndex:[0,1,2],height:16,bottom:6,borderColor:'#1c2738',textStyle:{color:'#8595ad'}}],
    series:[
      ln('VIX 3M Call Skew (25dC/ATM)',ratio,'#e34a4a',0),
      Object.assign(ln('VIX Dealer Total Gamma Imbalance',total,'#9aa6ba',1),zero),
      Object.assign(ln('VIX Call Gamma',call,'#9aa6ba',2),zero)
    ]
  });
})();
</script></body></html>"###;

const CREDIT_HTML: &str = r###"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skew Analytics — Credit</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>__STYLE__
  .wrap{padding:0}
  .sub{padding:10px 20px;font-size:12.5px;color:var(--dim)}
  #cr{height:calc(100vh - 130px);width:100%}
</style></head><body>__NAV__
<div class="wrap">
  <div class="sub"><b style="color:var(--txt)">Credit Spreads — leading cross-asset signal</b> · <span style="color:#e34a4a">HY OAS %</span> · <span style="color:#9aa6ba">IG OAS %</span> · grey = SPX (log). Bottom: HY 20-day widening rate (&gt;0 = stress building, tends to lead equity drawdowns). · <span id="info"></span></div>
  <div id="cr"></div>
</div>
<script>
document.getElementById('nav-r').classList.add('active');
const AX={axisLine:{lineStyle:{color:'#33415a'}},axisLabel:{color:'#8595ad'},nameTextStyle:{color:'#8595ad'}};
const chart=echarts.init(document.getElementById('cr'),null,{renderer:'canvas'});
window.addEventListener('resize',()=>chart.resize());
(async function(){
  const d=await (await fetch('/api/credit')).json();
  if(!d.length){document.getElementById('info').textContent='no FRED credit data — run `skew update` (needs network)';return;}
  const xs=d.map(p=>p.date), last=d[d.length-1];
  document.getElementById('info').textContent=`${d.length} days · latest HY OAS ${last.hy.toFixed(2)}% · 20d chg ${(last.hy_chg20>=0?'+':'')+last.hy_chg20.toFixed(2)}`;
  const G={left:64,right:64};
  chart.setOption({
    backgroundColor:'transparent',
    legend:{data:['HY OAS','IG OAS','SPX'],textStyle:{color:'#aeb6c4'},top:6},
    tooltip:{trigger:'axis',backgroundColor:'#0b1220',borderColor:'#1c2738',textStyle:{color:'#cdd6e5'}},
    axisPointer:{link:[{xAxisIndex:'all'}]},
    grid:[Object.assign({top:'9%',height:'55%'},G),Object.assign({top:'73%',height:'19%'},G)],
    xAxis:[
      Object.assign({type:'category',data:xs,gridIndex:0,boundaryGap:false,axisLabel:{show:false}},AX),
      Object.assign({type:'category',data:xs,gridIndex:1,boundaryGap:false},AX)
    ],
    yAxis:[
      Object.assign({type:'value',name:'OAS %',gridIndex:0,scale:true,splitLine:{lineStyle:{color:'#141d2b'}}},AX),
      Object.assign({type:'log',name:'SPX',gridIndex:0,position:'right',splitLine:{show:false}},AX),
      Object.assign({type:'value',name:'HY 20d Δ',gridIndex:1,splitLine:{lineStyle:{color:'#141d2b'}}},AX)
    ],
    dataZoom:[{type:'inside',xAxisIndex:[0,1]},{type:'slider',xAxisIndex:[0,1],height:16,bottom:6,borderColor:'#1c2738',textStyle:{color:'#8595ad'}}],
    series:[
      {name:'HY OAS',type:'line',xAxisIndex:0,yAxisIndex:0,data:d.map(p=>p.hy),showSymbol:false,sampling:'lttb',lineStyle:{color:'#e34a4a',width:1.4},areaStyle:{color:'rgba(227,74,74,0.08)'}},
      {name:'IG OAS',type:'line',xAxisIndex:0,yAxisIndex:0,data:d.map(p=>p.ig),showSymbol:false,sampling:'lttb',lineStyle:{color:'#9aa6ba',width:1}},
      {name:'SPX',type:'line',xAxisIndex:0,yAxisIndex:1,data:d.map(p=>p.close),showSymbol:false,sampling:'lttb',lineStyle:{color:'#5b6678',width:1},z:1},
      {name:'HY 20d Δ',type:'line',xAxisIndex:1,yAxisIndex:2,data:d.map(p=>p.hy_chg20==null?null:+p.hy_chg20.toFixed(3)),showSymbol:false,sampling:'lttb',lineStyle:{color:'#ffb454',width:1},areaStyle:{color:'rgba(255,180,84,0.10)'},markLine:{symbol:'none',silent:true,data:[{yAxis:0,lineStyle:{color:'#445268',type:'dashed',width:1}}]}}
    ]
  });
})();
</script></body></html>"###;

