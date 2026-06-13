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

use crate::{bs, dates, fetch, price};

const NX: usize = 40; // moneyness grid points
const NY: usize = 30; // dte grid points
const MNY0: f64 = 0.80;
const MNY1: f64 = 1.20;
const PCT_WIN: usize = 252;

#[derive(Serialize)]
struct Pt {
    date: String,
    skew_norm: f64,
    atm_iv: f64,
    ts_ratio: f64,
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
    tsr: f64,  // forward-filled term-structure ratio (browser runs inversion-onset hysteresis)
    sig: bool,
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
            ts_ratio: g(9),
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

fn build_ohlc(symbol: &str) -> Result<Vec<Ohlc>, String> {
    let cache = fetch::default_cache_root();
    let bars = price::load_ohlc(Path::new(&cache))?;
    let pts = build_series(symbol)?;
    let start = pts.first().map(|p| p.date.clone()).unwrap_or_default();
    let sorted: Vec<(String, (f64, f64, f64, f64))> = pts
        .iter()
        .map(|p| (p.date.clone(), (p.skew_pct, p.iv_pct, p.atm_iv, p.ts_ratio)))
        .collect();
    let bars: Vec<_> = bars.into_iter().filter(|b| b.0 >= start).collect();
    let n = bars.len();
    let (mut skp, mut ivp, mut ivff, mut tsff) =
        (vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n], vec![f64::NAN; n]);
    let mut j = 0;
    let mut last = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    for i in 0..n {
        while j < sorted.len() && sorted[j].0 <= bars[i].0 {
            last = sorted[j].1;
            j += 1;
        }
        skp[i] = last.0;
        ivp[i] = last.1;
        ivff[i] = last.2;
        tsff[i] = last.3;
    }
    let lb = 5;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &bars[i];
        // default spot-up-vol-up (5d); the browser recomputes at any lookback from `iv`.
        let sig = i >= lb
            && ivff[i].is_finite()
            && ivff[i - lb].is_finite()
            && b.4 > bars[i - lb].4
            && ivff[i] > ivff[i - lb];
        out.push(Ohlc {
            date: b.0.clone(), o: b.1, h: b.2, l: b.3, c: b.4,
            skew_pct: skp[i], iv_pct: ivp[i], iv: ivff[i], tsr: tsff[i], sig,
        });
    }
    Ok(out)
}

fn list_chain_dates(symbol: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(format!("data/dolt/{symbol}")) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json")
                && e.metadata().map(|m| m.len() > 2).unwrap_or(false)
            {
                if let Some(s) = p.file_stem().and_then(|x| x.to_str()) {
                    out.push(s.to_string());
                }
            }
        }
    }
    out.sort();
    out
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

/// Build a Moneyness × Expiration IV surface (OTM blend: OTM puts below spot,
/// OTM calls above). Spot is estimated from the front expiry's 50Δ strike.
fn build_surface(symbol: &str, date: &str) -> Surface {
    let mut empty = Surface { date: date.to_string(), moneyness: vec![], dtes: vec![], z: vec![] };
    let path = format!("data/dolt/{symbol}/{date}.json");
    let Ok(bytes) = std::fs::read(&path) else { return empty };
    let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return empty };
    let Some(arr) = val.as_array() else { return empty };
    let Some(d0) = dates::parse_ymd(date) else { return empty };

    // per-expiry quotes: (strike, delta, iv, is_put)
    let mut by: BTreeMap<i64, Vec<(f64, f64, f64, bool)>> = BTreeMap::new();
    for row in arr {
        let cp = row.get("call_put").and_then(|x| x.as_str()).unwrap_or("");
        let put = cp.eq_ignore_ascii_case("Put");
        let call = cp.eq_ignore_ascii_case("Call");
        if !put && !call {
            continue;
        }
        let pf = |k: &str| row.get(k).and_then(|x| x.as_str()).and_then(|s| s.parse::<f64>().ok());
        let exp = row.get("expiration").and_then(|x| x.as_str()).and_then(dates::parse_ymd);
        if let (Some(iv), Some(d), Some(k), Some(e)) = (pf("vol"), pf("delta"), pf("strike"), exp) {
            let dte = e - d0;
            if iv > 0.01 && iv < 2.5 && d.abs() >= 0.02 && d.abs() <= 0.98 && dte >= 3 && k > 0.0 {
                by.entry(dte).or_default().push((k, d, iv, put));
            }
        }
    }
    let dtes_keys: Vec<i64> = by.keys().copied().collect();
    if dtes_keys.len() < 2 {
        return empty;
    }
    // estimate spot from the front expiry's 50-delta put strike
    let front = &by[&dtes_keys[0]];
    let mut put_ds: Vec<(f64, f64)> = front
        .iter()
        .filter(|q| q.3)
        .map(|q| (q.1, q.0))
        .collect(); // (delta, strike)
    put_ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let spot = interp(&put_ds, -0.5);
    if !spot.is_finite() || spot <= 0.0 {
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
        let row: Vec<f64> = moneyness.iter().map(|&m| interp(&smile, m)).collect();
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
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
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
        "/surface" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(SURFACE_HTML).as_bytes()),
        "/gamma" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(GAMMA_HTML).as_bytes()),
        "/gex" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(GEX_HTML).as_bytes()),
        "/gexplus" => respond(&mut stream, "200 OK", "text/html; charset=utf-8", page(GEXPLUS_HTML).as_bytes()),
        "/api/ohlc" => respond(&mut stream, "200 OK", "application/json", ohlc.as_bytes()),
        "/api/dates" => respond(&mut stream, "200 OK", "application/json", dates_json.as_bytes()),
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
    // incremental update on startup: price gap-fill + today's CBOE snapshot + dealer gamma
    println!("== data update ==");
    match price::update_ohlc(Path::new(&fetch::default_cache_root())) {
        Ok((n, last)) => println!("  price: {n} bars, last {last}"),
        Err(e) => eprintln!("  price: {e}"),
    }
    crate::cboe::update_all();

    let ohlc = match build_ohlc(symbol) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}\n(run `skew backfill {symbol}` first)");
            return;
        }
    };
    let n_sig = ohlc.iter().filter(|b| b.sig).count();
    let ohlc_json = Arc::new(serde_json::to_string(&ohlc).unwrap());
    let chain_dates = list_chain_dates(symbol);
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
<nav><a href="/candles" id="nav-c">Candles</a><a href="/surface" id="nav-s">Vol Surface</a><a href="/gamma" id="nav-g">Dealer Gamma</a><a href="/gex" id="nav-x">Net GEX</a><a href="/gexplus" id="nav-p">GEX+</a></nav></header>"#;

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
  #candle{height:calc(100vh - 190px);width:100%} .chartwrap{position:relative}
  #legend{position:absolute;left:16px;top:10px;z-index:5;font-size:12.5px;background:#0b1220e6;border:1px solid var(--line);border-radius:7px;padding:7px 12px;opacity:0;pointer-events:none;white-space:nowrap}
  .ctrlbar{display:flex;gap:20px;align-items:center;padding:11px 20px;font-size:12.5px;color:var(--dim)}
  .ctrlbar input[type=number]{background:#0b1220;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:4px 8px;width:60px;font-size:13px}
  .leg{display:flex;align-items:center;gap:6px}
  .dot{display:inline-block;width:9px;height:9px;border-radius:50%}
  .dot.red{background:#ff1744;box-shadow:0 0 6px #ff1744} .dot.green{background:#26ff8a;box-shadow:0 0 6px #26ff8a}
  .ma-swatch{display:inline-block;width:16px;height:2px;background:#4fc3f7;vertical-align:middle}
</style></head><body>__NAV__
<div class="wrap">
  <div class="ctrlbar">
    <span class="leg"><span class="ma-swatch"></span>MA <input type="number" id="maPeriod" value="100" min="2" max="400" step="1"> d</span>
    <span class="leg"><span class="dot red"></span>spot↑vol↑ lookback <input type="number" id="suvuLb" value="5" min="1" max="60" step="1"> d</span>
    <span class="leg"><span class="dot green"></span>TS invert · fire&gt;<input type="number" id="tsUp" value="1.05" min="1" max="1.5" step="0.01"> rearm&lt;<input type="number" id="tsLo" value="1.00" min="0.8" max="1.2" step="0.01"></span>
    <span style="margin-left:auto">middle = 25Δ skew %ile · bottom = ATM IV %ile · hover for values</span>
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
  try{const p=chart.panes();if(p[0])p[0].setHeight(380);if(p[1])p[1].setHeight(110);if(p[2])p[2].setHeight(110);}catch(e){}
  // red: spot-up & vol-up over an adjustable lookback (close[i]>close[i-lb] AND IV[i]>IV[i-lb])
  function redMarkers(lb){const o=[];for(let i=lb;i<bars.length;i++){const a=bars[i],p=bars[i-lb];if(a.iv!=null&&p.iv!=null&&a.c>p.c&&a.iv>p.iv)o.push({time:a.date,position:'aboveBar',color:'#ff1744',shape:'circle'});}return o;}
  // green: term-structure inversion ONSET via hysteresis — fire when ts_ratio crosses above `up`,
  // re-arm only after it falls below `lo` (so each inversion episode flags exactly once, at its start)
  function greenMarkers(up,lo){const o=[];let inv=false;for(let i=0;i<bars.length;i++){const r=bars[i].tsr;if(r==null)continue;if(!inv&&r>up){o.push({time:bars[i].date,position:'aboveBar',color:'#26ff8a',shape:'circle'});inv=true;}else if(inv&&r<lo){inv=false;}}return o;}
  const markersPrim=LC.createSeriesMarkers(candle,[]);
  const suvuInput=document.getElementById('suvuLb'),upInput=document.getElementById('tsUp'),loInput=document.getElementById('tsLo');
  function refreshMarkers(){
    const lb=Math.max(1,Math.min(60,+suvuInput.value||5));
    const up=+upInput.value||1.05, lo=+loInput.value||1.0;
    const all=redMarkers(lb).concat(greenMarkers(up,lo)).sort((a,b)=> a.time<b.time?-1 : a.time>b.time?1 : (a.color==='#ff1744'?-1:1));
    markersPrim.setMarkers(all);
  }
  refreshMarkers();[suvuInput,upInput,loInput].forEach(el=>el.addEventListener('input',refreshMarkers));
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
