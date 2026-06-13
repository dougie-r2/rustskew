//! Long-option directional backtest.
//!
//! For each signal we BUY a 30-day option (call or put) on the signal day,
//! priced with Black-Scholes off our IV, and hold to expiry (~21 trading days),
//! settling at intrinsic value. Per-trade return = (payoff - premium)/premium.
//! This isolates the directional edge net of the premium paid (the VRP bleed)
//! and theta, with no exit-IV assumption.
//!
//! Signals:
//!   #3 cheap-vol breakout : enter only when IV percentile is LOW; direction by trend (SPX vs MA)
//!   #4 skew contrarian    : skew %ile >= HI -> contrarian CALL ; <= LO -> PUT
//! Benchmarks: always-long call / put (shows the structural bleed).

use std::collections::BTreeMap;
use std::path::Path;

use crate::{bs, fetch, price};

const T_YEARS: f64 = 30.0 / 365.0; // 30-calendar-day option
const HOLD_TD: usize = 21; // ~1 month to expiry, in trading days
const MA_WIN: usize = 100; // trend filter window (trading days)
const PCT_WIN: usize = 252; // rolling percentile window (skew observations)
const IV_PCT_LOW: f64 = 30.0; // "cheap vol" threshold
const SKEW_PCT_HI: f64 = 80.0;
const SKEW_PCT_LO: f64 = 20.0;
const TARGET_DELTA: f64 = 0.25;

#[derive(Clone)]
struct Decision {
    pidx: usize, // index into the (contiguous) price series
    s_entry: f64,
    s_exit: f64, // close HOLD_TD trading days later (≈ expiry)
    iv_atm: f64,
    iv_25p: f64,
    iv_25c: f64,
    iv_pct: f64,
    skew_pct: f64,
    uptrend: bool,
}

#[derive(Clone, Copy)]
enum Struct {
    Atm,
    Otm25,
}

fn parse_skew_csv(path: &str) -> Vec<(String, [f64; 4])> {
    // returns date -> [iv_25p, iv_25c, iv_atm, skew_norm], in file order (date-sorted)
    let mut out = Vec::new();
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return out,
    };
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() < 6 {
            continue;
        }
        let p = |i: usize| f[i].parse::<f64>().unwrap_or(f64::NAN);
        out.push((f[0].to_string(), [p(1), p(2), p(3), p(5)]));
    }
    out
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

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}
fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}
fn std(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

/// Run one (signal, structure) over ALL eligible decision days (overlapping
/// entries → larger sample for the edge estimate) and return the trade returns.
fn run_signal(
    decisions: &[Decision],
    structure: Struct,
    decide: &dyn Fn(&Decision) -> Option<bool>, // Some(is_call) to enter, None to skip
) -> Vec<f64> {
    let mut rets = Vec::new();
    for d in decisions {
        let is_call = match decide(d) {
            Some(c) => c,
            None => continue,
        };
        let (k, sigma) = match structure {
            Struct::Atm => (d.s_entry, d.iv_atm),
            Struct::Otm25 => {
                let sigma = if is_call { d.iv_25c } else { d.iv_25p };
                (
                    bs::strike_for_delta(is_call, d.s_entry, sigma, T_YEARS, TARGET_DELTA),
                    sigma,
                )
            }
        };
        if !(sigma.is_finite() && sigma > 0.0 && k.is_finite() && k > 0.0) {
            continue;
        }
        let premium = bs::bs_price(is_call, d.s_entry, k, T_YEARS, sigma);
        if premium <= 1e-6 {
            continue;
        }
        let payoff = bs::intrinsic(is_call, d.s_exit, k);
        rets.push((payoff - premium) / premium);
    }
    rets
}

fn report_row(label: &str, rets: &[f64]) {
    if rets.is_empty() {
        println!("  {:<28} (no trades)", label);
        return;
    }
    let m = mean(rets);
    let md = median(rets);
    let sd = std(rets);
    let win = rets.iter().filter(|r| **r > 0.0).count() as f64 / rets.len() as f64 * 100.0;
    let gains: f64 = rets.iter().filter(|r| **r > 0.0).sum();
    let losses: f64 = rets.iter().filter(|r| **r < 0.0).map(|r| r.abs()).sum();
    let pf = if losses > 0.0 { gains / losses } else { f64::INFINITY };
    // crude t-stat (overlapping entries inflate this — directional only)
    let tstat = m / (sd / (rets.len() as f64).sqrt());
    println!(
        "  {:<28} n={:>4}  win {:>4.0}%  mean {:>+7.1}%  med {:>+7.1}%  PF {:>4.2}  t≈{:>5.2}",
        label,
        rets.len(),
        win,
        m * 100.0,
        md * 100.0,
        pf,
        tstat
    );
}

pub fn run(symbol: &str) {
    let skew = parse_skew_csv(&format!("out/skew_{symbol}.csv"));
    if skew.is_empty() {
        eprintln!("no out/skew_{symbol}.csv — run `skew backfill {symbol}` first");
        return;
    }
    let cache = fetch::default_cache_root();
    let spx = match price::load_spx(Path::new(&cache)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("price load error: {e}");
            return;
        }
    };
    // contiguous price series + index
    let closes: Vec<f64> = spx.iter().map(|(_, c)| *c).collect();
    let idx_of: BTreeMap<&str, usize> =
        spx.iter().enumerate().map(|(i, (d, _))| (d.as_str(), i)).collect();
    // trend MA on the contiguous series
    let ma: Vec<f64> = (0..closes.len())
        .map(|i| {
            if i + 1 >= MA_WIN {
                mean(&closes[(i + 1 - MA_WIN)..=i])
            } else {
                f64::NAN
            }
        })
        .collect();

    // rolling percentiles over the (date-ordered) skew series
    let iv_series: Vec<f64> = skew.iter().map(|(_, a)| a[2]).collect(); // iv_atm
    let sk_series: Vec<f64> = skew.iter().map(|(_, a)| a[3]).collect(); // skew_norm
    let iv_pct = rolling_pct(&iv_series, PCT_WIN);
    let sk_pct = rolling_pct(&sk_series, PCT_WIN);

    // build decisions
    let mut decisions = Vec::new();
    for (k, (date, a)) in skew.iter().enumerate() {
        let Some(&pidx) = idx_of.get(date.as_str()) else { continue };
        if pidx + HOLD_TD >= closes.len() {
            continue;
        }
        if !(iv_pct[k].is_finite() && sk_pct[k].is_finite() && ma[pidx].is_finite()) {
            continue;
        }
        let [iv_25p, iv_25c, iv_atm, _sn] = *a;
        if !(iv_atm.is_finite() && iv_25p.is_finite() && iv_25c.is_finite()) {
            continue;
        }
        decisions.push(Decision {
            pidx,
            s_entry: closes[pidx],
            s_exit: closes[pidx + HOLD_TD],
            iv_atm,
            iv_25p,
            iv_25c,
            iv_pct: iv_pct[k],
            skew_pct: sk_pct[k],
            uptrend: closes[pidx] > ma[pidx],
        });
    }

    println!(
        "=== {symbol} long-option backtest  ({} eligible days, 30D option, hold {}td to expiry) ===",
        decisions.len(),
        HOLD_TD
    );
    println!("(per-trade % return on premium; r=q=0 BS; entries overlap so t-stat is optimistic)\n");

    let signals: Vec<(&str, Box<dyn Fn(&Decision) -> Option<bool>>)> = vec![
        (
            "#3 cheap-vol breakout",
            Box::new(|d: &Decision| {
                if d.iv_pct <= IV_PCT_LOW {
                    Some(d.uptrend) // uptrend -> call, downtrend -> put
                } else {
                    None
                }
            }),
        ),
        (
            "#4 skew contrarian",
            Box::new(|d: &Decision| {
                if d.skew_pct >= SKEW_PCT_HI {
                    Some(true) // very steep skew -> contrarian CALL
                } else if d.skew_pct <= SKEW_PCT_LO {
                    Some(false) // very flat/complacent -> PUT
                } else {
                    None
                }
            }),
        ),
        ("BENCH always-call", Box::new(|_d: &Decision| Some(true))),
        ("BENCH always-put", Box::new(|_d: &Decision| Some(false))),
    ];

    for (name, decide) in &signals {
        println!("{name}:");
        report_row("  ATM", &run_signal(&decisions, Struct::Atm, decide.as_ref()));
        report_row("  25Δ OTM", &run_signal(&decisions, Struct::Otm25, decide.as_ref()));
        println!();
    }
}
