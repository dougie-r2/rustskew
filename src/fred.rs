//! FRED (St. Louis Fed) keyless CSV series — fetch + cache. Used for credit spreads
//! and other cross-asset leading indicators. Same `curl` transport as the rest of the
//! app; FRED's `fredgraph.csv?id=<ID>` endpoint needs no API key.

use crate::fetch::curl_get;

// (FRED series id, local cache name). HY/IG OAS are in percent; "." marks missing days.
const SERIES: &[(&str, &str)] = &[
    ("BAMLH0A0HYM2", "credit_hy"), // ICE BofA US High Yield OAS (%)
    ("BAMLC0A0CM", "credit_ig"),   // ICE BofA US Corporate (IG) OAS (%)
];

/// Fetch every series and cache it to data/fred_<name>.csv (date,value).
pub fn update_all() {
    std::fs::create_dir_all("data").ok();
    for (id, name) in SERIES {
        match fetch_series(id) {
            Ok(rows) if !rows.is_empty() => {
                let mut s = String::from("date,value\n");
                for (d, v) in &rows {
                    s.push_str(&format!("{d},{v}\n"));
                }
                let _ = std::fs::write(format!("data/fred_{name}.csv"), s);
                println!(
                    "  fred {name}: {} rows, last {}",
                    rows.len(),
                    rows.last().map(|r| r.0.as_str()).unwrap_or("-")
                );
            }
            Ok(_) => eprintln!("  fred {name}: empty response"),
            Err(e) => eprintln!("  fred {name}: {e}"),
        }
    }
}

/// GET a FRED series CSV and parse (date, value), skipping missing (".") rows.
pub fn fetch_series(id: &str) -> Result<Vec<(String, f64)>, String> {
    // cosd pins the observation start so we get full history, not FRED's default ~3y window.
    let url = format!("https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd=1990-01-01");
    let bytes = curl_get(&url)?;
    let text = String::from_utf8_lossy(&bytes);
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let mut it = line.split(',');
        let (Some(d), Some(v)) = (it.next(), it.next()) else { continue };
        if let Ok(val) = v.trim().parse::<f64>() {
            out.push((d.trim().to_string(), val));
        }
    }
    Ok(out)
}

/// Load a cached FRED series (date, value) from data/fred_<name>.csv.
pub fn load(name: &str) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    if let Ok(t) = std::fs::read_to_string(format!("data/fred_{name}.csv")) {
        for line in t.lines().skip(1) {
            let mut it = line.split(',');
            if let (Some(d), Some(v)) = (it.next(), it.next()) {
                if let Ok(val) = v.trim().parse::<f64>() {
                    out.push((d.trim().to_string(), val));
                }
            }
        }
    }
    out
}
