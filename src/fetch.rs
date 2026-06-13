//! Network fetch helpers. We shell out to `curl` (proven-working in this
//! environment) instead of pulling in a TLS/HTTP crate. Responses are cached
//! to disk so backfills are resumable and we never re-hit the API for a day we
//! already have.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

const DOLT_API: &str =
    "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master";

/// Raw GET via curl. Returns stdout bytes.
pub fn curl_get(url: &str) -> Result<Vec<u8>, String> {
    let out = Command::new("curl")
        .args([
            "-s", "-m", "45", "--http1.1", "--retry", "3", "--retry-delay", "1", url,
        ])
        .output()
        .map_err(|e| format!("curl spawn failed: {e}"))?;
    if !out.status.success() {
        return Err(format!("curl exited with {:?}", out.status.code()));
    }
    Ok(out.stdout)
}

/// GET with a browser User-Agent (Yahoo requires one) over HTTP/1.1.
pub fn curl_get_ua(url: &str, ua: &str) -> Result<Vec<u8>, String> {
    let out = Command::new("curl")
        .args([
            "-s", "-m", "30", "--http1.1", "--retry", "2", "--retry-delay", "1", "-A", ua, url,
        ])
        .output()
        .map_err(|e| format!("curl spawn failed: {e}"))?;
    if !out.status.success() {
        return Err(format!("curl exited with {:?}", out.status.code()));
    }
    Ok(out.stdout)
}

/// Run a SQL query against the DoltHub options DB and return parsed JSON.
pub fn dolt_query(sql: &str) -> Result<Value, String> {
    let out = Command::new("curl")
        .args([
            "-s",
            "-m",
            "60",
            "--data-urlencode",
            &format!("q={sql}"),
            "-G",
            DOLT_API,
        ])
        .output()
        .map_err(|e| format!("curl spawn failed: {e}"))?;
    let v: Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("json parse failed: {e}; body={}", String::from_utf8_lossy(&out.stdout).chars().take(200).collect::<String>()))?;
    let status = v
        .get("query_execution_status")
        .and_then(|s| s.as_str())
        .unwrap_or("");
    if status != "Success" {
        let msg = v
            .get("query_execution_message")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown error");
        return Err(format!("dolt query error: {msg}"));
    }
    Ok(v)
}

/// Fetch one day's option-chain rows for a symbol, with on-disk caching.
/// Returns the `rows` JSON array. An empty array (no trading) is cached too,
/// so non-trading days aren't refetched.
pub fn dolt_day_rows(symbol: &str, date: &str, cache_root: &Path) -> Result<Vec<Value>, String> {
    let cache_dir = cache_root.join("dolt").join(symbol);
    let cache_file = cache_dir.join(format!("{date}.json"));

    if let Ok(bytes) = std::fs::read(&cache_file) {
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
            if let Some(arr) = v.as_array() {
                return Ok(arr.clone());
            }
        }
    }

    let sql = format!(
        "select `date`,expiration,strike,call_put,bid,ask,vol,delta \
         from option_chain where act_symbol='{symbol}' and `date`='{date}' \
         order by expiration,strike"
    );
    let v = dolt_query(&sql)?;
    let rows = v
        .get("rows")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();

    std::fs::create_dir_all(&cache_dir).map_err(|e| format!("mkdir {cache_dir:?}: {e}"))?;
    let _ = std::fs::write(&cache_file, serde_json::to_vec(&rows).unwrap_or_default());

    Ok(rows)
}

/// Default cache root: `<workdir>/data`.
pub fn default_cache_root() -> PathBuf {
    PathBuf::from("data")
}
