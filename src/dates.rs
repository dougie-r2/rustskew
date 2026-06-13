//! Minimal civil-date arithmetic (no external crates).
//! Based on Howard Hinnant's days_from_civil algorithm.

/// Days since 1970-01-01 for a (year, month, day) civil date.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Inverse of `days_from_civil`: returns (year, month, day).
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Parse "YYYY-MM-DD" (tolerates a trailing time component) into days since epoch.
pub fn parse_ymd(s: &str) -> Option<i64> {
    let datepart = s.trim().split(|c| c == ' ' || c == 'T').next()?;
    let mut it = datepart.split('-');
    let y: i64 = it.next()?.parse().ok()?;
    let m: i64 = it.next()?.parse().ok()?;
    let d: i64 = it.next()?.parse().ok()?;
    Some(days_from_civil(y, m, d))
}

/// Format days-since-epoch back to "YYYY-MM-DD".
pub fn fmt_ymd(days: i64) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// Day of week for days-since-epoch. 0 = Sunday .. 6 = Saturday.
pub fn weekday(days: i64) -> i64 {
    // 1970-01-01 was a Thursday (=4).
    ((days % 7) + 4 + 7) % 7
}

/// True for Saturday/Sunday.
pub fn is_weekend(days: i64) -> bool {
    let w = weekday(days);
    w == 0 || w == 6
}
