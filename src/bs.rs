//! Minimal Black-Scholes (r = q = 0, i.e. forward = spot) plus the normal CDF /
//! inverse-CDF needed for delta-targeted strike selection. Good enough for a
//! relative backtest where internal consistency matters more than absolute rates.

/// Standard normal CDF via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7).
pub fn norm_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327 * (-x * x / 2.0).exp();
    let p = d
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Inverse standard normal CDF (Acklam's rational approximation, |err| ~ 1e-9).
pub fn norm_inv(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Standard normal PDF.
pub fn norm_pdf(x: f64) -> f64 {
    0.3989422804014327 * (-0.5 * x * x).exp()
}

/// Black-Scholes gamma (r = q = 0): ∂²price/∂S².
pub fn bs_gamma(s: f64, k: f64, t: f64, sigma: f64) -> f64 {
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
    norm_pdf(d1) / (s * sigma * sqrt_t)
}

/// Black-Scholes vanna (r = q = 0): ∂²price/∂S∂σ = ∂delta/∂σ.
pub fn bs_vanna(s: f64, k: f64, t: f64, sigma: f64) -> f64 {
    if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    -norm_pdf(d1) * d2 / sigma
}

/// Black-Scholes price (r = q = 0). `is_call` selects call vs put.
pub fn bs_price(is_call: bool, s: f64, k: f64, t: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return intrinsic(is_call, s, k);
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    if is_call {
        s * norm_cdf(d1) - k * norm_cdf(d2)
    } else {
        k * norm_cdf(-d2) - s * norm_cdf(-d1)
    }
}

pub fn intrinsic(is_call: bool, s: f64, k: f64) -> f64 {
    if is_call {
        (s - k).max(0.0)
    } else {
        (k - s).max(0.0)
    }
}

/// Strike for a target absolute delta (r = q = 0).
/// `is_call`=true → OTM call with delta ≈ +target; false → OTM put, delta ≈ -target.
pub fn strike_for_delta(is_call: bool, s: f64, sigma: f64, t: f64, target_abs_delta: f64) -> f64 {
    if sigma <= 0.0 || t <= 0.0 {
        return s;
    }
    // call: N(d1) = target ;  put: N(d1) = 1 - target
    let d1 = if is_call {
        norm_inv(target_abs_delta)
    } else {
        norm_inv(1.0 - target_abs_delta)
    };
    let sqrt_t = t.sqrt();
    // ln(S/K) = d1*sigma*sqrt_t - 0.5*sigma^2*t  =>  K = S*exp(0.5 sigma^2 t - d1 sigma sqrt_t)
    s * (0.5 * sigma * sigma * t - d1 * sigma * sqrt_t).exp()
}
