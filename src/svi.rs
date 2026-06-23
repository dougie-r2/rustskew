//! Gatheral "raw" SVI smile fit (per expiry), in log-moneyness k = ln(K/F).
//!
//! Total implied variance:  w(k) = a + b·( ρ·(k−m) + sqrt((k−m)² + σ²) )
//! where w = IV²·T.  Five params: a (level), b (wing slope ≥0), ρ (skew, |ρ|<1),
//! m (shift), σ (ATM curvature >0).
//!
//! Calibration uses the Zeliade quasi-explicit trick: for fixed (m, σ) the model
//! is LINEAR in (a, b·ρ, b), so we grid-search (m, σ) and solve a 3×3 linear
//! least-squares inside each step.

#[derive(Clone, Copy)]
pub struct Svi {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

impl Svi {
    pub fn w(&self, k: f64) -> f64 {
        let km = k - self.m;
        self.a + self.b * (self.rho * km + (km * km + self.sigma * self.sigma).sqrt())
    }
}

/// Solve a symmetric 3×3 system via Cramer's rule. Returns None if near-singular.
fn solve3(
    a11: f64, a12: f64, a13: f64, a22: f64, a23: f64, a33: f64,
    b1: f64, b2: f64, b3: f64,
) -> Option<(f64, f64, f64)> {
    let det = a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a23 * a13)
        + a13 * (a12 * a23 - a22 * a13);
    if det.abs() < 1e-14 {
        return None;
    }
    let d1 = b1 * (a22 * a33 - a23 * a23) - a12 * (b2 * a33 - a23 * b3) + a13 * (b2 * a23 - a22 * b3);
    let d2 = a11 * (b2 * a33 - a23 * b3) - b1 * (a12 * a33 - a23 * a13) + a13 * (a12 * b3 - b2 * a13);
    let d3 = a11 * (a22 * b3 - b2 * a23) - a12 * (a12 * b3 - b2 * a13) + b1 * (a12 * a23 - a22 * a13);
    Some((d1 / det, d2 / det, d3 / det))
}

/// Fit SVI to (k, w) = (log-moneyness, total variance) points.
pub fn fit(points: &[(f64, f64)]) -> Option<Svi> {
    if points.len() < 4 {
        return None;
    }
    let kmin = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let kmax = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    if !(kmax > kmin) {
        return None;
    }
    let (nm, ns) = (28usize, 20usize);
    let mut best: Option<(f64, Svi)> = None;
    for im in 0..nm {
        let m = kmin + (kmax - kmin) * (im as f64) / ((nm - 1) as f64);
        for is in 0..ns {
            let sigma = 0.005 + (0.6 - 0.005) * (is as f64) / ((ns - 1) as f64);
            // basis [1, (k-m), sqrt((k-m)^2+sigma^2)] -> coeffs (a, p=b*rho, q=b)
            let (mut s11, mut s12, mut s13, mut s22, mut s23, mut s33) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut r1, mut r2, mut r3) = (0.0, 0.0, 0.0);
            for &(k, w) in points {
                let x1 = k - m;
                let x2 = (x1 * x1 + sigma * sigma).sqrt();
                s11 += 1.0;
                s12 += x1;
                s13 += x2;
                s22 += x1 * x1;
                s23 += x1 * x2;
                s33 += x2 * x2;
                r1 += w;
                r2 += w * x1;
                r3 += w * x2;
            }
            let Some((a, p, q)) = solve3(s11, s12, s13, s22, s23, s33, r1, r2, r3) else { continue };
            if q <= 0.0 {
                continue; // b must be >= 0
            }
            let rho = p / q;
            if rho.abs() >= 0.999 {
                continue;
            }
            // minimum total variance must be non-negative
            if a + q * sigma * (1.0 - rho * rho).sqrt() < 0.0 {
                continue;
            }
            let svi = Svi { a, b: q, rho, m, sigma };
            let sse: f64 = points.iter().map(|&(k, w)| (svi.w(k) - w).powi(2)).sum();
            if best.as_ref().map(|(s, _)| sse < *s).unwrap_or(true) {
                best = Some((sse, svi));
            }
        }
    }
    best.map(|(_, s)| s)
}
