//! Causal NaN-aware rolling mean along axis 0. Python reference:
//! `wattsentry.kernels.rolling.rolling_mean_py` (source of truth).
//!
//! x is row-major [t, m]. With prefix sums S (finite values, NaN/inf -> +0.0) and N (finite counts), S[0] = N[0] = 0:
//!   out[t] = (S[t+1] - S[lo]) / (N[t+1] - N[lo]) if the count > 0 else 0, lo = max(t + 1 - w, 0)
//! Same sequential additions and subtractions as numpy's cumsum differences, so the result is bit-identical.
//! Only the last min(w, t) + 1 prefix rows are kept (ring buffer). Serial: one pass is memory-bound.

pub fn rolling_mean_2d(x: &[f64], t: usize, m: usize, w: usize) -> Vec<f64> {
    if t == 0 || m == 0 {
        return vec![0.0f64; t * m];
    }
    let mut out = Vec::with_capacity(t * m);
    let r = w.min(t) + 1;
    let (mut rs, mut rn) = (vec![0.0f64; r * m], vec![0.0f64; r * m]);
    let (mut cs, mut cn) = (vec![0.0f64; m], vec![0.0f64; m]);
    for (i, xr) in x.chunks_exact(m).enumerate() {
        let idx = i + 1;
        let (lo, cur) = (idx.saturating_sub(w) % r * m, idx % r * m);
        for ((c, k), &v) in cs.iter_mut().zip(cn.iter_mut()).zip(xr) {
            let ok = v.is_finite();
            *c += if ok { v } else { 0.0 };
            *k += ok as u8 as f64;
        }
        rs[cur..cur + m].copy_from_slice(&cs);
        rn[cur..cur + m].copy_from_slice(&cn);
        let (ls, ln) = (&rs[lo..lo + m], &rn[lo..lo + m]);
        out.extend(cs.iter().zip(&cn).zip(ls).zip(ln).map(|(((c, k), a), b)| {
            let n = k - b;
            if n > 0.0 { (c - a) / n } else { 0.0 }
        }));
    }
    out
}
