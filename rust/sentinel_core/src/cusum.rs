//! Two-sided CUSUM (K3). Python reference: `sentinel.kernels.cusum.cusum_two_sided_py` (source of truth).
//!
//! Per series, for t = 0..n:
//!   if x[t] is not finite (NaN / ±inf = gap): s_pos = s_neg = 0, alarm = false
//!   else: s_pos = min(max(s_pos + x[t] - k, 0), cap); s_neg = min(max(s_neg - x[t] - k, 0), cap);
//!         alarm = s_pos > h || s_neg > h
//! Sums start at 0 and are not reset after an alarm. `cap = None` means no cap (min with +inf is exact).

use rayon::prelude::*;

/// One series. `x`, `sp`, `sn`, `al` all have the same length.
#[inline]
pub fn cusum_row(x: &[f64], k: f64, h: f64, cap: f64, sp: &mut [f64], sn: &mut [f64], al: &mut [bool]) {
    let (mut cp, mut cn) = (0.0f64, 0.0f64);
    for t in 0..x.len() {
        let v = x[t];
        if v.is_finite() {
            // Same operation order as numpy: (s + x) - k, then max with 0.
            cp = (cp + v - k).max(0.0).min(cap);
            cn = (cn - v - k).max(0.0).min(cap);
        } else {
            cp = 0.0;
            cn = 0.0;
        }
        sp[t] = cp;
        sn[t] = cn;
        al[t] = cp > h || cn > h;
    }
}

/// `x` is row-major (m series × n steps). Rows are processed in parallel. `cap` = +inf for no cap.
pub fn cusum_2d(x: &[f64], m: usize, n: usize, k: f64, h: f64, cap: f64) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut sp = vec![0.0f64; m * n];
    let mut sn = vec![0.0f64; m * n];
    let mut al = vec![false; m * n];
    if n == 0 || m == 0 {
        return (sp, sn, al);
    }
    x.par_chunks(n)
        .zip(sp.par_chunks_mut(n))
        .zip(sn.par_chunks_mut(n))
        .zip(al.par_chunks_mut(n))
        .for_each(|(((xr, spr), snr), alr)| cusum_row(xr, k, h, cap, spr, snr, alr));
    (sp, sn, al)
}
