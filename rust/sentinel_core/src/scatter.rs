//! Telemetry scatter into [sensor, step, turbine] bin means (K5). Python reference:
//! `sentinel.kernels.scatter.scatter_telemetry_py` (source of truth).
//!
//! Codes overwrite err[t, j] in row order (last wins). Then, in parallel over sensors k, one pass in row order adds
//! the finite non-code readings v * scale[k, j] of sensor k to sum[t, j] (f64) and 1 to cnt: the same order as
//! numpy's bincount, so the means sum / cnt (f64, then f32) are bit-identical. Bins with err != 0 are NaN.

use rayon::prelude::*;

#[inline]
fn bin(a: i64, j: i64, t: usize, n: usize) -> Option<usize> {
    (a >= 0 && (a as usize) < t && j >= 0 && (j as usize) < n).then(|| a as usize * n + j as usize)
}

pub fn scatter(ti: &[i64], tj: &[i64], sk: &[i64], v: &[f64], code: &[bool], scale: &[f64],
               s: usize, t: usize, n: usize) -> (Vec<f32>, Vec<i32>) {
    let tn = t * n;
    let mut err = vec![0i32; tn];
    for i in 0..v.len() {
        if let (true, Some(b)) = (code[i], bin(ti[i], tj[i], t, n)) {
            // numpy's f64 -> i64 -> i32 cast: NaN / inf / out of i64 range give i64::MIN, i.e. 0 after the wrap
            let x = v[i];
            err[b] = if x.abs() < 9.223_372_036_854_775_808e18 { x as i64 as i32 } else { 0 };
        }
    }
    let mut out = vec![0f32; s * tn];
    out.par_chunks_mut(tn.max(1)).enumerate().for_each(|(k, o)| {
        let (mut sum, mut cnt) = (vec![0.0f64; tn], vec![0u32; tn]);
        for i in 0..v.len() {
            let x = v[i];
            if sk[i] != k as i64 || code[i] || !x.is_finite() {
                continue;
            }
            if let Some(b) = bin(ti[i], tj[i], t, n) {
                sum[b] += x * scale[k * n + tj[i] as usize];
                cnt[b] += 1;
            }
        }
        for (((o, &m), &c), &e) in o.iter_mut().zip(&sum).zip(&cnt).zip(&err) {
            *o = if c > 0 && e == 0 { (m / c as f64) as f32 } else { f32::NAN };
        }
    });
    (out, err)
}
