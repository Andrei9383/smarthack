//! K4 fused feature build. Python reference: `sentinel.model.features.pv_arrays` / `load_arrays` (numpy, float32),
//! wrapped by `sentinel.kernels.features`.
//!
//! numpy builds each feature as a separate [B, ...] temporary (~40 passes over the batch); here every window b is
//! one pass over its own 14-day history and 96 target steps, windows run in parallel. Sums replicate numpy's
//! float32 reductions (pairwise over a contiguous row, sequential across days), max/min/clip propagate NaN like
//! numpy, so everything except the trig channels (cos/sin, atan2) is bit-identical to the reference.

use rayon::prelude::*;

pub const H: usize = 96;
const HD: usize = 14; // history days
const RD: usize = 7; // reference days
const D2R: f32 = std::f32::consts::PI / 180.0;
pub const PV_SEQ: usize = 17;
pub const LOAD_SEQ: usize = 14;
pub const REFH: usize = RD * 24;
pub const PV_SUM: usize = 5;
pub const LOAD_SUM: usize = 4;

/// numpy's float32 `add.reduce` over a contiguous run (pairwise_sum), bit for bit.
fn psum(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(-0.0f32, |s, &x| s + x);
    }
    if n <= 128 {
        let mut r = [0f32; 8];
        r.copy_from_slice(&a[..8]);
        let m = n - n % 8;
        for c in a[8..m].chunks_exact(8) {
            for j in 0..8 {
                r[j] += c[j];
            }
        }
        let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        return a[m..].iter().fold(res, |s, &x| s + x);
    }
    let n2 = n / 2 - (n / 2) % 8;
    psum(&a[..n2]) + psum(&a[n2..])
}

#[inline]
fn nmax(a: f32, b: f32) -> f32 {
    if a.is_nan() || a > b { a } else { b }
}
#[inline]
fn nmin(a: f32, b: f32) -> f32 {
    if a.is_nan() || a < b { a } else { b }
}
#[inline]
fn clip(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo } else if x > hi { hi } else { x }
}
/// np.clip(np.nan_to_num(x), -3, 3) on float64, then float32.
#[inline]
fn summ_val(x: f64) -> f32 {
    if x.is_nan() { 0.0 } else { x.clamp(-3.0, 3.0) as f32 }
}
fn rmax(a: &[f32]) -> f32 {
    a[1..].iter().fold(a[0], |m, &x| nmax(x, m))
}
fn rmin(a: &[f32]) -> f32 {
    a[1..].iter().fold(a[0], |m, &x| nmin(x, m))
}

/// 7 reference days (already scaled) -> per step mean / max into `mean`, `max`; hourly means into `refh`.
fn ref_stats(r: &[[f32; H]; RD], mean: &mut [f32; H], max: &mut [f32; H], refh: &mut [f32]) {
    for t in 0..H {
        let (mut s, mut m) = (r[0][t], r[0][t]);
        for d in r.iter().skip(1) {
            s += d[t];
            m = nmax(d[t], m);
        }
        mean[t] = s / RD as f32;
        max[t] = m;
    }
    for (j, o) in refh.iter_mut().enumerate() {
        let (d, t) = (j / 24, (j % 24) * 4);
        *o = psum(&r[d][t..t + 4]) / 4.0;
    }
}

/// One PV window. hist = the last 14*96 steps of power (kW); pvclear, pvlib: [H]; wx: [H*7]; geom: [H*3].
#[allow(clippy::too_many_arguments)]
fn pv_one(wx: &[f32], geom: &[f32], clear: &[f32], pvlib: &[f32], hist: &[f32], k: f32, seq: &mut [f32],
          refh: &mut [f32], summ: &mut [f32], night: &mut [u8]) {
    let mut r = [[0f32; H]; RD];
    for (d, rd) in r.iter_mut().enumerate() {
        for (o, &x) in rd.iter_mut().zip(&hist[(HD - RD + d) * H..]) {
            *o = x / k;
        }
    }
    let (mut mean, mut max) = ([0f32; H], [0f32; H]);
    ref_stats(&r, &mut mean, &mut max, refh);
    for t in 0..H {
        let (w, g) = (&wx[t * 7..t * 7 + 7], &geom[t * 3..t * 3 + 3]);
        let (zen, az) = (g[0] * D2R, g[1] * D2R);
        let (cz, cs) = (zen.cos(), g[2] / 1000.0);
        let pvc = clear[t] / k;
        let rcr = clip(max[t] / nmax(pvc, 0.02), 0.0, 2.0) * ((pvc > 0.01) as u8 as f32);
        seq[t * PV_SEQ..(t + 1) * PV_SEQ].copy_from_slice(&[
            w[0] / 1000.0, w[1] / 1000.0, w[2] / 500.0, w[3] / 20.0, w[4] / 100.0, w[5] / 10.0, w[6] / 10.0,
            cz, az.sin(), az.cos(), cs, pvlib[t] / k, pvc, mean[t], max[t], r[RD - 1][t], rcr,
        ]);
        night[t] = (cz <= 0.0 && cs <= 0.0) as u8;
    }
    pv_summ(clear, geom, hist, k, summ, PV_SUM);
}

/// pv_summary: [14, 5] daily stats of the last 14 days of power vs today's clear-sky profile, written to the
/// first 5 columns of rows of width `w`.
fn pv_summ(clear: &[f32], geom: &[f32], hist: &[f32], k: f32, summ: &mut [f32], w: usize) {
    let mut am = [0f32; H];
    for (t, a) in am.iter_mut().enumerate() {
        *a = (geom[t * 3 + 1] < 180.0) as u8 as f32; // sun in the east = morning
    }
    let e_clear = psum(clear) + 1e-3;
    let cmax = rmax(clear) + 1e-3;
    let mut tmp = [0f32; H];
    for (o, (&c, &a)) in tmp.iter_mut().zip(clear.iter().zip(&am)) {
        *o = c * a;
    }
    let am_clear = psum(&tmp) / e_clear;
    let thr = 0.1f32 * k;
    let mask: Vec<bool> = clear.iter().map(|&c| c > thr).collect();
    let nmask = mask.iter().filter(|&&m| m).count() as f64 + 1.0;
    for d in 0..HD {
        let day = &hist[d * H..(d + 1) * H];
        let sum = psum(day);
        for (o, (&x, &a)) in tmp.iter_mut().zip(day.iter().zip(&am)) {
            *o = x * a;
        }
        let am_share = psum(&tmp) / (sum + 1e-3);
        let (mut zero, mut froz) = (0u32, 0u32);
        for t in 0..H {
            let x = day[t];
            let prev = if d == 0 && t == 0 { f32::NAN } else { hist[d * H + t - 1] };
            if mask[t] {
                zero += (x < 0.01) as u32;
                froz += (x == prev && x > 0.02) as u32;
            }
        }
        summ[d * w..d * w + PV_SUM].copy_from_slice(&[
            summ_val((sum / e_clear) as f64), summ_val((rmax(day) / cmax) as f64),
            summ_val((am_share - am_clear) as f64), summ_val(zero as f64 / nmask), summ_val(froz as f64 / nmask),
        ]);
    }
}

/// np.remainder(x, 24) for float32.
#[inline]
fn rem24(x: f32) -> f32 {
    let m = x % 24.0;
    if m != 0.0 && m < 0.0 { m + 24.0 } else if m == 0.0 { 0.0 } else { m }
}

/// One load window. hist = exactly 14*96 steps of load (kW); wx: [H*7]; cal: [H*8]. Returns the scale.
fn load_one(wx: &[f32], cal: &[f32], hist: &[f32], seq: &mut [f32], refh: &mut [f32], summ: &mut [f32]) -> f32 {
    let s = psum(hist) / (HD * H) as f32 + 0.05;
    let mut r = [[0f32; H]; RD];
    for (d, rd) in r.iter_mut().enumerate() {
        for (o, &x) in rd.iter_mut().zip(&hist[(HD - RD + d) * H..]) {
            *o = x / s;
        }
    }
    let (mut mean, mut max) = ([0f32; H], [0f32; H]);
    ref_stats(&r, &mut mean, &mut max, refh);
    let mut eve = [0f32; H];
    for t in 0..H {
        let (w, c) = (&wx[t * 7..t * 7 + 7], &cal[t * 8..t * 8 + 8]);
        let o = &mut seq[t * LOAD_SEQ..(t + 1) * LOAD_SEQ];
        o[0] = w[3] / 20.0;
        o[1] = w[0] / 1000.0;
        o[2..10].copy_from_slice(c);
        o[10..].copy_from_slice(&[r[RD - 1][t], r[0][t], mean[t], max[t]]);
        let hour = rem24(c[0].atan2(c[1]) / (2.0 * std::f64::consts::PI) as f32 * 24.0);
        eve[t] = (hour >= 17.0 && hour < 23.0) as u8 as f32;
    }
    let mut tmp = [0f32; H];
    for d in 0..HD {
        let day = &hist[d * H..(d + 1) * H];
        let sum = psum(day);
        for (o, (&x, &e)) in tmp.iter_mut().zip(day.iter().zip(&eve)) {
            *o = x * e;
        }
        summ[d * LOAD_SUM..(d + 1) * LOAD_SUM].copy_from_slice(&[
            sum / H as f32 / s, rmax(day) / s, rmin(day) / s, psum(&tmp) / (sum + 1e-3),
        ]);
    }
    s
}

/// Batched PV features; hist rows have length hh >= 14*96 (the last 14 days are used).
#[allow(clippy::too_many_arguments)]
pub fn pv_batch(wx: &[f32], geom: &[f32], clear: &[f32], pvlib: &[f32], hist: &[f32], hh: usize, kwp: &[f32],
                seq: &mut [f32], refh: &mut [f32], summ: &mut [f32], night: &mut [u8]) {
    seq.par_chunks_mut(H * PV_SEQ)
        .zip(refh.par_chunks_mut(REFH))
        .zip(summ.par_chunks_mut(HD * PV_SUM))
        .zip(night.par_chunks_mut(H))
        .enumerate()
        .with_min_len(4)
        .for_each(|(b, (((s, r), m), n))| {
            let h = &hist[b * hh + hh - HD * H..(b + 1) * hh];
            pv_one(&wx[b * H * 7..(b + 1) * H * 7], &geom[b * H * 3..(b + 1) * H * 3], &clear[b * H..(b + 1) * H],
                   &pvlib[b * H..(b + 1) * H], h, kwp[b], s, r, m, n);
        });
}

/// Batched load features; hist rows have length exactly 14*96 (the scale is the mean over all of it).
pub fn load_batch(wx: &[f32], cal: &[f32], hist: &[f32], seq: &mut [f32], refh: &mut [f32], summ: &mut [f32],
                  scale: &mut [f32]) {
    seq.par_chunks_mut(H * LOAD_SEQ)
        .zip(refh.par_chunks_mut(REFH))
        .zip(summ.par_chunks_mut(HD * LOAD_SUM))
        .zip(scale.par_iter_mut())
        .enumerate()
        .with_min_len(4)
        .for_each(|(b, (((s, r), m), sc))| {
            *sc = load_one(&wx[b * H * 7..(b + 1) * H * 7], &cal[b * H * 8..(b + 1) * H * 8],
                           &hist[b * HD * H..(b + 1) * HD * H], s, r, m);
        });
}

pub const SITE_SEQ: usize = 18;
pub const SITE_SUM: usize = 8;

/// M4 SiteHealth inputs for a batch (reference: `sentinel.model.features.site_arrays`). Per-step arrays are
/// [b, H] (m1, geom: [b, H*3]); hist_pv rows have hh steps and hist_volt/status/soc/batt rows ht steps (the last
/// 14 days are used), hist_load rows hl steps (all used); per site: kwp, bkw, bkwh (battery kW / kWh), has_batt.
pub struct Site<'a> {
    pub pv: &'a [f32], pub m1: &'a [f32], pub m2: &'a [f32], pub geom: &'a [f32], pub pvlib: &'a [f32],
    pub clear: &'a [f32], pub volt: &'a [f32], pub status: &'a [i8], pub soc: &'a [f32], pub batt: &'a [f32],
    pub load: &'a [f32], pub valid: &'a [f32], pub hist_pv: &'a [f32], pub hist_load: &'a [f32],
    pub hist_volt: &'a [f32], pub hist_status: &'a [i8], pub hist_soc: &'a [f32], pub hist_batt: &'a [f32],
    pub hh: usize, pub hl: usize, pub ht: usize,
    pub kwp: &'a [f32], pub bkw: &'a [f32], pub bkwh: &'a [f32], pub has_batt: &'a [f32],
}

/// np.nan_to_num for float32: NaN -> 0, +-inf -> +-f32::MAX.
#[inline]
fn n2n(x: f32) -> f32 {
    if x.is_nan() { 0.0 } else { x.clamp(f32::MIN, f32::MAX) }
}

fn site_one(x: &Site, b: usize, seq: &mut [f32], summ: &mut [f32]) {
    let r = b * H..(b + 1) * H;
    let (pv, m2, pvlib, volt, st, soc, batt, load, valid) = (&x.pv[r.clone()], &x.m2[r.clone()],
        &x.pvlib[r.clone()], &x.volt[r.clone()], &x.status[r.clone()], &x.soc[r.clone()], &x.batt[r.clone()],
        &x.load[r.clone()], &x.valid[r.clone()]);
    let (m1, geom) = (&x.m1[b * H * 3..(b + 1) * H * 3], &x.geom[b * H * 3..(b + 1) * H * 3]);
    let hpv = &x.hist_pv[(b + 1) * x.hh - HD * H..(b + 1) * x.hh];
    let (k, bkw) = (x.kwp[b], nmax(x.bkw[b], 1.0));
    let k05 = 0.05f32 * k;
    let ls = psum(&x.hist_load[b * x.hl..(b + 1) * x.hl]) / x.hl as f32 + 0.05;
    let mut prev = hpv[HD * H - 1];
    for t in 0..H {
        let (p, q10, q50, q90, s) = (pv[t], m1[t * 3], m1[t * 3 + 1], m1[t * 3 + 2], st[t]);
        let o = &mut seq[t * SITE_SEQ..(t + 1) * SITE_SEQ];
        o.copy_from_slice(&[
            p / k, q10 / k, q50 / k, q90 / k, (p - q50) / k, clip((p - q50) / ((q90 - q10) + k05), -5.0, 5.0),
            pvlib[t] / k, (geom[t * 3] * D2R).cos(), (volt[t] - 230.0) / 10.0,
            (s == 1) as u8 as f32, (s == 2) as u8 as f32, (s == 3) as u8 as f32,
            (p == prev && p > 0.02) as u8 as f32, soc[t], batt[t] / bkw,
            clip((load[t] - m2[t]) / ls, -5.0, 5.0), clip(p / (q50 + k05), 0.0, 3.0), valid[t],
        ]);
        o.iter_mut().for_each(|v| *v = n2n(*v));
        prev = p;
    }
    pv_summ(&x.clear[r], geom, hpv, k, summ, SITE_SUM);
    let tr = (b + 1) * x.ht - HD * H..(b + 1) * x.ht;
    let (hv, hs, hso, hb) = (&x.hist_volt[tr.clone()], &x.hist_status[tr.clone()], &x.hist_soc[tr.clone()],
                             &x.hist_batt[tr]);
    let bkwh = nmax(x.bkwh[b], 1e-3);
    let z = |v: f32| if v.is_nan() { 0.0 } else { v };
    let mut tmp = [0f32; H];
    for d in 0..HD {
        let day = d * H..(d + 1) * H;
        let trips = hs[day.clone()].iter().filter(|&&s| s == 3).count();
        for (o, &v) in tmp.iter_mut().zip(&hb[day.clone()]) {
            *o = v.abs();
        }
        let thr = psum(&tmp) * 0.25 / bkwh; // full-cycle equivalents
        let swing = if x.has_batt[b] > 0.0 {
            (rmax(&hso[day.clone()]) - rmin(&hso[day.clone()])) / (0.5 * thr + 0.05)
        } else { 1.0 };
        summ[d * SITE_SUM + PV_SUM..(d + 1) * SITE_SUM].copy_from_slice(&[
            z((rmax(&hv[day]) - 230.0) / 10.0), (trips as f64 / H as f64 * 10.0) as f32, z(clip(swing, 0.0, 5.0)),
        ]);
    }
}

/// Batched M4 site features: seq [b, H*18], summ [b, 14*8].
pub fn site_batch(x: &Site, seq: &mut [f32], summ: &mut [f32]) {
    seq.par_chunks_mut(H * SITE_SEQ)
        .zip(summ.par_chunks_mut(HD * SITE_SUM))
        .enumerate()
        .with_min_len(4)
        .for_each(|(b, (s, m))| site_one(x, b, s, m));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_small() {
        assert_eq!(psum(&[1.0, 2.0, 3.0]), 6.0);
        let a: Vec<f32> = (0..1344).map(|i| i as f32).collect();
        assert_eq!(psum(&a), (1343 * 1344 / 2) as f32);
        assert_eq!(rem24(-1.0), 23.0);
        assert!(nmax(f32::NAN, 1.0).is_nan() && nmax(1.0, f32::NAN).is_nan());
    }
}
