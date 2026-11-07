//! K1 battery rollout. Python reference: `sentinel.opt.commit.rollout` (source of truth). Floating-point
//! operations are written in numpy's order so the results are bit-identical.

use rayon::prelude::*;

pub struct Params<'a> {
    pub cap: &'a [f64],
    pub pmax: &'a [f64],
    pub eta_c: &'a [f64],
    pub eta_d: &'a [f64],
    pub e_lo: &'a [f64],
    pub e_hi: &'a [f64],
}

#[inline]
fn row(req: &[f64], soc0: f64, i: usize, p: &Params, dt: f64, soc: &mut [f64], batt: &mut [f64]) {
    let (cap, pmax, eta_c, eta_d, e_lo, e_hi) = (p.cap[i], p.pmax[i], p.eta_c[i], p.eta_d[i], p.e_lo[i], p.e_hi[i]);
    let inv_cap = if cap > 0.0 { 1.0 / cap } else { 0.0 };
    let mut e = soc0 * cap;
    for t in 0..req.len() {
        // np.clip(a, lo, hi) == minimum(maximum(a, lo), hi)
        let mut q = req[t].max(-pmax).min(pmax);
        let room_c = ((e_hi - e) / (eta_c * dt)).max(0.0);
        let room_d = ((e - e_lo) * eta_d / dt).max(0.0);
        q = if q >= 0.0 { q.min(room_c) } else { q.max(-room_d) };
        let e_next = e + if q >= 0.0 { eta_c * q * dt } else { q * dt / eta_d };
        e = e_next.max(e_lo.min(e)).min(e_hi);
        batt[t] = q;
        soc[t] = e * inv_cap;
    }
}

/// `req` is row-major (n scenarios × s sites × t steps), `soc0` (n × s); rows run in parallel.
pub fn rollout_3d(req: &[f64], soc0: &[f64], n: usize, s: usize, t: usize, p: &Params, dt: f64)
    -> (Vec<f64>, Vec<f64>) {
    let mut soc = vec![0.0f64; n * s * t];
    let mut batt = vec![0.0f64; n * s * t];
    if n * s == 0 || t == 0 {
        return (soc, batt);
    }
    req.par_chunks(t)
        .zip(soc.par_chunks_mut(t))
        .zip(batt.par_chunks_mut(t))
        .enumerate()
        .for_each(|(k, ((r, so), ba))| row(r, soc0[k], k % s, p, dt, so, ba));
    (soc, batt)
}
