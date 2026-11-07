//! K7 self-consumption battery rule over time (#135). Python reference (source of truth):
//! `sentinel.sim.fleet.battery.simulate_self_consumption` / `apply_power`. Every floating-point operation is
//! written in numpy's order, and min/max/clip follow numpy's semantics (NaN propagates; on ties, including
//! -0.0 vs +0.0, `maximum(a, b)` / `minimum(a, b)` return `b`), so the f32 outputs are bit-identical.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub struct Params<'a> {
    pub cap: &'a [f64],
    pub pmax_kw: &'a [f64],
    pub eta_c: &'a [f64],
    pub eta_d: &'a [f64],
    pub e_lo: &'a [f64],
    pub e_hi: &'a [f64],
    pub inv_cap: &'a [f64],
}

/// np.maximum(a, b)
#[inline(always)]
fn npmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || a > b { a } else { b } // b NaN: a > b is false -> b
}

/// np.minimum(a, b)
#[inline(always)]
fn npmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || a < b { a } else { b }
}

/// np.clip(x, lo, hi) for floats: min(max(x, lo), hi), keeping a NaN x.
#[inline(always)]
fn npclip(x: f64, lo: f64, hi: f64) -> f64 {
    let t = if x.is_nan() || x > lo { x } else { lo };
    if t.is_nan() || t < hi { t } else { hi }
}

/// One site's year. `pv`, `load` are the site's f32 rows; writes `batt` and `soc`; returns the end energy (kWh).
#[inline]
#[allow(clippy::too_many_arguments)]
fn site(pv: &[f32], load: &[f32], i: usize, p: &Params, e0: f64, dt: f64, batt: &mut [f32], soc: &mut [f32]) -> f64 {
    let inv_cap = p.inv_cap[i];
    let mut e = e0;
    if p.cap[i] <= 0.0 || p.cap[i].is_nan() {
        // no battery: applied = 0, energy unchanged
        let s = (e * inv_cap) as f32;
        batt.fill(0.0);
        soc.fill(s);
        return e;
    }
    let (eta_c, eta_d, lo, hi) = (p.eta_c[i], p.eta_d[i], p.e_lo[i], p.e_hi[i]);
    let p_max_w = p.pmax_kw[i] * 1000.0;
    let neg_p_max_w = -p_max_w;
    let eta_c_dt = eta_c * dt;
    for t in 0..pv.len() {
        let surplus = (pv[t] - load[t]) as f64; // float32 subtraction, as numpy does on two float32 arrays
        let req = npclip(surplus, neg_p_max_w, p_max_w);
        let room_w = npmax(0.0, (hi - e) / eta_c_dt) * 1000.0;
        let avail_w = npmax(0.0, (e - lo) * eta_d / dt) * 1000.0;
        let applied = if req >= 0.0 { npmin(req, room_w) } else { npmax(req, -avail_w) };
        let e_next = if applied >= 0.0 { e + eta_c * applied / 1000.0 * dt } else { e + applied / 1000.0 * dt / eta_d };
        e = npmin(hi, npmax(lo, e_next));
        batt[t] = applied as f32;
        soc[t] = (e * inv_cap) as f32;
    }
    e
}

/// `pv`, `load`, `batt`, `soc` are row-major (n sites × t steps); sites run in parallel. Returns e_end (n,).
#[allow(clippy::too_many_arguments)]
pub fn run(pv: &[f32], load: &[f32], n: usize, t: usize, p: &Params, e0: &[f64], dt: f64,
           batt: &mut [f32], soc: &mut [f32]) -> Vec<f64> {
    let mut e_end = e0.to_vec();
    if n == 0 || t == 0 {
        return e_end;
    }
    pv.par_chunks(t)
        .zip(load.par_chunks(t))
        .zip(batt.par_chunks_mut(t))
        .zip(soc.par_chunks_mut(t))
        .zip(e_end.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((((pv, ld), b), s), e))| *e = site(pv, ld, i, p, e0[i], dt, b, s));
    e_end
}

/// self_consumption(pv f32[n, t], load f32[n, t], cap, pmax_kw, eta_c, eta_d, e_lo, e_hi, inv_cap, e0: f64[n], dt,
///                  batt f32[n, t] (out), soc f32[n, t] (out)) -> e_end f64[n]
///
/// K7: the fleet simulator's self-consumption battery rule; see `selfcons.rs`. The wrapper
/// `sentinel.kernels.selfcons.self_consumption_rust` allocates the outputs.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn self_consumption<'py>(
    py: Python<'py>,
    pv: PyReadonlyArray2<'py, f32>,
    load: PyReadonlyArray2<'py, f32>,
    cap: PyReadonlyArray1<'py, f64>,
    pmax_kw: PyReadonlyArray1<'py, f64>,
    eta_c: PyReadonlyArray1<'py, f64>,
    eta_d: PyReadonlyArray1<'py, f64>,
    e_lo: PyReadonlyArray1<'py, f64>,
    e_hi: PyReadonlyArray1<'py, f64>,
    inv_cap: PyReadonlyArray1<'py, f64>,
    e0: PyReadonlyArray1<'py, f64>,
    dt: f64,
    mut batt: PyReadwriteArray2<'py, f32>,
    mut soc: PyReadwriteArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n, t) = (pv.shape()[0], pv.shape()[1]);
    if load.shape() != [n, t] || batt.shape() != [n, t] || soc.shape() != [n, t] {
        return Err(PyValueError::new_err("pv, load, batt, soc must all have shape (n, t)"));
    }
    let contig = |e| PyValueError::new_err(format!("{e}: arrays must be C-contiguous"));
    let (pv, load) = (pv.as_slice().map_err(contig)?, load.as_slice().map_err(contig)?);
    let sl = [&cap, &pmax_kw, &eta_c, &eta_d, &e_lo, &e_hi, &inv_cap, &e0].map(|a| a.as_slice());
    let sl: Vec<&[f64]> = sl.into_iter().collect::<Result<_, _>>().map_err(contig)?;
    if sl.iter().any(|v| v.len() != n) {
        return Err(PyValueError::new_err("battery arrays and e0 must have shape (n,)"));
    }
    let p = Params { cap: sl[0], pmax_kw: sl[1], eta_c: sl[2], eta_d: sl[3], e_lo: sl[4], e_hi: sl[5],
                     inv_cap: sl[6] };
    let (b, s) = (batt.as_slice_mut().map_err(contig)?, soc.as_slice_mut().map_err(contig)?);
    let e_end = py.detach(|| run(pv, load, n, t, &p, sl[7], dt, b, s));
    Ok(e_end.into_pyarray(py))
}

