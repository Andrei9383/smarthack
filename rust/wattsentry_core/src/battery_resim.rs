//! K7 battery re-simulation for the PV fault injector (#134). Python reference:
//! `wattsentry.kernels.battery_resim.battery_resim_py` (source of truth; the loop from
//! `wattsentry.sim.faults.inject._resimulate_battery` + `wattsentry.sim.fleet.battery.apply_power`). Float operations
//! are written in numpy's order, with numpy's NaN semantics, so results are bit-identical.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// battery_resim(surplus, soc0, batt0, cf, pf: f64[m, t], start i64[m], e0, cap, pmax, eta_c, eta_d, soc_min,
///   soc_max: f64[m], t0, dt) -> (batt f64[m, t], soc f64[m, t])
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn battery_resim<'py>(
    py: Python<'py>,
    surplus: PyReadonlyArray2<'py, f64>, soc0: PyReadonlyArray2<'py, f64>, batt0: PyReadonlyArray2<'py, f64>,
    cf: PyReadonlyArray2<'py, f64>, pf: PyReadonlyArray2<'py, f64>, start: PyReadonlyArray1<'py, i64>,
    e0: PyReadonlyArray1<'py, f64>, cap: PyReadonlyArray1<'py, f64>, pmax: PyReadonlyArray1<'py, f64>,
    eta_c: PyReadonlyArray1<'py, f64>, eta_d: PyReadonlyArray1<'py, f64>, soc_min: PyReadonlyArray1<'py, f64>,
    soc_max: PyReadonlyArray1<'py, f64>, t0: usize, dt: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (m, t) = (surplus.shape()[0], surplus.shape()[1]);
    let contig = |e| PyValueError::new_err(format!("{e}: inputs must be C-contiguous"));
    let f2 = [&surplus, &soc0, &batt0, &cf, &pf].map(|a| a.as_slice());
    let f2: Vec<&[f64]> = f2.into_iter().collect::<Result<_, _>>().map_err(contig)?;
    let f1 = [&e0, &cap, &pmax, &eta_c, &eta_d, &soc_min, &soc_max].map(|a| a.as_slice());
    let f1: Vec<&[f64]> = f1.into_iter().collect::<Result<_, _>>().map_err(contig)?;
    let st = start.as_slice().map_err(contig)?;
    if f2.iter().any(|v| v.len() != m * t) || f1.iter().any(|v| v.len() != m) || st.len() != m {
        return Err(PyValueError::new_err("series must be (m, t); start and battery arrays (m,)"));
    }
    let x = Series { surplus: f2[0], soc0: f2[1], cf: f2[3], pf: f2[4] };
    let p = Params { start: st, e0: f1[0], cap: f1[1], pmax: f1[2], eta_c: f1[3], eta_d: f1[4], soc_min: f1[5],
                     soc_max: f1[6] };
    let (b, s) = py.detach(|| resim(&x, f2[2], m, t, &p, t0, dt));
    let e = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    Ok((Array2::from_shape_vec((m, t), b).map_err(e)?.into_pyarray(py),
        Array2::from_shape_vec((m, t), s).map_err(e)?.into_pyarray(py)))
}

/// np.maximum (scalar loop): NaN in either argument propagates.
#[inline(always)]
fn np_max(a: f64, b: f64) -> f64 {
    if a >= b || a.is_nan() { a } else { b }
}

/// np.minimum (scalar loop): NaN in either argument propagates.
#[inline(always)]
fn np_min(a: f64, b: f64) -> f64 {
    if a <= b || a.is_nan() { a } else { b }
}

/// np.clip(x, lo, hi) == _NPY_MIN(_NPY_MAX(x, lo), hi) with NaN in x propagating.
#[inline(always)]
fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    let m = if x.is_nan() || x > lo { x } else { lo };
    if m.is_nan() || m < hi { m } else { hi }
}

/// np.nan_to_num for f64.
#[inline(always)]
fn nan_to_num(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else if x == f64::INFINITY {
        f64::MAX
    } else if x == f64::NEG_INFINITY {
        f64::MIN
    } else {
        x
    }
}

pub struct Params<'a> {
    pub start: &'a [i64],
    pub e0: &'a [f64],
    pub cap: &'a [f64],
    pub pmax: &'a [f64],
    pub eta_c: &'a [f64],
    pub eta_d: &'a [f64],
    pub soc_min: &'a [f64],
    pub soc_max: &'a [f64],
}

pub struct Series<'a> {
    pub surplus: &'a [f64],
    pub soc0: &'a [f64],
    pub cf: &'a [f64],
    pub pf: &'a [f64],
}

/// One row, in place on `batt` / `soc` (pre-filled with the original batt0 / soc0). Steps before
/// max(t0, start) keep the original series, as in the reference (its `before` branch).
#[allow(clippy::too_many_arguments)]
#[inline]
fn row(surplus: &[f64], soc0: &[f64], cf: &[f64], pf: &[f64], i: usize, p: &Params, t0: usize, dt: f64,
       batt: &mut [f64], soc: &mut [f64]) {
    let t_len = surplus.len();
    if t_len == 0 || t0 >= t_len {
        return;
    }
    let (ncap, npow, eta_c, eta_d, smin, smax) =
        (p.cap[i], p.pmax[i], p.eta_c[i], p.eta_d[i], p.soc_min[i], p.soc_max[i]);
    let s = p.start[i].max(0) as usize;
    let s = s.max(t0);
    if s >= t_len {
        return;
    }
    // State entering step s: in the reference, every step t0 <= t < s sets e = nan_to_num(soc0[t]) * ncap and
    // prev_cap = ncap * cf[t]; prev_cap starts at ncap * cf[t0] and e at e0.
    let (mut e, mut prev_cap) = if s > t0 {
        (nan_to_num(soc0[s - 1]) * ncap, ncap * cf[s - 1])
    } else {
        (p.e0[i], ncap * cf[t0])
    };
    let eta_dt = eta_c * dt;
    for t in s..t_len {
        let cap = ncap * cf[t];
        let pmax = npow * pf[t];
        e = e * if prev_cap > 0.0 { cap / prev_cap } else { 1.0 };
        prev_cap = cap;
        // apply_power(sub, e, surplus[t], dt)
        let has = cap > 0.0;
        let lo = smin * cap;
        let hi = smax * cap;
        let p_max_w = pmax * 1000.0;
        let req = np_clip(surplus[t], -p_max_w, p_max_w);
        let room_w = np_max(0.0, (hi - e) / eta_dt) * 1000.0;
        let avail_w = np_max(0.0, (e - lo) * eta_d / dt) * 1000.0;
        let mut applied = if req >= 0.0 { np_min(req, room_w) } else { np_max(req, -avail_w) };
        if !has {
            applied = 0.0;
        }
        let e_next = if applied >= 0.0 {
            e + eta_c * applied / 1000.0 * dt
        } else {
            e + applied / 1000.0 * dt / eta_d
        };
        e = if has { np_min(hi, np_max(lo, e_next)) } else { e };
        batt[t] = applied;
        soc[t] = e / np_max(cap, 1e-9);
    }
}

/// All arrays row-major (m rows × t steps). Returns (batt, soc); rows run in parallel.
pub fn resim(x: &Series, batt0: &[f64], m: usize, t_len: usize, p: &Params, t0: usize, dt: f64) -> (Vec<f64>, Vec<f64>) {
    let mut batt = batt0.to_vec();
    let mut soc = x.soc0.to_vec();
    if m == 0 || t_len == 0 {
        return (batt, soc);
    }
    batt.par_chunks_mut(t_len)
        .zip(soc.par_chunks_mut(t_len))
        .enumerate()
        .for_each(|(i, (b, so))| {
            let r = i * t_len..(i + 1) * t_len;
            row(&x.surplus[r.clone()], &x.soc0[r.clone()], &x.cf[r.clone()], &x.pf[r], i, p, t0, dt, b, so)
        });
    (batt, soc)
}
