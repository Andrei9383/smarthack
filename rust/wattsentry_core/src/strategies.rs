//! K8 battery strategies (card #172, spec I step 2c). Python reference: `wattsentry.kernels.strategies.
//! battery_strategies_py` (source of truth). Floating-point operations are written in numpy's order so the results
//! are bit-identical. Homes run in parallel (rayon); each home is a loop over time (SoC carries over).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

pub struct Homes<'a> {
    pub strategy: &'a [i8],
    pub cap: &'a [f64],
    pub pmax: &'a [f64],
    pub reserve: &'a [f64],
    pub eta: &'a [f64],
    pub k_top: &'a [i32],
    pub soc0: &'a [f64],
}

pub struct Steps<'a> {
    pub price: &'a [f64],
    pub rank: &'a [i32],
    pub med: &'a [f64],
    pub day: &'a [i32],
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn row(pv: &[f64], load: &[f64], i: usize, h: &Homes, st: &Steps, comply: &[u8], dt: f64, batt: &mut [f64],
       socs: &mut [f64]) {
    let (strategy, cap, pmax, eta, k_top) = (h.strategy[i], h.cap[i], h.pmax[i], h.eta[i], h.k_top[i]);
    let has = strategy != 0;
    let pdriven = strategy == 2;
    let floor = h.reserve[i] * cap;
    let mut soc = h.soc0[i];
    for t in 0..pv.len() {
        let s = pv[t] - load[t];
        let room = (cap - soc).max(0.0) / (eta * dt);
        let avail = (soc - floor).max(0.0) * eta / dt;
        let charge = s.min(pmax).min(room);
        let cover = (-s).min(pmax).min(avail);
        let full = pmax.min(avail);
        let mode = pdriven && comply[st.day[t] as usize] != 0;
        let dis = if mode {
            if st.rank[t] < k_top { full } else if st.price[t] >= st.med[t] { cover } else { 0.0 }
        } else {
            cover
        };
        let c = if has && s > 0.0 { charge } else { 0.0 };
        let d = if has && !(s > 0.0) { dis } else { 0.0 };
        soc = soc + eta * c * dt - d * dt / eta;
        batt[t] = c - d;
        socs[t] = soc;
    }
}

/// Row-major pv, load (n × t), comply (n × n_days); rows run in parallel.
#[allow(clippy::too_many_arguments)]
pub fn strategies_2d(pv: &[f64], load: &[f64], n: usize, t: usize, h: &Homes, st: &Steps, comply: &[u8], nd: usize,
                     dt: f64) -> (Vec<f64>, Vec<f64>) {
    let mut batt = vec![0.0f64; n * t];
    let mut socs = vec![0.0f64; n * t];
    if n == 0 || t == 0 {
        return (batt, socs);
    }
    pv.par_chunks(t)
        .zip(load.par_chunks(t))
        .zip(batt.par_chunks_mut(t))
        .zip(socs.par_chunks_mut(t))
        .enumerate()
        .for_each(|(i, (((p, l), b), s))| row(p, l, i, h, st, &comply[i * nd..(i + 1) * nd], dt, b, s));
    (batt, socs)
}

/// battery_strategies(pv f64[n, t], load f64[n, t], strategy i8[n], cap, pmax, reserve, eta f64[n], k_top i32[n],
///                    soc0 f64[n], price f64[t], rank i32[t], med f64[t], day i32[t], comply u8[n, d], dt)
///   -> (batt f64[n, t], soc f64[n, t])
///
/// The wrapper `wattsentry.kernels.strategies.battery_strategies_rust` validates dtypes and day bounds.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn battery_strategies<'py>(
    py: Python<'py>,
    pv: PyReadonlyArray2<'py, f64>,
    load: PyReadonlyArray2<'py, f64>,
    strategy: PyReadonlyArray1<'py, i8>,
    cap: PyReadonlyArray1<'py, f64>,
    pmax: PyReadonlyArray1<'py, f64>,
    reserve: PyReadonlyArray1<'py, f64>,
    eta: PyReadonlyArray1<'py, f64>,
    k_top: PyReadonlyArray1<'py, i32>,
    soc0: PyReadonlyArray1<'py, f64>,
    price: PyReadonlyArray1<'py, f64>,
    rank: PyReadonlyArray1<'py, i32>,
    med: PyReadonlyArray1<'py, f64>,
    day: PyReadonlyArray1<'py, i32>,
    comply: PyReadonlyArray2<'py, u8>,
    dt: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (n, t) = (pv.shape()[0], pv.shape()[1]);
    let nd = comply.shape()[1];
    let contig = |e| PyValueError::new_err(format!("{e}: inputs must be C-contiguous"));
    let (p, l, c) = (pv.as_slice().map_err(contig)?, load.as_slice().map_err(contig)?,
                     comply.as_slice().map_err(contig)?);
    let h = Homes {
        strategy: strategy.as_slice().map_err(contig)?, cap: cap.as_slice().map_err(contig)?,
        pmax: pmax.as_slice().map_err(contig)?, reserve: reserve.as_slice().map_err(contig)?,
        eta: eta.as_slice().map_err(contig)?, k_top: k_top.as_slice().map_err(contig)?,
        soc0: soc0.as_slice().map_err(contig)?,
    };
    let st = Steps { price: price.as_slice().map_err(contig)?, rank: rank.as_slice().map_err(contig)?,
                     med: med.as_slice().map_err(contig)?, day: day.as_slice().map_err(contig)? };
    let per_home = [h.strategy.len(), h.cap.len(), h.pmax.len(), h.reserve.len(), h.eta.len(), h.k_top.len(),
                    h.soc0.len()];
    if l.len() != n * t || c.len() != n * nd || per_home.iter().any(|&k| k != n) {
        return Err(PyValueError::new_err("load (n, t), comply (n, d) and per-home arrays (n,) must match pv"));
    }
    if [st.price.len(), st.rank.len(), st.med.len(), st.day.len()].iter().any(|&k| k != t) {
        return Err(PyValueError::new_err("price, rank, med, day must have shape (t,)"));
    }
    if st.day.iter().any(|&d| d < 0 || d as usize >= nd) {
        return Err(PyValueError::new_err("day indexes must lie in [0, comply.shape[1])"));
    }
    let (batt, socs) = py.detach(|| strategies_2d(p, l, n, t, &h, &st, c, nd, dt));
    let e = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    Ok((Array2::from_shape_vec((n, t), batt).map_err(e)?.into_pyarray(py),
        Array2::from_shape_vec((n, t), socs).map_err(e)?.into_pyarray(py)))
}
