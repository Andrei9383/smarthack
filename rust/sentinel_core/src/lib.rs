//! sentinel._core: Rust kernels (team/infra.md §1-4). Each has a Python twin in `sentinel/kernels` (the source of
//! truth, which also validates and reshapes inputs). The GIL is released; work is parallel over rows with rayon.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
            PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod cusum;
mod rollout;
mod rolling;
mod scatter;
mod windows;

/// cusum_two_sided(x f64[m, n], k, h, cap=None) -> (s_pos f64[m, n], s_neg f64[m, n], alarm bool[m, n])
#[pyfunction]
#[pyo3(signature = (x, k, h, cap=None))]
fn cusum_two_sided<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: f64,
    h: f64,
    cap: Option<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    if !k.is_finite() || !h.is_finite() {
        return Err(PyValueError::new_err("k and h must be finite"));
    }
    let cap = match cap {
        None => f64::INFINITY,
        Some(c) if c.is_finite() && c >= 0.0 => c,
        Some(_) => return Err(PyValueError::new_err("cap must be finite and >= 0 (or None)")),
    };
    let (m, n) = (x.shape()[0], x.shape()[1]);
    let xs = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let (sp, sn, al) = py.detach(|| cusum::cusum_2d(xs, m, n, k, h, cap));
    let to2 = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    let sp = Array2::from_shape_vec((m, n), sp).map_err(to2)?.into_pyarray(py);
    let sn = Array2::from_shape_vec((m, n), sn).map_err(to2)?.into_pyarray(py);
    let al = Array2::from_shape_vec((m, n), al).map_err(to2)?.into_pyarray(py);
    Ok((sp, sn, al))
}

/// battery_rollout(req f64[n, s, t], soc0 f64[n, s], cap, pmax, eta_c, eta_d, e_lo, e_hi: f64[s], dt)
///   -> (soc f64[n, s, t], batt f64[n, s, t])
#[pyfunction]
#[pyo3(signature = (req, soc0, cap, pmax, eta_c, eta_d, e_lo, e_hi, dt))]
#[allow(clippy::too_many_arguments)]
fn battery_rollout<'py>(
    py: Python<'py>,
    req: PyReadonlyArray3<'py, f64>,
    soc0: PyReadonlyArray2<'py, f64>,
    cap: PyReadonlyArray1<'py, f64>,
    pmax: PyReadonlyArray1<'py, f64>,
    eta_c: PyReadonlyArray1<'py, f64>,
    eta_d: PyReadonlyArray1<'py, f64>,
    e_lo: PyReadonlyArray1<'py, f64>,
    e_hi: PyReadonlyArray1<'py, f64>,
    dt: f64,
) -> PyResult<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<f64>>)> {
    let (n, s, t) = (req.shape()[0], req.shape()[1], req.shape()[2]);
    let contig = |e| PyValueError::new_err(format!("{e}: inputs must be C-contiguous"));
    let r = req.as_slice().map_err(contig)?;
    let s0 = soc0.as_slice().map_err(contig)?;
    if s0.len() != n * s {
        return Err(PyValueError::new_err("soc0 must have shape (n, s)"));
    }
    let sl = [&cap, &pmax, &eta_c, &eta_d, &e_lo, &e_hi].map(|a| a.as_slice());
    let sl: Vec<&[f64]> = sl.into_iter().collect::<Result<_, _>>().map_err(contig)?;
    if sl.iter().any(|v| v.len() != s) {
        return Err(PyValueError::new_err("battery arrays must have shape (s,)"));
    }
    let p = rollout::Params { cap: sl[0], pmax: sl[1], eta_c: sl[2], eta_d: sl[3], e_lo: sl[4], e_hi: sl[5] };
    let (soc, batt) = py.detach(|| rollout::rollout_3d(r, s0, n, s, t, &p, dt));
    let to3 = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    let soc = Array3::from_shape_vec((n, s, t), soc).map_err(to3)?.into_pyarray(py);
    let batt = Array3::from_shape_vec((n, s, t), batt).map_err(to3)?.into_pyarray(py);
    Ok((soc, batt))
}

/// K2 window gathers: out[b] = conv(src[rows[b], starts[b]*step + offset ..][..length]); the Python wrapper
/// `sentinel.kernels.windows.window_gather_rust` views arrays as bytes (same dtype) or float16 bits.
macro_rules! window_gather {
    ($name:ident, $src:ty, $out:ty, $conv:expr) => {
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(py: Python<'py>, src: PyReadonlyArray2<'py, $src>, rows: PyReadonlyArray1<'py, i64>,
                      starts: PyReadonlyArray1<'py, i64>, step: i64, offset: i64, length: i64,
                      mut out: PyReadwriteArray2<'py, $out>) -> PyResult<()> {
            if rows.len() != starts.len() {
                return Err(PyValueError::new_err("rows and starts must have the same length"));
            }
            if length < 0 {
                return Err(PyValueError::new_err("length must be >= 0"));
            }
            let (b, length) = (rows.len(), length as usize);
            if out.shape() != [b, length] {
                return Err(PyValueError::new_err(format!("out must have shape ({b}, {length}), got {:?}", out.shape())));
            }
            let (r, w) = (src.shape()[0], src.shape()[1]);
            let contig = |e| PyValueError::new_err(format!("{e}: arrays must be C-contiguous"));
            let (s, ro, st) = (src.as_slice().map_err(contig)?, rows.as_slice().map_err(contig)?,
                               starts.as_slice().map_err(contig)?);
            let o = out.as_slice_mut().map_err(contig)?;
            let conv = $conv;
            py.detach(|| windows::gather(s, r, w, ro, st, step, offset, length, o, conv)).map_err(PyValueError::new_err)
        }
    };
}
window_gather!(window_gather_bytes, u8, u8, |x: u8| x);
window_gather!(window_gather_f16_f32, u16, f32, {
    let t = windows::f16_table();
    move |h: u16| f32::from_bits(t[h as usize])
});

/// rolling_mean(x: f64[t, m], w: int) -> f64[t, m]
///
/// Causal NaN-aware rolling mean over the last w rows; the Python wrapper
/// `sentinel.kernels.rolling.rolling_mean_rust` reshapes (T, ...) input to 2-D.
#[pyfunction]
fn rolling_mean<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, w: i64) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if w < 0 {
        return Err(PyValueError::new_err("w must be >= 0"));
    }
    let (t, m) = (x.shape()[0], x.shape()[1]);
    let xs = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let out = py.detach(|| rolling::rolling_mean_2d(xs, t, m, w as usize));
    Ok(Array2::from_shape_vec((t, m), out).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py))
}

/// scatter_telemetry(ti, tj, sk: i64[n], v: f64[n], is_code: bool[n], scale: f64[s, m], t: int)
///   -> (sens f32[s, t, m], err i32[t, m])
///
/// Long telemetry rows -> bin means per (sensor, step, turbine) + last code per bin; see `scatter.rs`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn scatter_telemetry<'py>(
    py: Python<'py>,
    ti: PyReadonlyArray1<'py, i64>,
    tj: PyReadonlyArray1<'py, i64>,
    sk: PyReadonlyArray1<'py, i64>,
    v: PyReadonlyArray1<'py, f64>,
    is_code: PyReadonlyArray1<'py, bool>,
    scale: PyReadonlyArray2<'py, f64>,
    t: usize,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<i32>>)> {
    let contig = |_| PyValueError::new_err("inputs must be C-contiguous");
    let (s, m) = (scale.shape()[0], scale.shape()[1]);
    let (a, j, k, x, c, sc) = (ti.as_slice().map_err(contig)?, tj.as_slice().map_err(contig)?,
                               sk.as_slice().map_err(contig)?, v.as_slice().map_err(contig)?,
                               is_code.as_slice().map_err(contig)?, scale.as_slice().map_err(contig)?);
    let n = x.len();
    if a.len() != n || j.len() != n || k.len() != n || c.len() != n {
        return Err(PyValueError::new_err("ti, tj, sk, v, is_code must have the same length"));
    }
    let (sens, err) = py.detach(|| scatter::scatter(a, j, k, x, c, sc, s, t, m));
    let e = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    Ok((Array3::from_shape_vec((s, t, m), sens).map_err(e)?.into_pyarray(py),
        Array2::from_shape_vec((t, m), err).map_err(e)?.into_pyarray(py)))
}

/// Number of rayon worker threads (for the bench log).
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(cusum_two_sided, m)?)?;
    m.add_function(wrap_pyfunction!(battery_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(window_gather_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(window_gather_f16_f32, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(scatter_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}
