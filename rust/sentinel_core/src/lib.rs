//! sentinel._core: Rust kernels for Prosumer Sentinel (see team/infra.md §1-4).
//!
//! Rules: every kernel has a Python twin in `sentinel/kernels` with the same signature; that twin is the
//! source of truth. Inputs are C-contiguous numpy arrays plus scalars; outputs are newly allocated arrays.
//! The GIL is released for the loop, and work is parallel over the outer axis with rayon.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
            PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod cusum;
mod rollout;
mod windows;

/// cusum_two_sided(x: f64[m, n], k: float, h: float) -> (s_pos f64[m, n], s_neg f64[m, n], alarm bool[m, n])
///
/// The Python wrapper `sentinel.kernels.cusum_two_sided` handles 1-D input and validation.
#[pyfunction]
#[pyo3(signature = (x, k, h))]
fn cusum_two_sided<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: f64,
    h: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)> {
    if !k.is_finite() || !h.is_finite() {
        return Err(PyValueError::new_err("k and h must be finite"));
    }
    let shape = x.shape();
    let (m, n) = (shape[0], shape[1]);
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let (sp, sn, al) = py.detach(|| cusum::cusum_2d(xs, m, n, k, h));
    let to2 = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    let sp = Array2::from_shape_vec((m, n), sp).map_err(to2)?.into_pyarray(py);
    let sn = Array2::from_shape_vec((m, n), sn).map_err(to2)?.into_pyarray(py);
    let al = Array2::from_shape_vec((m, n), al).map_err(to2)?.into_pyarray(py);
    Ok((sp, sn, al))
}

/// battery_rollout(req f64[n, s, t], soc0 f64[n, s], cap, pmax, eta_c, eta_d, e_lo, e_hi: f64[s], dt)
///   -> (soc f64[n, s, t], batt f64[n, s, t])
///
/// The Python wrapper `sentinel.kernels.rollout.battery_rollout_rust` broadcasts and validates the inputs.
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
    let sh = req.shape();
    let (n, s, t) = (sh[0], sh[1], sh[2]);
    let contig = |e| PyValueError::new_err(format!("{e}: inputs must be C-contiguous"));
    let r = req.as_slice().map_err(contig)?;
    let s0 = soc0.as_slice().map_err(contig)?;
    if s0.len() != n * s {
        return Err(PyValueError::new_err("soc0 must have shape (n, s)"));
    }
    let arrs = [&cap, &pmax, &eta_c, &eta_d, &e_lo, &e_hi];
    let mut sl = Vec::with_capacity(6);
    for a in arrs {
        let v = a.as_slice().map_err(contig)?;
        if v.len() != s {
            return Err(PyValueError::new_err("battery arrays must have shape (s,)"));
        }
        sl.push(v);
    }
    let p = rollout::Params { cap: sl[0], pmax: sl[1], eta_c: sl[2], eta_d: sl[3], e_lo: sl[4], e_hi: sl[5] };
    let (soc, batt) = py.detach(|| rollout::rollout_3d(r, s0, n, s, t, &p, dt));
    let to3 = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    let soc = Array3::from_shape_vec((n, s, t), soc).map_err(to3)?.into_pyarray(py);
    let batt = Array3::from_shape_vec((n, s, t), batt).map_err(to3)?.into_pyarray(py);
    Ok((soc, batt))
}

/// Shared argument checks for the K2 window gathers; returns (r, w, length) as usize.
fn window_args(src_shape: &[usize], rows: usize, starts: usize, length: i64, out_shape: &[usize])
    -> PyResult<(usize, usize, usize)> {
    if rows != starts {
        return Err(PyValueError::new_err("rows and starts must have the same length"));
    }
    if length < 0 {
        return Err(PyValueError::new_err("length must be >= 0"));
    }
    let length = length as usize;
    if out_shape != [rows, length] {
        return Err(PyValueError::new_err(format!("out must have shape ({rows}, {length}), got {out_shape:?}")));
    }
    Ok((src_shape[0], src_shape[1], length))
}

/// window_gather_bytes(src u8[r, w], rows i64[b], starts i64[b], step, offset, length, out u8[b, length])
///
/// out[k] = src[rows[k], starts[k]*step + offset ..][..length] (units = bytes). Same-dtype window copy; the
/// Python wrapper `sentinel.kernels.windows.window_gather_rust` views the arrays as bytes.
#[pyfunction]
#[pyo3(signature = (src, rows, starts, step, offset, length, out))]
#[allow(clippy::too_many_arguments)]
fn window_gather_bytes<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, u8>,
    rows: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    step: i64,
    offset: i64,
    length: i64,
    mut out: PyReadwriteArray2<'py, u8>,
) -> PyResult<()> {
    let (r, w, length) = window_args(src.shape(), rows.len(), starts.len(), length, out.shape())?;
    let contig = |e| PyValueError::new_err(format!("{e}: arrays must be C-contiguous"));
    let (s, ro, st) = (src.as_slice().map_err(contig)?, rows.as_slice().map_err(contig)?,
                       starts.as_slice().map_err(contig)?);
    let o = out.as_slice_mut().map_err(contig)?;
    py.detach(|| windows::gather_copy(s, r, w, ro, st, step, offset, length, o)).map_err(PyValueError::new_err)
}

/// window_gather_f16_f32(src u16[r, w] (float16 bits), rows, starts, step, offset, length, out f32[b, length])
///
/// Like `window_gather_bytes` (units = elements) with float16 -> float32 conversion, bit-identical to numpy.
#[pyfunction]
#[pyo3(signature = (src, rows, starts, step, offset, length, out))]
#[allow(clippy::too_many_arguments)]
fn window_gather_f16_f32<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, u16>,
    rows: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    step: i64,
    offset: i64,
    length: i64,
    mut out: PyReadwriteArray2<'py, f32>,
) -> PyResult<()> {
    let (r, w, length) = window_args(src.shape(), rows.len(), starts.len(), length, out.shape())?;
    let contig = |e| PyValueError::new_err(format!("{e}: arrays must be C-contiguous"));
    let (s, ro, st) = (src.as_slice().map_err(contig)?, rows.as_slice().map_err(contig)?,
                       starts.as_slice().map_err(contig)?);
    let o = out.as_slice_mut().map_err(contig)?;
    py.detach(|| windows::gather_f16_f32(s, r, w, ro, st, step, offset, length, o)).map_err(PyValueError::new_err)
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
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}
