//! sentinel._core: Rust kernels for Prosumer Sentinel (see team/infra.md §1-4).
//!
//! Rules: every kernel has a Python twin in `sentinel/kernels` with the same signature; that twin is the
//! source of truth. Inputs are C-contiguous numpy arrays plus scalars; outputs are newly allocated arrays.
//! The GIL is released for the loop, and work is parallel over the outer axis with rayon.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod cusum;

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

/// Number of rayon worker threads (for the bench log).
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(cusum_two_sided, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}
