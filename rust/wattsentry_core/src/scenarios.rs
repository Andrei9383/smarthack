//! K9 park_scenarios (#187): one park's [S, T] kW scenarios from per-turbine P10/P50/P90, fused (no temporaries).
//! Python reference: `wattsentry.kernels.scenarios.park_scenarios_py` (the source of truth). Same op order as numpy:
//! the knot difference in q's dtype, then f64; NaN-propagating max/min; the turbine sum is sequential from the first turbine,
//! which is what numpy's reduce does for the [S][n][T] layout that `q_kw[:, cols]` (fancy index) produces.
use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
            PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::ops::Sub;

struct Args<'a, Q> {
    q: &'a [Q], // [T, N, 3]
    n_all: usize,
    rated: &'a [f64], // [N]
    cols: &'a [usize],
    ratio: &'a [f64],           // [T]
    u: &'a [f64],               // [S, T]
    pu: Option<&'a [f64]>,      // [T, N]
    der: Option<&'a [f64]>,     // [T, N]
    v: Option<&'a [f64]>,       // [S, n] Bernoulli draws
}

fn run<Q: Copy + Sub<Output = Q> + Into<f64> + Send + Sync>(a: &Args<Q>, s: usize, t: usize) -> Vec<f64> {
    let (n, na) = (a.cols.len(), a.n_all);
    let mut out = vec![0.0; s * t];
    out.par_chunks_mut(t).enumerate().for_each(|(si, row)| {
        for (ti, o) in row.iter_mut().enumerate() {
            let u = a.u[si * t + ti];
            let um = u - 0.5;
            let mut acc = 0.0;
            for (k, &c) in a.cols.iter().enumerate() {
                let i = ti * na + c;
                let (q10, q50, q90) = (a.q[3 * i], a.q[3 * i + 1], a.q[3 * i + 2]);
                let d: f64 = if u < 0.5 { (q50 - q10).into() } else { (q90 - q50).into() };
                let mut y = q50.into() + um * d / 0.4;
                if !(y >= 0.0 || y.is_nan()) {
                    y = 0.0;
                }
                let cap = a.rated[c];
                if !(y <= cap || y.is_nan()) {
                    y = cap;
                }
                if let Some(der) = a.der {
                    y *= der[i];
                }
                if let Some(pu) = a.pu {
                    y *= match a.v {
                        Some(v) => (v[si * n + k] < pu[i]) as u8 as f64,
                        None => pu[i],
                    };
                }
                acc = if k == 0 { y } else { acc + y };
            }
            *o = acc * a.ratio[ti];
        }
    });
    out
}

fn sl<'a, T: Element>(a: &'a PyReadonlyArray1<T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

/// park_scenarios(q f32|f64[T, N, 3], rated f64[N], cols i64[n], ratio f64[T], u f64[S, T],
///                p_up f64[T, N] | None, derate f64[T, N] | None, v f64[S, n] | None) -> f64[S, T]
#[pyfunction]
#[pyo3(signature = (q, rated, cols, ratio, u, p_up=None, derate=None, v=None))]
#[allow(clippy::too_many_arguments)]
pub fn park_scenarios<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyAny>,
    rated: PyReadonlyArray1<'py, f64>,
    cols: PyReadonlyArray1<'py, i64>,
    ratio: PyReadonlyArray1<'py, f64>,
    u: PyReadonlyArray2<'py, f64>,
    p_up: Option<PyReadonlyArray2<'py, f64>>,
    derate: Option<PyReadonlyArray2<'py, f64>>,
    v: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let err = |m: &str| PyValueError::new_err(m.to_string());
    let (s, t) = (u.shape()[0], u.shape()[1]);
    let rs = sl(&rated, "rated")?;
    let na = rs.len();
    let cols: Vec<usize> = sl(&cols, "cols")?
        .iter()
        .map(|&c| usize::try_from(c).ok().filter(|&c| c < na))
        .collect::<Option<_>>()
        .ok_or_else(|| err("cols out of range"))?;
    let ratio = sl(&ratio, "ratio")?;
    let us = u.as_slice().map_err(|_| err("u must be C-contiguous"))?;
    if ratio.len() != t {
        return Err(err("ratio must have length T"));
    }
    let tn = |x: &Option<PyReadonlyArray2<'py, f64>>| x.as_ref().map_or(true, |a| a.shape() == [t, na] && a.as_slice().is_ok());
    if !tn(&p_up) || !tn(&derate) {
        return Err(err("p_up / derate must be C-contiguous [T, N]"));
    }
    if let Some(vv) = &v {
        if vv.shape() != [s, cols.len()] || vv.as_slice().is_err() || p_up.is_none() {
            return Err(err("v must be C-contiguous [S, n] and needs p_up"));
        }
    }
    let pu = p_up.as_ref().map(|a| a.as_slice().unwrap());
    let der = derate.as_ref().map(|a| a.as_slice().unwrap());
    let vs = v.as_ref().map(|a| a.as_slice().unwrap());
    macro_rules! go {
        ($ty:ty) => {{
            let qa = q.extract::<PyReadonlyArray3<'py, $ty>>()?;
            if qa.shape() != [t, na, 3] {
                return Err(err("q must be [T, N, 3]"));
            }
            let qs = qa.as_slice().map_err(|_| err("q must be C-contiguous"))?;
            let a = Args { q: qs, n_all: na, rated: rs, cols: &cols, ratio, u: us, pu, der, v: vs };
            py.detach(|| run(&a, s, t))
        }};
    }
    let out = if q.extract::<PyReadonlyArray3<'py, f32>>().is_ok() { go!(f32) } else { go!(f64) };
    Ok(Array2::from_shape_vec((s, t), out).map_err(|e| err(&e.to_string()))?.into_pyarray(py))
}
