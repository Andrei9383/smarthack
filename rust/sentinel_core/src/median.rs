//! Exact medians for the WindNBM label-free normal selection (card #136). Python references (source of truth):
//! `sentinel.kernels.median.rolling_median_py` (pandas `rolling(w, center=True, min_periods).median()`) and
//! `sentinel.kernels.median.binned_median_mad_py` (per-bin `np.median` + MAD).
//!
//! Both pick order statistics of the same values numpy/pandas pick; an even count averages the two middle values as
//! `(a + b) / 2`, as pandas' skiplist and `np.median` do, so the outputs are bit-identical (up to the sign of a zero).
//! Series are rows of a row-major [c, t] buffer, processed in parallel with rayon.

use rayon::prelude::*;

/// Insert v into the sorted window.
#[inline]
fn insert(win: &mut Vec<f64>, v: f64) {
    let i = win.partition_point(|&a| a < v);
    win.insert(i, v);
}

/// Remove one value equal to v from the sorted window (it is there: it was inserted when it entered).
#[inline]
fn remove(win: &mut Vec<f64>, v: f64) {
    let i = win.partition_point(|&a| a < v);
    if i < win.len() && win[i] == v {
        win.remove(i);
    }
}

/// Swap one value equal to `old` for `new`, shifting only the values between their two positions.
#[inline]
fn replace(win: &mut [f64], old: f64, new: f64) {
    let i = win.partition_point(|&a| a < old);
    debug_assert!(i < win.len() && win[i] == old);
    let j = win.partition_point(|&a| a < new);
    if j > i {
        win.copy_within(i + 1..j, i);
        win[j - 1] = new;
    } else {
        win.copy_within(j..i, j + 1);
        win[j] = new;
    }
}

#[inline]
fn mid(win: &[f64]) -> f64 {
    let n = win.len();
    let h = n / 2;
    if n % 2 == 1 { win[h] } else { (win[h] + win[h - 1]) / 2.0 }
}

/// Columns per task: one 64-byte cache line per row of the row-major [t, c] input.
const BLOCK: usize = 8;

/// Columns per task for c columns: up to BLOCK, but at least ~4 tasks per rayon thread (117 turbines on 20 threads
/// -> 1 column per task) so every thread gets work.
fn block_for(c: usize) -> usize {
    (c / (4 * rayon::current_num_threads())).clamp(1, BLOCK)
}

/// Raw output pointer shared by the column tasks; each task writes only its own columns.
#[derive(Clone, Copy)]
struct OutPtr(*mut f64);
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Centred rolling median of one contiguous series into out (same semantics as `rolling_median_cols`).
fn rolling_median_1d(x: &[f64], w: usize, minp: usize, out: &mut [f64], win: &mut Vec<f64>) {
    let t = x.len();
    let off = (w - 1) / 2;
    let need = minp.max(1);
    win.clear();
    let (mut s0, mut e0) = (0usize, 0usize);
    for (i, o) in out.iter_mut().enumerate() {
        let e = (i + 1 + off).min(t);
        let s = (i + 1 + off).saturating_sub(w).min(t);
        if e == e0 + 1 && s == s0 + 1 && s0 < e0 {
            // steady state: one value leaves, one enters
            let (old, new) = (x[s0], x[e0]);
            match (!old.is_finite(), !new.is_finite()) {
                (false, false) => replace(win, old, new),
                (true, false) => insert(win, new),
                (false, true) => remove(win, old),
                (true, true) => {}
            }
        } else {
            for &v in &x[e0.max(s)..e] {
                if v.is_finite() {
                    insert(win, v);
                }
            }
            for &v in &x[s0..s.min(e0)] {
                if v.is_finite() {
                    remove(win, v);
                }
            }
        }
        (s0, e0) = (s, e);
        *o = if win.len() >= need { mid(win) } else { f64::NAN };
    }
}

/// Centred rolling median down each column of row-major x [t, c] (pandas `rolling(w, center=True,
/// min_periods=minp).median()`): NaN and +-inf skipped (pandas maps inf to NaN), NaN where fewer than max(minp, 1)
/// values. Window bounds as pandas' FixedWindowIndexer(center=True): end = i + 1 + (w - 1) / 2, start = end - w,
/// both clipped to [0, t].
/// One task per block of 8 columns (one cache line per row): gather them into contiguous buffers, run each series
/// with its sorted window (<= w values) hot in L1, scatter back.
pub fn rolling_median_cols(x: &[f64], t: usize, c: usize, w: usize, minp: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; t * c];
    if t == 0 || c == 0 || w == 0 {
        return out;
    }
    let ptr = OutPtr(out.as_mut_ptr());
    let bw = block_for(c);
    (0..c.div_ceil(bw)).into_par_iter().for_each_init(
        || (vec![0.0f64; t * bw], vec![0.0f64; t * bw], Vec::with_capacity(w.min(t))),
        |(cols, res, win), blk| {
            let p = ptr;
            let (c0, nb) = (blk * bw, bw.min(c - blk * bw));
            for i in 0..t {
                for k in 0..nb {
                    cols[k * t + i] = x[i * c + c0 + k];
                }
            }
            for k in 0..nb {
                rolling_median_1d(&cols[k * t..(k + 1) * t], w, minp, &mut res[k * t..(k + 1) * t], win);
            }
            for i in 0..t {
                for k in 0..nb {
                    // SAFETY: (i, c0 + k) is in bounds and columns c0..c0 + nb belong to this task only.
                    unsafe { *p.0.add(i * c + c0 + k) = res[k * t + i] };
                }
            }
        },
    );
    out
}

/// Median of v (reordered in place), NaN if empty or any NaN (as np.median).
fn median_inplace(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 || v.iter().any(|a| a.is_nan()) {
        return f64::NAN;
    }
    let h = n / 2;
    let (lo, &mut m, _) = v.select_nth_unstable_by(h, |a, b| a.partial_cmp(b).unwrap());
    if n % 2 == 1 {
        m
    } else {
        let a = lo.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (m + a) / 2.0
    }
}

/// Per column j of row-major [t, c] inputs: med/mad [n_bins, c] of x[ok, j] grouped by b[ok, j] (a bin with fewer
/// than min_n values stays NaN). Err if a kept b is outside [0, n_bins). One task per block of 8 columns.
pub fn binned_median_mad_cols(b: &[i64], x: &[f64], ok: &[bool], t: usize, c: usize, n_bins: usize, min_n: usize)
    -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut med = vec![f64::NAN; n_bins * c];
    let mut mad = vec![f64::NAN; n_bins * c];
    if t == 0 || c == 0 {
        return Ok((med, mad));
    }
    if b.iter().zip(ok).any(|(&bi, &k)| k && (bi < 0 || bi as usize >= n_bins)) {
        return Err(format!("b must be in [0, {n_bins}) where ok"));
    }
    let (pm, pa) = (OutPtr(med.as_mut_ptr()), OutPtr(mad.as_mut_ptr()));
    let bw = block_for(c);
    (0..c.div_ceil(bw)).into_par_iter().for_each(|blk| {
        let (pm, pa) = (pm, pa);
        let (c0, nb) = (blk * bw, bw.min(c - blk * bw));
        // counting sort of the kept values by (column, bin)
        let nk = n_bins + 1;
        let mut cnt = vec![0usize; nb * nk + 1];
        for i in 0..t {
            for k in 0..nb {
                let q = i * c + c0 + k;
                if ok[q] {
                    cnt[k * nk + b[q] as usize + 1] += 1;
                }
            }
        }
        for i in 0..nb * nk {
            cnt[i + 1] += cnt[i];
        }
        let mut pos = cnt.clone();
        let mut vals = vec![0.0f64; cnt[nb * nk]];
        for i in 0..t {
            for k in 0..nb {
                let q = i * c + c0 + k;
                if ok[q] {
                    let p = &mut pos[k * nk + b[q] as usize];
                    vals[*p] = x[q];
                    *p += 1;
                }
            }
        }
        let mut dev: Vec<f64> = Vec::new();
        for k in 0..nb {
            for bin in 0..n_bins {
                let seg = &mut vals[cnt[k * nk + bin]..cnt[k * nk + bin + 1]];
                if seg.is_empty() || seg.len() < min_n {
                    continue;
                }
                let m = median_inplace(seg);
                dev.clear();
                dev.extend(seg.iter().map(|&v| (v - m).abs()));
                let d = median_inplace(&mut dev);
                // SAFETY: (bin, c0 + k) is in bounds and columns c0..c0 + nb belong to this task only.
                unsafe {
                    *pm.0.add(bin * c + c0 + k) = m;
                    *pa.0.add(bin * c + c0 + k) = d;
                }
            }
        }
    });
    Ok((med, mad))
}

// ---- Python bindings (registered in lib.rs) -------------------------------------------------------------------
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// rolling_median(x f64[t, c], window, min_periods) -> f64[t, c]: centred NaN/inf-skipping rolling median down each
/// column (pandas semantics); the wrapper `sentinel.kernels.median.rolling_median_rust` reshapes (T, ...) to 2-D.
#[pyfunction]
pub fn rolling_median<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, window: i64, min_periods: i64)
    -> PyResult<Bound<'py, PyArray2<f64>>> {
    if window < 1 || min_periods < 0 || min_periods > window {
        return Err(PyValueError::new_err("need window >= 1 and 0 <= min_periods <= window"));
    }
    let (t, c) = (x.shape()[0], x.shape()[1]);
    let xs = x.as_slice().map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let out = py.detach(|| rolling_median_cols(xs, t, c, window as usize, min_periods as usize));
    Ok(Array2::from_shape_vec((t, c), out).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py))
}

/// binned_median_mad(b i64[t, c], x f64[t, c], ok bool[t, c], n_bins, min_n) -> (med, mad) f64[n_bins, c]
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn binned_median_mad<'py>(py: Python<'py>, b: PyReadonlyArray2<'py, i64>, x: PyReadonlyArray2<'py, f64>,
                              ok: PyReadonlyArray2<'py, bool>, n_bins: usize, min_n: usize)
    -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (t, c) = (x.shape()[0], x.shape()[1]);
    if b.shape() != [t, c] || ok.shape() != [t, c] {
        return Err(PyValueError::new_err("b, x, ok must have the same shape"));
    }
    let contig = |_| PyValueError::new_err("inputs must be C-contiguous");
    let (bs, xs, ks) = (b.as_slice().map_err(contig)?, x.as_slice().map_err(contig)?, ok.as_slice().map_err(contig)?);
    let (med, mad) = py.detach(|| binned_median_mad_cols(bs, xs, ks, t, c, n_bins, min_n))
        .map_err(PyValueError::new_err)?;
    let e = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    Ok((Array2::from_shape_vec((n_bins, c), med).map_err(e)?.into_pyarray(py),
        Array2::from_shape_vec((n_bins, c), mad).map_err(e)?.into_pyarray(py)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_small() {
        let x = [1.0, f64::NAN, 3.0, 2.0, 5.0];
        let out = rolling_median_cols(&x, 5, 1, 3, 1);
        assert_eq!(out, vec![1.0, 2.0, 2.5, 3.0, 3.5]);
    }

    #[test]
    fn binned_small() {
        let b = [0i64, 0, 1, 1, 1, 0];
        let x = [1.0, 3.0, 5.0, 7.0, 6.0, 2.0];
        let ok = [true; 6];
        let (md, ma) = binned_median_mad_cols(&b, &x, &ok, 6, 1, 2, 1).unwrap();
        assert_eq!(md, vec![2.0, 6.0]);
        assert_eq!(ma, vec![1.0, 1.0]);
    }
}
