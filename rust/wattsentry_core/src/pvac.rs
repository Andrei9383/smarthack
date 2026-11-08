//! K8 pv_ac_batch (#133): pvlib isotropic plane-of-array irradiance + the PVWatts-like AC model, fused, for many
//! sites under one locality's weather. Python reference: `wattsentry.kernels.pvac.pv_ac_batch_py`
//! (= `wattsentry.model.physics.poa` + `ac_power_kw` broadcast over sites; the source of truth).
//!
//! Every floating-point operation follows numpy's order (no FMA, NaN-propagating max/min like np.maximum /
//! np.minimum / np.clip). The per-site and per-step trig (cosd/sind of tilt and zenith) comes in precomputed by
//! numpy; the per-element cos / acos here use libm, which is what numpy uses for float64 on CPUs without its
//! AVX-512 SVML paths, so the result is bit-identical there (and within a few ulp elsewhere).

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const D2R: f64 = std::f64::consts::PI / 180.0; // np.radians(x) == x * (pi / 180)
const R2D: f64 = 180.0 / std::f64::consts::PI; // np.rad2deg(x) == x * (180 / pi)

pub struct Model {
    pub albedo: f64,
    pub losses: f64,
    pub gamma: f64,
    pub noct_rise: f64,
    pub inv_eta: f64,
}

/// Per-step inputs, all of length t (temp and wind may be None: constant temp / no wind cooling).
pub struct Steps<'a> {
    pub ghi: &'a [f64],
    pub dni: &'a [f64],
    pub dhi: &'a [f64],
    pub temp: &'a [f64],
    pub wind: Option<&'a [f64]>,
    pub cos_zen: &'a [f64],
    pub sin_zen: &'a [f64],
    pub sun_az: &'a [f64],
}

/// Per-site inputs, all of length n.
pub struct Sites<'a> {
    pub cos_tilt: &'a [f64],
    pub sin_tilt: &'a [f64],
    pub azimuth: &'a [f64],
    pub kwp: &'a [f64],
    pub inverter_kw: &'a [f64],
}

#[inline]
fn npmax(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else if a >= b { a } else { b }
}

#[inline]
fn npmin(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else if a <= b { a } else { b }
}

#[inline]
fn nan_to_num(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else if x == f64::INFINITY { f64::MAX } else if x == f64::NEG_INFINITY { f64::MIN } else { x }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn row(i: usize, s: &Sites, w: &Steps, m: &Model, out: &mut [f64]) {
    let (ct, st, az, kwp, inv) = (s.cos_tilt[i], s.sin_tilt[i], s.azimuth[i], s.kwp[i], s.inverter_kw[i]);
    let one_m_losses = 1.0 - m.losses;
    let pdc0 = inv / m.inv_eta;
    let eta_scale = m.inv_eta / 0.9637;
    let pdc0_floor = npmax(pdc0, 1e-6);
    for t in 0..out.len() {
        // pvlib.irradiance.aoi_projection -> aoi -> poa_components (isotropic sky + ground diffuse)
        // dni == 0 (night, overcast): numpy's direct term is ±0, and ±0 + diffuse gives the same g for every
        // diffuse value that can reach the output (a -0 g only arises when diffuse is -0, and then dc <= 0 -> ac = 0),
        // so the three libm calls are skipped without changing a bit of the result.
        let dni = w.dni[t];
        let direct = if dni == 0.0 {
            0.0
        } else {
            let proj = ct * w.cos_zen[t] + st * w.sin_zen[t] * ((w.sun_az[t] - az) * D2R).cos();
            let proj = npmin(npmax(proj, -1.0), 1.0);
            let aoi = proj.acos() * R2D;
            npmax(dni * (aoi * D2R).cos(), 0.0)
        };
        let sky = w.dhi[t] * (1.0 + ct) * 0.5;
        let ground = w.ghi[t] * m.albedo * (1.0 - ct) * 0.5;
        let g = direct + (sky + ground);
        let g = npmax(nan_to_num(g), 0.0); // physics.poa: nan_to_num(...).clip(min=0)
        // physics.ac_power_kw
        let cool = match w.wind {
            Some(wd) => 1.0 / (1.0 + 0.08 * npmin(npmax(wd[t], 0.0), 15.0)),
            None => 1.0,
        };
        let t_cell = w.temp[t] + m.noct_rise * g * cool;
        let dc = kwp * g / 1000.0 * (1.0 + m.gamma * (t_cell - 25.0)) * one_m_losses;
        let dc = npmax(dc, 0.0);
        let zeta = npmax(dc / pdc0_floor, 1e-4);
        let eta = eta_scale * (-0.0162 * zeta - 0.0059 / zeta + 0.9858);
        let ac = if dc > 0.0 { dc * npmin(npmax(eta, 0.0), 1.0) } else { 0.0 };
        out[t] = npmin(ac, inv);
    }
}

/// Returns row-major (n sites × t steps) AC kW; rows run in parallel.
pub fn pv_ac(s: &Sites, w: &Steps, m: &Model, n: usize, t: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * t];
    if n == 0 || t == 0 {
        return out;
    }
    out.par_chunks_mut(t).enumerate().for_each(|(i, o)| row(i, s, w, m, o));
    out
}

/// pv_ac_batch(cos_tilt, sin_tilt, azimuth, kwp, inverter_kw: f64[n], ghi, dni, dhi, temp, cos_zen, sin_zen,
///             sun_az: f64[t], wind: f64[t] | None, albedo, losses, gamma, noct_rise, inv_eta) -> f64[n, t]
///
/// The wrapper `wattsentry.kernels.pvac.pv_ac_batch_rust` precomputes the trig of tilt and zenith with numpy.
#[pyfunction]
#[pyo3(signature = (cos_tilt, sin_tilt, azimuth, kwp, inverter_kw, ghi, dni, dhi, temp, cos_zen, sin_zen, sun_az,
                    wind, albedo, losses, gamma, noct_rise, inv_eta))]
#[allow(clippy::too_many_arguments)]
pub fn pv_ac_batch<'py>(
    py: Python<'py>,
    cos_tilt: PyReadonlyArray1<'py, f64>, sin_tilt: PyReadonlyArray1<'py, f64>, azimuth: PyReadonlyArray1<'py, f64>,
    kwp: PyReadonlyArray1<'py, f64>, inverter_kw: PyReadonlyArray1<'py, f64>,
    ghi: PyReadonlyArray1<'py, f64>, dni: PyReadonlyArray1<'py, f64>, dhi: PyReadonlyArray1<'py, f64>,
    temp: PyReadonlyArray1<'py, f64>, cos_zen: PyReadonlyArray1<'py, f64>, sin_zen: PyReadonlyArray1<'py, f64>,
    sun_az: PyReadonlyArray1<'py, f64>, wind: Option<PyReadonlyArray1<'py, f64>>,
    albedo: f64, losses: f64, gamma: f64, noct_rise: f64, inv_eta: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let contig = |e| PyValueError::new_err(format!("{e}: arrays must be C-contiguous"));
    let s = Sites { cos_tilt: cos_tilt.as_slice().map_err(contig)?, sin_tilt: sin_tilt.as_slice().map_err(contig)?,
                    azimuth: azimuth.as_slice().map_err(contig)?, kwp: kwp.as_slice().map_err(contig)?,
                    inverter_kw: inverter_kw.as_slice().map_err(contig)? };
    let wind_s = match &wind { Some(a) => Some(a.as_slice().map_err(contig)?), None => None };
    let w = Steps { ghi: ghi.as_slice().map_err(contig)?, dni: dni.as_slice().map_err(contig)?,
                    dhi: dhi.as_slice().map_err(contig)?, temp: temp.as_slice().map_err(contig)?, wind: wind_s,
                    cos_zen: cos_zen.as_slice().map_err(contig)?, sin_zen: sin_zen.as_slice().map_err(contig)?,
                    sun_az: sun_az.as_slice().map_err(contig)? };
    let (n, t) = (s.kwp.len(), w.ghi.len());
    for (name, len, want) in [("cos_tilt", s.cos_tilt.len(), n), ("sin_tilt", s.sin_tilt.len(), n),
                              ("azimuth", s.azimuth.len(), n), ("inverter_kw", s.inverter_kw.len(), n),
                              ("dni", w.dni.len(), t), ("dhi", w.dhi.len(), t), ("temp", w.temp.len(), t),
                              ("cos_zen", w.cos_zen.len(), t), ("sin_zen", w.sin_zen.len(), t),
                              ("sun_az", w.sun_az.len(), t), ("wind", w.wind.map_or(t, |x| x.len()), t)] {
        if len != want {
            return Err(PyValueError::new_err(format!("{name}: expected {want} elements, got {len}")));
        }
    }
    let m = Model { albedo, losses, gamma, noct_rise, inv_eta };
    let out = py.detach(|| pv_ac(&s, &w, &m, n, t));
    Ok(Array2::from_shape_vec((n, t), out).map_err(|e| PyValueError::new_err(e.to_string()))?.into_pyarray(py))
}
