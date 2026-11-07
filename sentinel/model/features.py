"""Numpy feature builders shared by training (data.py arrays) and the onnxruntime predictor (Window).

Both paths first build a *raw* window dict (batch axis first, power in kW), then call the same `pv_inputs`,
`load_inputs` or `site_inputs`. Nothing here needs torch.

Raw keys (B = batch, H = 96 target steps, HH = 14*96 history steps right before the target):
  wx [B,H,7] weather (actual for nowcast, forecast for forecast mode), geom [B,H,3] (zenith, azimuth, csghi),
  cal [B,H,8], pvlib [B,H], pvclear [B,H], hist_pv/hist_load [B,HH], and per-site static fields.
  M4 adds pv, volt, status, soc, batt, load [B,H] for the assessed window, m1 [B,H,3], m2 [B,H] and
  hist_volt / hist_status / hist_soc / hist_batt [B,HH].
"""

from __future__ import annotations

import numpy as np

from sentinel.kernels import load_features, pv_features, site_features, window_gather

from . import physics as ph

H = ph.DAY
HIST_DAYS = 14
REF_DAYS = 7
ARCHETYPES = ["base", "electric_heating", "heat_pump", "ac", "ev"]

PV_SEQ = ("ghi", "dni", "dhi", "temp", "cloud", "wind", "snow", "cos_zen", "sin_az", "cos_az", "csghi",
          "pvlib", "pvclear", "ref_mean", "ref_max", "yesterday", "ref_clear_ratio")
PV_SUM = ("energy_ratio", "max_ratio", "am_shift", "zero_frac", "frozen_frac")
PV_STATIC = ("lat", "lon", "kwp", "tilt", "az_sin", "az_cos", "inv_ratio", "has_batt", "doy_sin", "doy_cos")
LOAD_SEQ = ("temp", "ghi") + ph.CAL_CHANNELS + ("yesterday", "last_week", "mean7", "max7")
LOAD_SUM = ("energy", "peak", "min", "evening_share")
HOUSE_FLAGS = ("has_ac", "has_ev", "has_electric_boiler", "works_from_home")
LOAD_STATIC = tuple(f"arch_{a}" for a in ARCHETYPES) + ("has_batt", "lat", "lon", "log_scale") + HOUSE_FLAGS
SITE_SEQ = ("pv", "q10", "q50", "q90", "res", "zres", "pvlib", "cos_zen", "volt", "st_prod", "st_limit",
            "st_trip", "frozen", "soc", "batt", "load_res", "ratio", "valid")
SITE_SUM = PV_SUM + ("vmax", "trip_frac", "soc_swing")
SITE_STATIC = ("has_batt", "kwp", "inv_ratio", "batt_kwh", "batt_kw")


def _geom_feats(geom):
    zen = np.deg2rad(geom[..., 0].astype(np.float32))
    az = np.deg2rad(geom[..., 1].astype(np.float32))
    return np.cos(zen), np.sin(az), np.cos(az), geom[..., 2].astype(np.float32) / 1000.0


def _hist_days(h, n_days):
    """[B, HH] -> [B, n_days, H] last n_days."""
    B = h.shape[0]
    return h[:, -n_days * H:].reshape(B, n_days, H)


def frozen_flags(x: np.ndarray, prev: np.ndarray | None = None) -> np.ndarray:
    """1 where a positive reading repeats the previous one exactly (stuck meter)."""
    p = np.concatenate([x[..., :1] * np.nan if prev is None else prev[..., None], x[..., :-1]], -1)
    return ((x == p) & (x > 0.02)).astype(np.float32)


def pv_summary(hist_pv, pvclear, geom, kwp):
    """[B, 14, 5] daily stats of the site's own past power vs today's clear-sky profile (time-of-day aligned)."""
    days = _hist_days(hist_pv, HIST_DAYS).astype(np.float32)
    clear = pvclear.astype(np.float32)[:, None, :]
    e_clear = clear.sum(-1) + 1e-3
    energy_ratio = days.sum(-1) / e_clear
    max_ratio = days.max(-1) / (clear.max(-1) + 1e-3)
    am = (geom[..., 1] < 180).astype(np.float32)[:, None, :]  # sun in the east = morning
    am_share = (days * am).sum(-1) / (days.sum(-1) + 1e-3)
    am_clear = (clear * am).sum(-1) / e_clear
    day_mask = clear > 0.1 * kwp[:, None, None]
    zero_frac = ((days < 0.01) & day_mask).sum(-1) / (day_mask.sum(-1) + 1)
    fz = frozen_flags(days.reshape(days.shape[0], -1)).reshape(days.shape)
    frozen_frac = (fz * day_mask).sum(-1) / (day_mask.sum(-1) + 1)
    out = np.stack([energy_ratio, max_ratio, am_share - am_clear, zero_frac, frozen_frac], -1)
    return np.clip(np.nan_to_num(out), -3, 3).astype(np.float32)


def pv_arrays(wx, geom, pvclear, pvlib, hist_pv, kwp):
    """K4 reference (sentinel.kernels.pv_features): float32 seq [B,H,17], hourly ref [B,168], summ [B,14,5], night."""
    kwp = kwp.astype(np.float32)
    k = kwp[:, None]
    wx = wx.astype(np.float32)
    cz, sa, ca, cs = _geom_feats(geom)
    pvc = pvclear.astype(np.float32) / k
    ref = _hist_days(hist_pv, REF_DAYS).astype(np.float32) / k[:, :, None]
    ref_mean, ref_max = ref.mean(1), ref.max(1)
    seq = np.stack([
        wx[..., 0] / 1000, wx[..., 1] / 1000, wx[..., 2] / 500, wx[..., 3] / 20, wx[..., 4] / 100,
        wx[..., 5] / 10, wx[..., 6] / 10, cz, sa, ca, cs,
        pvlib.astype(np.float32) / k, pvc, ref_mean, ref_max, ref[:, -1],
        np.clip(ref_max / np.maximum(pvc, 0.02), 0, 2) * (pvc > 0.01),
    ], -1)
    B = seq.shape[0]
    refh = ref.reshape(B, REF_DAYS * 24, 4).mean(-1)
    summ = pv_summary(hist_pv, pvclear, geom, kwp)
    night = (cz <= 0.0) & (cs <= 0.0)
    return seq.astype(np.float32), refh.astype(np.float32), summ, night


def pv_inputs(raw: dict) -> dict:
    kwp = raw["kwp_recorded"].astype(np.float32)
    seq, refh, summ, night = pv_features(raw["wx"], raw["geom"], raw["pvclear"], raw["pvlib"], raw["hist_pv"], kwp)
    azr = np.deg2rad(raw["azimuth_recorded"].astype(np.float32))
    stat = np.stack([
        (raw["lat"] - 45.5) / 1.5, (raw["lon"] - 25.0) / 2.5, kwp / 10, raw["tilt_recorded"] / 45,
        np.sin(azr), np.cos(azr), raw["inverter_kw"] / kwp, raw["has_batt"],
        raw["doy_sin"], raw["doy_cos"],
    ], -1).astype(np.float32)
    return {"seq": seq, "ref": refh, "summ": summ, "stat": stat, "scale": kwp, "night": night}


def load_arrays(wx, cal, hist_load):
    """K4 reference (sentinel.kernels.load_features): float32 seq [B,H,14], hourly ref [B,168], summ [B,14,4], scale."""
    hl = hist_load.astype(np.float32)
    scale = hl.mean(1) + 0.05
    s = scale[:, None]
    days = _hist_days(hl, HIST_DAYS)
    ref = days[:, -REF_DAYS:] / s[:, :, None]
    wx = wx.astype(np.float32)
    cal = cal.astype(np.float32)
    seq = np.concatenate([
        np.stack([wx[..., 3] / 20, wx[..., 0] / 1000], -1), cal,
        np.stack([ref[:, -1], ref[:, 0], ref.mean(1), ref.max(1)], -1),
    ], -1)
    B = seq.shape[0]
    refh = ref.reshape(B, REF_DAYS * 24, 4).mean(-1)
    hour = (np.arctan2(cal[..., 0], cal[..., 1]) / (2 * np.pi) * 24) % 24
    eve = ((hour >= 17) & (hour < 23)).astype(np.float32)[:, None, :]
    summ = np.stack([
        days.mean(-1) / s, days.max(-1) / s, days.min(-1) / s, (days * eve).sum(-1) / (days.sum(-1) + 1e-3),
    ], -1)
    return seq.astype(np.float32), refh.astype(np.float32), summ.astype(np.float32), scale.astype(np.float32)


def load_inputs(raw: dict) -> dict:
    seq, refh, summ, scale = load_features(raw["wx"], raw["cal"], raw["hist_load"])
    B = seq.shape[0]
    arch = np.zeros((B, len(ARCHETYPES)), np.float32)
    a_idx = raw["arch_idx"].astype(int)
    arch[np.arange(B)[a_idx >= 0], a_idx[a_idx >= 0]] = 1
    flags = np.stack([np.asarray(raw.get(k, np.zeros(B)), np.float32) for k in HOUSE_FLAGS], -1)
    stat = np.concatenate([arch, np.stack([raw["has_batt"], (raw["lat"] - 45.5) / 1.5, (raw["lon"] - 25.0) / 2.5,
                                           np.log(scale)], -1), flags], -1).astype(np.float32)
    return {"seq": seq, "ref": refh, "summ": summ, "stat": stat, "scale": scale,
            "night": np.zeros(seq.shape[:2], bool)}


def site_summary(raw: dict) -> np.ndarray:
    """[B, 14, 8] = pv_summary + daily max voltage, trip fraction, SoC swing per kWh of throughput."""
    kwp = raw["kwp_recorded"].astype(np.float32)
    base = pv_summary(raw["hist_pv"], raw["pvclear"], raw["geom"], kwp)
    B = base.shape[0]
    if raw.get("hist_volt") is None:  # degraded mode: telemetry history not supplied
        extra = np.tile(np.array([0.8, 0.0, 1.0], np.float32), (B, HIST_DAYS, 1))
        return np.concatenate([base, extra], -1)
    v = _hist_days(raw["hist_volt"], HIST_DAYS).astype(np.float32)
    st = _hist_days(raw["hist_status"], HIST_DAYS)
    soc = _hist_days(raw["hist_soc"], HIST_DAYS).astype(np.float32)
    bt = _hist_days(raw["hist_batt"], HIST_DAYS).astype(np.float32)
    bkwh = np.maximum(raw["battery_kwh"].astype(np.float32), 1e-3)[:, None]
    thr = np.abs(bt).sum(-1) * 0.25 / bkwh  # full-cycle equivalents
    swing = (soc.max(-1) - soc.min(-1)) / (0.5 * thr + 0.05)
    swing = np.where(raw["has_batt"][:, None] > 0, swing, 1.0)
    extra = np.stack([(v.max(-1) - 230) / 10, (st == 3).mean(-1) * 10, np.clip(swing, 0, 5)], -1)
    return np.concatenate([base, np.nan_to_num(extra).astype(np.float32)], -1)


def site_arrays(raw: dict):
    """K4 reference (sentinel.kernels.site_features): float32 seq [B,H,18] (nan_to_num), summ [B,14,8]."""
    kwp = raw["kwp_recorded"].astype(np.float32)
    k = kwp[:, None]
    pv = raw["pv"].astype(np.float32)
    m1 = raw["m1"].astype(np.float32)
    q10, q50, q90 = m1[..., 0], m1[..., 1], m1[..., 2]
    cz, _, _, _ = _geom_feats(raw["geom"])
    st = raw["status"].astype(np.int64)
    bkw = np.maximum(raw["battery_kw"].astype(np.float32), 1.0)[:, None]
    lscale = raw["hist_load"].astype(np.float32).mean(1)[:, None] + 0.05
    prev = raw["hist_pv"][:, -1].astype(np.float32)
    seq = np.stack([
        pv / k, q10 / k, q50 / k, q90 / k, (pv - q50) / k,
        np.clip((pv - q50) / (q90 - q10 + 0.05 * k), -5, 5),
        raw["pvlib"].astype(np.float32) / k, cz,
        (raw["volt"].astype(np.float32) - 230) / 10,
        (st == 1).astype(np.float32), (st == 2).astype(np.float32), (st == 3).astype(np.float32),
        frozen_flags(pv, prev), raw["soc"].astype(np.float32), raw["batt"].astype(np.float32) / bkw,
        np.clip((raw["load"].astype(np.float32) - raw["m2"].astype(np.float32)) / lscale, -5, 5),
        np.clip(pv / (q50 + 0.05 * k), 0, 3),
        np.asarray(raw.get("valid", np.ones(pv.shape, bool)), np.float32),  # logger dropout (0-filled) != a fault
    ], -1)
    return np.nan_to_num(seq).astype(np.float32), site_summary(raw)


def site_inputs(raw: dict) -> dict:
    kwp = raw["kwp_recorded"].astype(np.float32)
    seq, summ = site_features(raw)
    stat = np.stack([raw["has_batt"], kwp / 10, raw["inverter_kw"] / kwp, raw["battery_kwh"] / 10,
                     raw["battery_kw"] / 5], -1).astype(np.float32)
    return {"seq": seq, "summ": summ, "stat": stat, "scale": kwp}


# --- gathering raw windows from a WindowSource -------------------------------------------------------------
def _site_static(src, sites):
    meta = {}
    for key in ("kwp_recorded", "tilt_recorded", "azimuth_recorded", "inverter_kw", "lat", "lon",
                "battery_kwh", "battery_kw"):
        meta[key] = src.site_meta(key)[sites]
    meta["has_batt"] = (meta["battery_kwh"] > 0).astype(np.float32)
    arch = [s.get("archetype", "base") for s in src.sites]
    meta["arch_idx"] = np.array([ARCHETYPES.index(a) if a in ARCHETYPES else -1 for a in arch])[sites]
    for k in HOUSE_FLAGS:
        meta[k] = house_flags(src.sites, k)[sites]
    return meta


def house_flags(site_rows, key: str) -> np.ndarray:
    """Household flags from the sites table (#11); derived from the archetype when the column is absent (synth)."""
    derive = {"has_ac": lambda r: r.get("archetype") == "ac", "has_ev": lambda r: r.get("archetype") == "ev"}
    out = []
    for r in site_rows:
        v = r.get(key)
        out.append(float(bool(v)) if v is not None else float(derive.get(key, lambda r: False)(r)))
    return np.array(out, np.float32)


def gather(src, sites: np.ndarray, starts: np.ndarray, kind: str, mode: str = "nowcast",
           m1: np.ndarray | None = None, m2: np.ndarray | None = None) -> dict:
    """Raw windows for target steps [start, start+96) with 14 days of history; kind = pv | load | site."""
    sites = np.asarray(sites, np.int64)
    starts = np.asarray(starts, np.int64)
    idx = starts[:, None] + np.arange(H)[None]
    loc = src.loc_idx()[sites]
    raw = _site_static(src, sites)

    def g(a, rows=sites, off=0, n=H, dt=np.float32):  # Rust window gather (sentinel.kernels), numpy fallback
        return window_gather(a, rows, starts, off, n, dt)

    hist = dict(off=-HIST_DAYS * H, n=HIST_DAYS * H)
    fc = mode == "forecast"
    raw["wx"] = g(src.array("weather_fc" if fc else "weather_act"), loc)
    raw["geom"] = g(src.array("geom"), loc)
    raw["cal"] = np.asarray(src.array("calendar")[idx], np.float32)
    raw["pvclear"] = g(src.array("pvclear_rec"))
    raw["pvlib"] = g(src.array("pvlib_rec_fc" if fc else "pvlib_rec"))
    raw["doy_sin"], raw["doy_cos"] = raw["cal"][:, 0, 6], raw["cal"][:, 0, 7]
    raw["fault"] = g(src.array("fault"), dt=None)
    raw["valid"] = g(src.array("valid"), dt=None) > 0 if src.has("valid") else np.ones(idx.shape, bool)
    if kind in ("pv", "site"):
        raw["hist_pv"] = g(src.array("pv_ac"), **hist)
    if kind == "pv":
        raw["y"] = g(src.array("pv_ac"))
        if "pv_healthy" in src.index["arrays"]:
            raw["y_healthy"] = g(src.array("pv_healthy"))
    if kind in ("load", "site"):
        raw["hist_load"] = g(src.array("load"), **hist)
    if kind == "load":
        raw["y"] = g(src.array("load"))
    if kind == "site":
        for k_src, k in (("pv_ac", "pv"), ("inv_v", "volt"), ("inv_status", "status"), ("soc", "soc"),
                         ("batt", "batt"), ("load", "load")):
            raw[k] = g(src.array(k_src), dt=None)
        for k_src, k in (("inv_v", "hist_volt"), ("inv_status", "hist_status"), ("soc", "hist_soc"),
                         ("batt", "hist_batt")):
            raw[k] = g(src.array(k_src), dt=None, **hist)
        raw["m1"], raw["m2"] = g(np.ascontiguousarray(m1)), g(np.ascontiguousarray(m2))
    return raw


def build(kind: str, raw: dict) -> dict:
    return {"pv": pv_inputs, "load": load_inputs, "site": site_inputs}[kind](raw)


# --- normalisation (train-only stats) ----------------------------------------------------------------------
def fit_stats(batches: list[dict]) -> dict:
    """Per-channel mean/std over training batches for seq, ref, summ, stat."""
    out = {}
    for key in ("seq", "ref", "summ", "stat"):
        if key not in batches[0]:
            continue
        x = np.concatenate([b[key].reshape(-1, b[key].shape[-1]) for b in batches], 0).astype(np.float64)
        mu, sd = x.mean(0), x.std(0)
        sd = np.where(sd < 1e-4, 1.0, sd)
        out[key] = {"mean": mu.astype(np.float32).tolist(), "std": sd.astype(np.float32).tolist()}
    return out


def normalise(x: dict, stats: dict) -> dict:
    y = dict(x)
    for key, st in stats.items():
        y[key] = ((x[key] - np.asarray(st["mean"], np.float32)) / np.asarray(st["std"], np.float32)).astype(np.float32)
    return y
