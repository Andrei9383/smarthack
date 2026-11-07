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
LOAD_STATIC = tuple(f"arch_{a}" for a in ARCHETYPES) + ("has_batt", "lat", "lon", "log_scale")
SITE_SEQ = ("pv", "q10", "q50", "q90", "res", "zres", "pvlib", "cos_zen", "volt", "st_prod", "st_limit",
            "st_trip", "frozen", "soc", "batt", "load_res", "ratio")
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


def pv_inputs(raw: dict) -> dict:
    kwp = raw["kwp_recorded"].astype(np.float32)
    k = kwp[:, None]
    wx = raw["wx"].astype(np.float32)
    cz, sa, ca, cs = _geom_feats(raw["geom"])
    pvc = raw["pvclear"].astype(np.float32) / k
    ref = _hist_days(raw["hist_pv"], REF_DAYS).astype(np.float32) / k[:, :, None]
    ref_mean, ref_max = ref.mean(1), ref.max(1)
    seq = np.stack([
        wx[..., 0] / 1000, wx[..., 1] / 1000, wx[..., 2] / 500, wx[..., 3] / 20, wx[..., 4] / 100,
        wx[..., 5] / 10, wx[..., 6] / 10, cz, sa, ca, cs,
        raw["pvlib"].astype(np.float32) / k, pvc, ref_mean, ref_max, ref[:, -1],
        np.clip(ref_max / np.maximum(pvc, 0.02), 0, 2) * (pvc > 0.01),
    ], -1)
    B = seq.shape[0]
    refh = ref.reshape(B, REF_DAYS * 24, 4).mean(-1)
    summ = pv_summary(raw["hist_pv"], raw["pvclear"], raw["geom"], kwp)
    azr = np.deg2rad(raw["azimuth_recorded"].astype(np.float32))
    stat = np.stack([
        (raw["lat"] - 45.5) / 1.5, (raw["lon"] - 25.0) / 2.5, kwp / 10, raw["tilt_recorded"] / 45,
        np.sin(azr), np.cos(azr), raw["inverter_kw"] / kwp, raw["has_batt"],
        raw["doy_sin"], raw["doy_cos"],
    ], -1).astype(np.float32)
    night = (cz <= 0.0) & (cs <= 0.0)
    return {"seq": seq.astype(np.float32), "ref": refh.astype(np.float32), "summ": summ, "stat": stat,
            "scale": kwp, "night": night}


def load_inputs(raw: dict) -> dict:
    hl = raw["hist_load"].astype(np.float32)
    scale = hl.mean(1) + 0.05
    s = scale[:, None]
    days = _hist_days(hl, HIST_DAYS)
    ref = days[:, -REF_DAYS:] / s[:, :, None]
    wx = raw["wx"].astype(np.float32)
    cal = raw["cal"].astype(np.float32)
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
    arch = np.zeros((B, len(ARCHETYPES)), np.float32)
    a_idx = raw["arch_idx"].astype(int)
    arch[np.arange(B)[a_idx >= 0], a_idx[a_idx >= 0]] = 1
    stat = np.concatenate([arch, np.stack([raw["has_batt"], (raw["lat"] - 45.5) / 1.5, (raw["lon"] - 25.0) / 2.5,
                                           np.log(scale)], -1)], -1).astype(np.float32)
    return {"seq": seq.astype(np.float32), "ref": refh.astype(np.float32), "summ": summ.astype(np.float32),
            "stat": stat, "scale": scale.astype(np.float32), "night": np.zeros(seq.shape[:2], bool)}


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


def site_inputs(raw: dict) -> dict:
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
    ], -1)
    stat = np.stack([raw["has_batt"], kwp / 10, raw["inverter_kw"] / kwp, raw["battery_kwh"] / 10,
                     raw["battery_kw"] / 5], -1).astype(np.float32)
    return {"seq": np.nan_to_num(seq).astype(np.float32), "summ": site_summary(raw), "stat": stat, "scale": kwp}


# --- gathering raw windows from a WindowSource -------------------------------------------------------------
def _site_static(src, sites):
    meta = {}
    for key in ("kwp_recorded", "tilt_recorded", "azimuth_recorded", "inverter_kw", "lat", "lon",
                "battery_kwh", "battery_kw"):
        meta[key] = src.site_meta(key)[sites]
    meta["has_batt"] = (meta["battery_kwh"] > 0).astype(np.float32)
    arch = [s.get("archetype", "base") for s in src.sites]
    meta["arch_idx"] = np.array([ARCHETYPES.index(a) if a in ARCHETYPES else -1 for a in arch])[sites]
    return meta


def gather(src, sites: np.ndarray, starts: np.ndarray, kind: str, mode: str = "nowcast",
           m1: np.ndarray | None = None, m2: np.ndarray | None = None) -> dict:
    """Raw windows for target steps [start, start+96) with 14 days of history; kind = pv | load | site."""
    sites = np.asarray(sites, np.int64)
    starts = np.asarray(starts, np.int64)
    idx = starts[:, None] + np.arange(H)[None]
    hidx = starts[:, None] + np.arange(-HIST_DAYS * H, 0)[None]
    s2 = sites[:, None]
    loc = src.loc_idx()[sites][:, None]
    raw = _site_static(src, sites)
    wname = "weather_fc" if mode == "forecast" else "weather_act"
    raw["wx"] = np.asarray(src.array(wname)[loc, idx], np.float32)
    raw["geom"] = np.asarray(src.array("geom")[loc, idx], np.float32)
    raw["cal"] = np.asarray(src.array("calendar")[idx], np.float32)
    raw["pvclear"] = np.asarray(src.array("pvclear_rec")[s2, idx], np.float32)
    raw["pvlib"] = np.asarray(src.array("pvlib_rec_fc" if mode == "forecast" else "pvlib_rec")[s2, idx], np.float32)
    raw["doy_sin"], raw["doy_cos"] = raw["cal"][:, 0, 6], raw["cal"][:, 0, 7]
    raw["fault"] = np.asarray(src.array("fault")[s2, idx])
    if kind in ("pv", "site"):
        raw["hist_pv"] = np.asarray(src.array("pv_ac")[s2, hidx], np.float32)
    if kind == "pv":
        raw["y"] = np.asarray(src.array("pv_ac")[s2, idx], np.float32)
        if "pv_healthy" in src.index["arrays"]:
            raw["y_healthy"] = np.asarray(src.array("pv_healthy")[s2, idx], np.float32)
    if kind in ("load", "site"):
        raw["hist_load"] = np.asarray(src.array("load")[s2, hidx], np.float32)
    if kind == "load":
        raw["y"] = np.asarray(src.array("load")[s2, idx], np.float32)
    if kind == "site":
        for k_src, k in (("pv_ac", "pv"), ("inv_v", "volt"), ("inv_status", "status"), ("soc", "soc"),
                         ("batt", "batt"), ("load", "load")):
            raw[k] = np.asarray(src.array(k_src)[s2, idx])
        for k_src, k in (("inv_v", "hist_volt"), ("inv_status", "hist_status"), ("soc", "hist_soc"),
                         ("batt", "hist_batt")):
            raw[k] = np.asarray(src.array(k_src)[s2, hidx])
        raw["m1"] = np.asarray(m1[s2, idx], np.float32)
        raw["m2"] = np.asarray(m2[s2, idx], np.float32)
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
