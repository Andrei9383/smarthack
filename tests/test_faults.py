"""PV fault injector (#12): one test per class checks its signature; labels match the contract; deterministic;
works on a PV-only (PVDAQ-like) series."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from wattsentry.contracts import schemas
from wattsentry.contracts.envelope import PvFault
from wattsentry.sim.faults import INV_OVERVOLTAGE, FaultConfig, inject_batch, inject_dataset
from wattsentry.sim.faults.inject import TELEMETRY_COLS
from wattsentry.sim.fleet import FleetConfig, generate

FLEET = FleetConfig(n_sites=30, n_feeders=2, n_congested=1, start_year=2024, end_year=2024, battery_share=0.7)
KINDS = [k.value for k in PvFault if k is not PvFault.HEALTHY]


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    out = generate(FLEET, out_dir=tmp_path_factory.mktemp("fleet"), plots=False, log=lambda *_: None)
    sites = pd.read_parquet(out / "sites.parquet")
    tel = pd.read_parquet(out / "telemetry" / "year=2024.parquet").sort_values(["site_id", "ts"])
    n = len(sites)
    T = len(tel) // n
    ts = pd.DatetimeIndex(tel.ts.to_numpy()[:T])
    arrays = {c: tel[c].to_numpy().reshape(n, T).astype(np.int16 if c == "inv_status" else np.float64)
              for c in TELEMETRY_COLS}
    return {"dir": out, "sites": sites.set_index("site_id").loc[tel.site_id.unique()].reset_index(), "ts": ts,
            "arrays": arrays}


def only(kind: str, **kw) -> FaultConfig:
    w = {k: (1.0 if k == kind else 0.0) for k in FaultConfig().weights}
    return dataclasses.replace(FaultConfig(), weights=w, faulty_share=0.3, **kw)


def run(fleet, kind):
    clean = {k: v.copy() for k, v in fleet["arrays"].items()}
    faulty = {k: v.copy() for k, v in fleet["arrays"].items()}
    labels = inject_batch(fleet["ts"], faulty, fleet["sites"], only(kind))
    assert labels, f"no {kind} episode"
    assert {l["fault"] for l in labels} == {kind}
    rows = {sid: i for i, sid in enumerate(fleet["sites"].site_id)}
    for l in labels:
        i0 = fleet["ts"].get_loc(l["onset"])
        i1 = fleet["ts"].get_loc(l["end"]) if pd.notna(l["end"]) else len(fleet["ts"])
        l["r"], l["i0"], l["i1"], l["p"] = rows[l["asset_id"]], i0, i1, json.loads(l["params"])
    return clean, faulty, labels


def window(a, l, col):
    return a[col][l["r"], l["i0"]:l["i1"]]


def test_soiling(fleet):
    clean, faulty, labels = run(fleet, "soiling")
    for l in labels:
        c, f = window(clean, l, "pv_ac_w"), window(faulty, l, "pv_ac_w")
        day = c > 50
        ratio = f[day] / c[day]
        half = len(ratio) // 2
        assert ratio[:half].mean() > ratio[half:].mean()  # loss grows with time
        assert ratio.min() >= 1 - l["p"]["max_loss"] - 1e-3  # params are rounded
        assert np.array_equal(clean["pv_ac_w"][l["r"], l["i1"]:], faulty["pv_ac_w"][l["r"], l["i1"]:]) or \
            l["i1"] == len(fleet["ts"]) or any(m is not l and m["r"] == l["r"] and m["i0"] >= l["i1"] for m in labels)


def test_shading(fleet):
    clean, faulty, labels = run(fleet, "shading")
    for l in labels:
        c, f = window(clean, l, "pv_ac_w"), window(faulty, l, "pv_ac_w")
        hit = f < c - 1e-6
        assert hit.any() and hit.mean() < 0.5  # only part of the day, only some sun positions
        assert np.allclose(f[hit] / c[hit], 1 - l["p"]["depth"], atol=1e-3)


def test_string_outage(fleet):
    clean, faulty, labels = run(fleet, "string_outage")
    for l in labels:
        c, f = window(clean, l, "pv_ac_w"), window(faulty, l, "pv_ac_w")
        ac = fleet["sites"].inverter_kw[l["r"]] * 1000
        ok = (c > 50) & (c < 0.95 * ac)
        assert np.allclose(f[ok] / c[ok], 1 - 1 / l["p"]["n_strings"], atol=1e-6)


def test_clipping(fleet):
    clean, faulty, labels = run(fleet, "clipping")
    for l in labels:
        c, f = window(clean, l, "pv_ac_w"), window(faulty, l, "pv_ac_w")
        lim = l["p"]["limit_w"]
        assert np.allclose(f, np.minimum(c, lim), atol=0.1) and (c > lim + 1).any()  # limit_w is rounded


def test_overvoltage_trip(fleet):
    clean, faulty, labels = run(fleet, "overvoltage_trip")
    for l in labels:
        st, v = window(faulty, l, "inv_status"), window(faulty, l, "inv_voltage_v")
        tripped = st == INV_OVERVOLTAGE
        assert tripped.sum() >= l["p"]["trips"] >= 1
        starts = np.flatnonzero(tripped & ~np.r_[False, tripped[:-1]])
        assert (v[starts] > l["p"]["trip_setting_v"] - 0.05).all()  # trip_setting_v is rounded to 0.1 V
        assert (window(faulty, l, "pv_ac_w")[tripped] <= window(clean, l, "pv_ac_w")[tripped] + 1e-6).all()
        assert (st[~tripped] == 0).all()


def test_battery_fade(fleet):
    clean, faulty, labels = run(fleet, "battery_fade")
    for l in labels:
        cyc_c = np.abs(window(clean, l, "batt_w")).sum()
        cyc_f = np.abs(window(faulty, l, "batt_w")).sum()
        assert cyc_f <= cyc_c + 1  # a fade starting while the battery idles (winter) shows later
        if cyc_c > 50_000:  # the battery really cycled in this window
            assert cyc_f < cyc_c
        soc = window(faulty, l, "soc")
        assert np.nanmin(soc) >= -1e-6 and np.nanmax(soc) <= 1 + 1e-6
    # physics stays consistent: grid = pv - load - batt everywhere
    f = faulty
    assert np.allclose(f["grid_w"], f["pv_ac_w"] - f["load_w"] - f["batt_w"], atol=2.0)
    # before the first onset of each site, nothing changed
    for l in labels:
        first = min(m["i0"] for m in labels if m["r"] == l["r"])
        assert np.allclose(faulty["batt_w"][l["r"], :first], clean["batt_w"][l["r"], :first])


def test_sensor_fault(fleet):
    clean, faulty, labels = run(fleet, "sensor_fault")
    modes = set()
    for l in labels:
        m = l["p"]["mode"]
        modes.add(m)
        c, f = window(clean, l, "pv_ac_w"), window(faulty, l, "pv_ac_w")
        if m == "stuck":
            assert np.all(f == f[0])
        elif m == "dropout":
            assert np.isnan(f).all() and np.isnan(window(faulty, l, "grid_w")).all()
        else:
            assert (f > c * 2.9).any() and (f >= c - 1e-6).all()
        if m != "dropout":  # physics untouched: battery and meter still see the real PV
            assert np.array_equal(window(faulty, l, "grid_w"), window(clean, l, "grid_w"))
    assert modes


def test_labels_contract_and_determinism(fleet, tmp_path):
    a = inject_dataset(fleet["dir"], FaultConfig(seed=3), out_dir=tmp_path / "a", log=lambda *_: None)
    b = inject_dataset(fleet["dir"], FaultConfig(seed=3), out_dir=tmp_path / "b", log=lambda *_: None)
    la, lb = pd.read_parquet(a / "fault_labels.parquet"), pd.read_parquet(b / "fault_labels.parquet")
    pd.testing.assert_frame_equal(la, lb)
    assert schemas.check_columns("fault_labels", la.columns) == []
    assert set(la.fault) <= set(KINDS) and (la.asset_kind == "site").all()
    assert la.severity.between(0, 1).all() and (la.onset.dt.tz is not None)
    tel = pd.read_parquet(a / "telemetry" / "year=2024.parquet")
    assert schemas.check_columns("telemetry", tel.columns) == []
    # episodes on one site never overlap
    for _, g in la.groupby("asset_id"):
        g = g.sort_values("onset")
        assert (g.onset.iloc[1:].to_numpy() >= g.end.iloc[:-1].fillna(pd.Timestamp.max.tz_localize("UTC")).to_numpy()).all()


def test_pv_only_series():
    """PVDAQ-like input: only ts, site_id, pv_ac_w; battery/voltage classes are skipped automatically."""
    from wattsentry.sim.faults import inject

    ts = pd.date_range("2024-01-01", "2024-12-31", freq="15min", tz="UTC")
    h = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    pv = np.clip(np.sin(np.pi * (h - 5) / 14), 0, None) * 3000
    tel = pd.DataFrame({"ts": ts, "site_id": "pvdaq_1430", "pv_ac_w": pv})
    sites = pd.DataFrame({"site_id": ["pvdaq_1430"], "lat": [39.74], "lon": [-105.18], "inverter_kw": [3.5]})
    out, lab = inject(tel, sites, dataclasses.replace(FaultConfig(), faulty_share=0.3))
    assert len(out) == len(tel) and len(lab) > 0
    assert not set(lab.fault) & {"battery_fade", "overvoltage_trip"}
    assert (out.pv_ac_w.fillna(0) != tel.pv_ac_w).any()
