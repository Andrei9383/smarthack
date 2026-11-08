"""Wind fault injector: every class injected with an exact onset, and the labels match the telemetry.

Each test enables ONE kind at a high rate on a small ERA5-driven world, re-renders the clean telemetry with the
same noise, and checks that (a) nothing changes before a turbine's first onset, (b) the class signature is there
from the onset on, and (c) the fault_labels rows follow the fault_labels contract + the WindAnomaly classes.
"""

import json

import numpy as np
import polars as pl
import pytest

from wattsentry.contracts.envelope import WindAnomaly
from wattsentry.contracts.schemas import check_columns
from wattsentry.model.wind.data import CLASS_INDEX, _ts_index, fault_labels_table, labels_from_episodes
from wattsentry.sim.wind.config import WindSimConfig
from wattsentry.sim.wind.generate import generate_clean
from wattsentry.sim.wind_faults import KIND_CLASS, KINDS, FaultConfig, inject

CFG = WindSimConfig(start="2024-01-01", end="2024-02-15", max_turbines_per_park=5)
BASE = FaultConfig()


@pytest.fixture(scope="module")
def clean():
    return generate_clean(CFG, log=lambda *a: None)


@pytest.fixture(scope="module")
def base(clean):
    return clean.render()


def _run(clean, kind, **rates):
    res = inject(clean, BASE.only(kind, **rates))
    eps = [e | {"p": json.loads(e["params"])} for e in res.episodes]
    eps = [e for e in eps if e["p"]["kind"] == kind]  # (overheating also emits the failure stops that follow)
    assert eps, f"no {kind} episodes injected"
    assert all(e["fault"] == KIND_CLASS[kind] for e in eps)
    return res, eps


def _span(clean, e):
    s = _ts_index(clean.ts, e["onset"])
    en = clean.T if e["end"] is None else _ts_index(clean.ts, e["end"])
    assert clean.ts[s] == np.datetime64(e["onset"].rstrip("Z"), "m")  # onset sits exactly on the 15-min grid
    return s, en


def _unchanged_before_first_onset(clean, res, base, eps, sensors=None):
    col = {t["turbine_id"]: j for j, t in enumerate(clean.turbines)}
    first = {}
    for e in eps:
        j = col[e["asset_id"]]
        first[j] = min(first.get(j, clean.T), _span(clean, e)[0])
    for k in sensors or base:
        a, b = res.sensors[k], base[k]
        for j in range(clean.N):
            f = first.get(j, clean.T)
            np.testing.assert_array_equal(a[:f, j], b[:f, j], err_msg=f"{k} changed before onset on turbine {j}")
    return col


def test_kinds_map_to_contract_classes():
    assert set(KIND_CLASS.values()) == {a.value for a in WindAnomaly} - {"normal"}


def test_fault_stop(clean, base):
    res, eps = _run(clean, "fault_stop", rate_stop=40.0)
    col = _unchanged_before_first_onset(clean, res, base, eps)
    for e in eps:
        s, en = _span(clean, e)
        p = res.sensors["act_pwt"][s:en, col[e["asset_id"]]]
        assert (p < 0).all(), "a stopped turbine draws auxiliary power from the onset to the end"


def test_curtailment_park_wide_cap(clean, base):
    res, eps = _run(clean, "curtailment", rate_curtail=150.0)
    col = _unchanged_before_first_onset(clean, res, base, eps)
    events = {}
    for e in eps:
        events.setdefault(e["p"]["park_event"], []).append(e)
    for ev in events.values():
        assert len({(e["onset"], e["end"]) for e in ev}) == 1  # one setpoint, same window for the whole park
        park = ev[0]["asset_id"].split(".")[0]
        assert all(e["asset_id"].startswith(park + ".") for e in ev)
    n_rows = sum(len(ev) for ev in events.values())
    n_slots = sum(sum(t["park_id"] == ev[0]["asset_id"].split(".")[0] for t in clean.turbines)
                  for ev in events.values())
    assert n_rows / n_slots > 0.85  # ~95 % of the park's turbines receive each setpoint
    n_cut = 0
    spans = [(col[e["asset_id"]], *_span(clean, e)) for e in eps]
    for e, (j, s, en) in zip(eps, spans):
        cap = e["p"]["cap_frac"] * clean.rated[j]
        assert (res.sensors["act_pwt"][s:en, j] <= cap + 0.02 * clean.rated[j]).all()
        if any(jj == j and ss < en and s < ee and (ss, ee) != (s, en) for jj, ss, ee in spans):
            continue  # overlapping setpoints: the lower one does the cutting
        bites = clean.avail[s:en, j] > cap + 0.02 * clean.rated[j]
        assert e["p"]["power_cut"] == bool(bites.any()) and e["p"]["cut_steps"] == int(bites.sum())
        if e["p"]["power_cut"]:
            n_cut += 1
            cs = _ts_index(clean.ts, e["p"]["cut_onset"])
            assert bites[cs - s] and not bites[: cs - s].any()
            assert res.sensors["act_pwt"][cs, j] < base["act_pwt"][cs, j]
    assert 0 < n_cut < len(eps) or n_cut == len(eps)


def test_icing_follows_real_era5(clean, base):
    res, eps = _run(clean, "icing", p_icing_spell=1.0)
    col = _unchanged_before_first_onset(clean, res, base, eps, sensors=["act_pwt"])
    w = clean.weather.q15
    pidx = {p: i for i, p in enumerate(clean.park_ids)}
    for e in eps:
        j = col[e["asset_id"]]
        s, en = _span(clean, e)
        p = pidx[clean.turbines[j]["park_id"]]
        assert w["temp_c"][s, p] < BASE.icing_temp_c
        assert w["rh_pct"][s, p] >= BASE.icing_rh or w["precip_mm"][s, p] > BASE.icing_precip_mm
        a, b = res.sensors["act_pwt"][s:en, j], base["act_pwt"][s:en, j]
        assert (a <= b + 1e-3).all()
        if e["p"]["power_cut"]:
            assert (a < b - 1e-3).any()
    assert any(e["p"]["power_cut"] for e in eps)


@pytest.mark.parametrize("kind,rates", [("yaw", {"rate_yaw": 8.0}), ("power_curve", {"rate_power_curve": 8.0})])
def test_underperformance(clean, base, kind, rates):
    res, eps = _run(clean, kind, **rates)
    col = _unchanged_before_first_onset(clean, res, base, eps, sensors=["act_pwt"])
    for e in eps:
        j = col[e["asset_id"]]
        s, en = _span(clean, e)
        a, b = res.sensors["act_pwt"][s:en, j], base["act_pwt"][s:en, j]
        assert (a <= b + 1e-3).all()
        producing = b > 0.1 * clean.rated[j]
        if producing.sum() > 10:
            assert (a[producing] < b[producing]).mean() > 0.5


def test_overheating_drift_then_failure(clean, base):
    res, eps = _run(clean, "overheating", rate_overheat=8.0)
    col = _unchanged_before_first_onset(clean, res, base, eps)
    stops = [(col[e["asset_id"]], *_span(clean, e)) for e in res.episodes if e["fault"] == "fault_stop"]
    checked = 0
    for e in eps:
        j = col[e["asset_id"]]
        s, en = _span(clean, e)
        if any(jj == j and ss < en and s < ee for jj, ss, ee in stops):
            continue  # an earlier failure stop overlaps: the turbine is cold, not a clean drift check
        checked += 1
        comp = e["p"]["component"]
        assert comp in ("gearbox_t", "gen1_t", "gen2_t")
        d = res.sensors[comp][s:en, j] - base[comp][s:en, j]
        assert d.min() > -1e-3 and d[-8:].mean() > 0.3 * e["p"]["amp_c"]  # residual drifts up to the end
        assert d[: max(1, len(d) // 10)].mean() < d[-8:].mean()
    assert checked >= 3
    stops = [e for e in res.episodes if e["fault"] == "fault_stop"]
    assert stops and all(json.loads(e["params"])["cause"].startswith("failure_after_overheating") for e in stops)


def test_anemometer_fault_power_untouched(clean, base):
    res, eps = _run(clean, "anemometer", rate_anem=12.0)
    np.testing.assert_array_equal(res.sensors["act_pwt"], base["act_pwt"])
    col = _unchanged_before_first_onset(clean, res, base, eps)
    modes = set()
    spans = [(col[e["asset_id"]], *_span(clean, e)) for e in eps]
    for e, (j, s, en) in zip(eps, spans):
        modes.add(e["p"]["mode"])
        a, b = res.sensors["wd_spd"][s:en, j], base["wd_spd"][s:en, j]
        assert a[0] != b[0]  # visible from the onset step
        alone = not any(jj == j and ss < en and s < ee and (ss, ee) != (s, en) for jj, ss, ee in spans)
        if e["p"]["mode"] == "stuck" and alone:
            assert (a == a[0]).all()
    assert modes == {"bias", "stuck", "erratic"}


def test_comm_gaps_and_error_codes(clean, base):
    res, eps = _run(clean, "data_gap", rate_gap_short=60.0, p_gap_coded=0.5)
    col = _unchanged_before_first_onset(clean, res, base, eps)
    gap = np.zeros((clean.T, clean.N), bool)
    for e in eps:
        j = col[e["asset_id"]]
        s, en = _span(clean, e)
        gap[s:en, j] = True
        for k in res.sensors:
            assert np.isnan(res.sensors[k][s:en, j]).all()
    for e in eps:  # codes are integral and present on coded gaps (a later overlapping gap may rewrite them)
        if e["p"]["coded"]:
            s, en = _span(clean, e)
            assert (res.err[s:en, col[e["asset_id"]]] != 0).any()
    assert (res.err[~gap] == 0).all() and any(e["p"]["coded"] for e in eps)
    for k in res.sensors:  # outside gaps nothing changed
        np.testing.assert_array_equal(res.sensors[k][~gap], base[k][~gap])


def test_all_kinds_labels_contract_and_modes(clean):
    res = inject(clean, FaultConfig(rate_curtail=150.0, p_icing_spell=1.0, rate_stop=20.0))
    kinds = {json.loads(e["params"])["kind"] for e in res.episodes}
    assert {"fault_stop", "curtailment", "icing", "data_gap"} <= kinds <= set(KINDS)
    tids = [t["turbine_id"] for t in clean.turbines]
    data = clean.to_wind_data(res.sensors, res.err, labels_from_episodes(res.episodes, clean.ts, tids),
                              res.episodes)
    fl = fault_labels_table(data)
    assert not check_columns("fault_labels", fl.columns)
    assert set(fl["fault"]) <= {a.value for a in WindAnomaly} and set(fl["asset_kind"]) == {"turbine"}
    assert (fl.drop_nulls("end").select((pl.col("end") > pl.col("onset")).all())).item()
    eff = labels_from_episodes(res.episodes, clean.ts, tids, mode="effective")
    whole = labels_from_episodes(res.episodes, clean.ts, tids, mode="episode")
    c = CLASS_INDEX["curtailment"]
    assert (whole == c).sum() > (eff == c).sum() > 0  # whole episodes include caps that never bit
    # the modes differ only where a curtailment / icing episode is involved
    ci = [CLASS_INDEX["curtailment"], CLASS_INDEX["icing"]]
    differ = eff != whole
    assert np.isin(eff[differ], ci).any() or np.isin(whole[differ], ci).all()
    assert (np.isin(eff[differ], ci) | np.isin(whole[differ], ci)).all()


def test_rates_are_configurable_and_streams_independent(clean):
    a = inject(clean, BASE.only("fault_stop", "data_gap"))
    b = inject(clean, BASE.only("fault_stop", "data_gap", rate_gap_short=0.0, rate_gap_long=0.0))
    stops = lambda r: [e for e in r.episodes if e["fault"] == "fault_stop"]  # noqa: E731
    assert stops(a) == stops(b)  # changing one kind's rate never moves another kind's episodes
    assert not [e for e in b.episodes if e["fault"] == "data_gap"]
