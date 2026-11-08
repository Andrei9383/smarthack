"""Grid-twin (#13): finds wrong GIS feeder / phase records from voltages; tau and G only ever tighten the limit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wattsentry.gridtwin.run import score
from wattsentry.gridtwin.twin import effective_limit, feeder_specs, load_voltages, run_twin, to_geojson
from wattsentry.sim.fleet import FleetConfig, generate

CFG = FleetConfig(n_sites=150, n_feeders=4, n_congested=1, start_year=2024, end_year=2024,
                  gis_wrong_phase_share=0.12, gis_wrong_feeder_share=0.05)


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    fleet = generate(CFG, out_dir=tmp_path_factory.mktemp("fleet"), plots=False, log=lambda *_: None)
    sites, v = load_voltages(fleet, "2024-05-01", 60)
    feeders = pd.read_parquet(fleet / "feeders.parquet")
    return sites, feeders, run_twin(sites, v, feeders)


def test_recovers_wrong_records(twin):
    sites, _, res = twin
    m = score(res, sites)
    assert m["n_wrong_feeder"] > 0 and m["n_wrong_phase"] > 5
    assert m["feeder_recovery"] >= 0.8 and m["feeder_precision"] >= 0.8
    assert m["phase_recovery"] >= 0.5 and m["phase_precision"] >= 0.4
    assert m["n_flags"] < 0.35 * len(sites)  # review requests, not a reshuffle


def test_limits_only_tighten(twin):
    _, feeders, res = twin
    f = res.feeders
    assert f.tau.between(0, 1).all()
    assert (f.g_kw <= f.g_max_kw + 1e-9).all() and (f.g_kw >= f.g_conservative_kw - 1e-9).all()
    specs = feeder_specs(res)
    assert set(specs) == set(feeders.feeder_id)
    assert all(abs(specs[r.feeder_id].g_kw - r.g_kw) < 1e-9 for r in f.itertuples())


def test_geojson_and_messages(twin, tmp_path):
    _, _, res = twin
    gj = to_geojson(res)
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == len(res.flags)
    for ft in gj["features"]:
        lon, lat = ft["geometry"]["coordinates"]
        assert 20 < lon < 30 and 43 < lat < 49  # Romania
        p = ft["properties"]
        assert p["status"] == "review_requested" and 0 <= p["score"] <= 1
        assert p["message"].startswith("likely L") and "GIS says" in p["message"]
    json.dumps(gj)
    (tmp_path / "feeders.json").write_text(json.dumps(res.feeders.to_dict("records")))
    fid = res.feeders.feeder_id.iloc[0]
    assert effective_limit(fid, tmp_path) == pytest.approx(float(res.feeders.g_kw.iloc[0]))
