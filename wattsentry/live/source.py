"""Where readings come from, one 15-min step at a time.

`SimulatedSource.readings(ts)` returns only the step starting at `ts` (unix s, UTC), so a consumer calling it tick by
tick never sees the future. PV and load come from the world's telemetry (`observed_site_day`, stale readings ->
None); the battery runs the self-consumption rule here, so its state of charge carries across ticks and restarts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from wattsentry.contracts import BatterySpec
from wattsentry.contracts.opt import self_consumption_step

STEP_S = 900
LABEL = "synthetic RO, simulated readings"


@dataclass
class Reading:
    site_id: str
    feeder_id: str
    pv_kw: Optional[float]
    load_kw: Optional[float]
    batt_kw: Optional[float]
    soc: Optional[float]
    net_kw: Optional[float]  # pv - load - batt, + = export

    def row(self) -> dict[str, Any]:
        return asdict(self)


class SimulatedSource:
    """A World replayed one step at a time. `feeders` = feeder labels to keep (default: all)."""

    label = LABEL

    def __init__(self, world: Any, feeders: Optional[Sequence[str]] = None, initial_soc: float = 0.5):
        self.world = world
        keep = list(feeders) if feeders else list(world.feeders)
        missing = [f for f in keep if f not in world.feeders]
        if missing:
            raise KeyError(f"feeders not in the world: {missing} (have {sorted(world.feeders)})")
        self._feeders = {f: world.feeders[f] for f in keep}
        self._sites = {s: world.sites[s] for f in keep for s in world.feeders[f].site_ids if s in world.sites}
        self.e: dict[str, float] = {}
        for sid, site in self._sites.items():
            b: Optional[BatterySpec] = site.battery
            if b is not None and b.capacity_kwh > 0:
                self.e[sid] = min(b.e_max, max(b.soc_min * b.capacity_kwh, initial_soc * b.capacity_kwh))
        self.last_ts: Optional[int] = None

    def sites(self) -> Mapping[str, Any]:
        return self._sites

    def feeders(self) -> Mapping[str, Any]:
        return self._feeders

    def span(self) -> tuple[int, int]:
        return self.world.day_ts(0), self.world.day_ts(self.world.days)

    def restore(self, soc: Mapping[str, Optional[float]]) -> None:
        """Battery state after a restart: the last stored soc per site."""
        for sid, s in soc.items():
            b = self._sites.get(sid).battery if sid in self._sites else None
            if s is not None and b is not None and b.capacity_kwh > 0:
                self.e[sid] = float(s) * b.capacity_kwh

    def soc(self, sid: str) -> Optional[float]:
        b = self._sites[sid].battery
        return self.e[sid] / b.capacity_kwh if sid in self.e else None

    def readings(self, ts: int) -> dict[str, Reading]:
        lo, hi = self.span()
        if not lo <= ts < hi:
            return {}
        if self.last_ts is not None and ts <= self.last_ts:
            raise ValueError(f"readings asked out of order: {ts} after {self.last_ts}")
        self.last_ts = ts
        d = (ts - lo) // 86400
        k = (ts - self.world.day_ts(d)) // STEP_S
        out: dict[str, Reading] = {}
        for sid, site in self._sites.items():
            day = self.world.observed_site_day(sid, d)
            fid = site.feeder_id
            pv, load = float(day.pv_kw[k]), float(day.load_kw[k])
            e = self.e.get(sid, 0.0)
            batt, e_next, grid = self_consumption_step(site.battery, e, pv, load)
            if sid in self.e:  # the battery keeps running whether or not the logger reports
                self.e[sid] = e_next
            if day.telemetry_age_s[k] > 0:  # stale logger: nothing arrived for this step
                out[sid] = Reading(sid, fid, None, None, None, None, None)
                continue
            out[sid] = Reading(sid, fid, round(pv, 4), round(load, 4), round(batt, 4), self.soc(sid), round(grid, 4))
        return out
