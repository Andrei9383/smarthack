"""One explainer plot per fault class: clean vs faulty series around a real episode (for the slides)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TITLES = {
    "soiling": "Soiling: slow loss, reset by rain", "shading": "New shading: loss in one sun sector",
    "string_outage": "String outage: 1/N step down", "clipping": "Inverter derating: capped output",
    "overvoltage_trip": "Overvoltage trips: drops to 0 at high voltage",
    "battery_fade": "Battery fade: SoC swings wider for the same energy", "sensor_fault": "Sensor fault: stuck / dropout / spikes",
}


def _series(d: Path, site: str, t0, t1) -> pd.DataFrame:
    parts = [pd.read_parquet(f, filters=[("site_id", "==", site), ("ts", ">=", t0), ("ts", "<", t1)])
             for f in sorted((d / "telemetry").glob("year=*.parquet"))]
    return pd.concat(parts).sort_values("ts").set_index("ts")


def explainer_plots(fleet_dir: Path, fault_dir: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(fault_dir / "fault_labels.parquet")
    for kind, title in TITLES.items():
        cand = lab[lab.fault == kind]
        if kind == "battery_fade":
            cand = cand[cand.params.str.contains('"mode": "fade"')]
        nice = cand[cand.end.notna() & cand.onset.dt.month.between(4, 9)]
        cand = (nice if len(nice) else cand).sort_values("severity", ascending=False)
        if cand.empty:
            continue
        # a clear but not extreme example; for fade the strongest one (on big batteries it barely shows)
        ep = cand.iloc[0] if kind == "battery_fade" else cand.iloc[len(cand) // 4]
        long = kind in ("soiling", "battery_fade")
        end = ep.end if pd.notna(ep.end) else ep.onset + pd.Timedelta(days=30)
        t0 = ep.onset - pd.Timedelta(days=5 if long else 2)
        span = pd.Timedelta(days=120 if kind == "battery_fade" else 60)
        t1 = min(end + pd.Timedelta(days=5), ep.onset + span) if long else ep.onset + pd.Timedelta(days=4)
        a, b = _series(fleet_dir, ep.asset_id, t0, t1), _series(fault_dir, ep.asset_id, t0, t1)
        fig, ax = plt.subplots(figsize=(9, 3.4))
        tl = a.index.tz_convert("Europe/Bucharest")
        if kind == "battery_fade":
            ax.plot(tl, b.soc, color="#c92a2a", lw=0.9, alpha=0.7, label="SoC reported (faulty)")
            ax.plot(tl, a.soc, color="#212529", lw=0.8, label="SoC clean")  # on top: it swings less
            ax2 = ax.twinx()
            ax2.plot(tl, (b.batt_w.abs().rolling(96).sum() / 4000), color="#3b5bdb", lw=1.2,
                     label="kWh cycled / day, faulty")
            ax2.plot(tl, (a.batt_w.abs().rolling(96).sum() / 4000), color="#212529", lw=1, ls="--",
                     label="kWh cycled / day, clean")
            ax2.legend(fontsize=7, loc="upper right")
            ax2.set_ylabel("kWh / day")
        elif long:
            ax.plot(tl, a.pv_ac_w.rolling(96).sum() / 4000, color="#adb5bd", lw=1.4, label="clean")
            ax.plot(tl, b.pv_ac_w.rolling(96).sum() / 4000, color="#c92a2a", lw=1.2, label="faulty")
            ax.set_ylabel("PV kWh / day (rolling)")
        else:
            ax.plot(tl, a.pv_ac_w / 1000, color="#adb5bd", lw=1.4, label="clean")
            ax.plot(tl, b.pv_ac_w / 1000, color="#c92a2a", lw=1.0, label="faulty")
            ax.set_ylabel("PV kW")
            if kind == "overvoltage_trip":
                ax2 = ax.twinx()
                ax2.plot(tl, b.inv_voltage_v, color="#ae3ec9", lw=0.8, alpha=0.6)
                ax2.axhline(json.loads(ep.params)["trip_setting_v"], color="#ae3ec9", ls=":", lw=1)
                ax2.set_ylabel("V")
        ax.axvline(ep.onset.tz_convert("Europe/Bucharest"), color="#2b8a3e", ls="--", lw=1, label="onset (label)")
        ax.set_title(f"{title} — {ep.asset_id}, severity {ep.severity:.2f}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out / f"{kind}.png", dpi=100)
        plt.close(fig)
