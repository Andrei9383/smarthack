"""Sanity plots: one panel per feeder (sunniest summer day vs limits; export duration curve), and one battery site."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def sanity_plots(out: Path, feeders: pd.DataFrame, ts: pd.DatetimeIndex, net_w: np.ndarray) -> None:
    pdir = out / "plots"
    pdir.mkdir(exist_ok=True)
    nf = len(feeders)
    cols = 5
    rows = int(np.ceil(nf / cols))
    local = ts.tz_convert("Europe/Bucharest")
    day = (local.normalize() - local[0].normalize()).days.to_numpy()
    summer = np.isin(local.month.to_numpy(), [5, 6, 7])

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), sharex=True, squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= nf:
            ax.axis("off")
            continue
        f = feeders.iloc[k]
        x = net_w[k]
        best = day[summer][np.argmax(np.where(summer, x, -np.inf)[summer])]
        m = day == best
        ax.plot(local[m].hour + local[m].minute / 60, x[m] / 1000, color="#3b5bdb", lw=1.4, label="net export")
        ax.axhline(f.g_max_kw, color="#c92a2a", lw=1, label="G_max")
        ax.axhline(f.g_conservative_kw, color="#b7791f", lw=1, ls="--", label="G_conservative")
        ax.set_title(f"{f.feeder_id} {f.locality} ({f.n_sites} PV)", fontsize=9)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("local hour, peak summer day")
    fig.supylabel("kW")
    fig.tight_layout()
    fig.savefig(pdir / "feeders_peak_day.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in range(nf):
        f = feeders.iloc[k]
        dur = np.sort(net_w[k] / 1000 / f.g_max_kw)[::-1][: len(ts) // 4]
        ax.plot(np.arange(len(dur)) / 4, dur, lw=1, label=f.feeder_id)
    ax.axhline(1, color="#c92a2a", lw=1)
    ax.set_xlabel("hours (top quarter of the period)")
    ax.set_ylabel("net export / G_max")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(pdir / "feeders_duration.png", dpi=110)
    plt.close(fig)

    sites = pd.read_parquet(out / "sites.parquet")
    site = sites[sites.battery_kwh > 0].iloc[0].site_id
    first = sorted((out / "telemetry").glob("*.parquet"))[-1]
    tel = pd.read_parquet(first, filters=[("site_id", "==", site)])
    tel = tel[(tel.ts.dt.month == 6)].head(96 * 7)
    fig, ax = plt.subplots(figsize=(10, 4))
    t = tel.ts.dt.tz_convert("Europe/Bucharest")
    for col, c in [("pv_ac_w", "#f08c00"), ("load_w", "#495057"), ("batt_w", "#2b8a3e"), ("grid_w", "#3b5bdb")]:
        ax.plot(t, tel[col] / 1000, lw=1, color=c, label=col)
    ax2 = ax.twinx()
    ax2.plot(t, tel.soc, color="#ae3ec9", lw=1, ls=":", label="soc")
    ax2.set_ylim(0, 1)
    ax.set_ylabel("kW")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"{site}: one June week")
    fig.tight_layout()
    fig.savefig(pdir / "site_week.png", dpi=110)
    plt.close(fig)
