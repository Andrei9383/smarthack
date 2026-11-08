"""data-v2: PPC prosumers with contracts, meters and hidden battery strategies (card #172, spec I steps 2-3).

A NEW dataset next to the fleet (wattsentry.sim.fleet, data-v4), which stays byte-identical. Layout: SCHEMA.md.

    from wattsentry.sim.prosumers_v2 import ProsumerConfig, generate, load_training
    out = generate(ProsumerConfig())        # cached at $WATTSENTRY_CACHE/prosumers-v2/v1-<digest>
    tr = load_training(out, years=[2024])   # only what PPC sees: public home attributes + meter data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PV2_VERSION, QUICK, ProsumerConfig
from .generate import FORBIDDEN_COLUMNS, HIDDEN_FILES, TRAINING_FILES, generate, resolve_out

__all__ = ["FORBIDDEN_COLUMNS", "HIDDEN_FILES", "PV2_VERSION", "QUICK", "TRAINING_FILES", "ProsumerConfig",
           "generate", "load_training", "load_truth", "resolve_out"]


def _years(out: Path, sub: str, years: list[int] | None, homes: list[str] | None, columns=None) -> pd.DataFrame:
    """Year files (settlement/year=Y.parquet) or year dirs of chunk parts (meter_observed/year=Y/part-CCC.parquet),
    in home then time order."""
    files = sorted((Path(out) / sub).glob("year=*"), key=lambda p: int(p.name.split("=")[1].split(".")[0]))
    if years:
        files = [p for p in files if int(p.name.split("=")[1].split(".")[0]) in years]
    files = [q for p in files for q in (sorted(p.glob("part-*.parquet")) if p.is_dir() else [p])]
    filt = [("home_id", "in", homes)] if homes else None
    return pd.concat([pd.read_parquet(p, filters=filt, columns=columns) for p in files], ignore_index=True)


def load_training(out: Path, years: list[int] | None = None, homes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Everything a model may train on: homes_public, meter_observed (smart, D+1), meter_monthly, settlement, prices."""
    out = Path(out)
    mm = pd.read_parquet(out / "meter_monthly.parquet")
    if years:
        mm = mm[mm.month.str[:4].astype(int).isin(years)]
    if homes:
        mm = mm[mm.home_id.isin(homes)]
    return {"homes": pd.read_parquet(out / "homes_public.parquet"),
            "meter_observed": _years(out, "meter_observed", years, homes),
            "meter_monthly": mm.reset_index(drop=True),
            "settlement": _years(out, "settlement", years, None),
            "prices": pd.read_parquet(out / "prices.parquet")}


def load_truth(out: Path, years: list[int] | None = None, homes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Hidden ground truth (labels / evaluation only): homes (incl. strategy), per-home 15-min, portfolio sums."""
    out = Path(out)
    return {"homes": pd.read_parquet(out / "homes.parquet"),
            "home_15min": _years(out, "truth/home_15min", years, homes),
            "portfolio": _years(out, "truth/portfolio", years, None)}
