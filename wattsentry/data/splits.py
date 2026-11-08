"""Train / val / calibration / test splits. One definition for every model, baseline and eval.

Time, by the TARGET day (UTC date of the window's first step):
  train  2022-2023
  val    2024, ISO weeks with week % 4 != 3
  cal    2024, ISO weeks with week % 4 == 3   (conformal calibration + threshold tuning; never trained on)
  test   2025
Sites: `split` column of the sites table: `train` (70 %) or `unseen` (30 %, never in train/val/cal).
train / val / cal use train sites only; test uses all sites, reported seen vs unseen.
Windows need `history_days` of past data, so the first days of the series are never targets.

Small series (tests, smoke runs) that miss a year fall back to proportional time blocks, so every split is non-empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "cal", "test")
TRAIN_YEARS = (2022, 2023)
VAL_YEAR = 2024
TEST_YEAR = 2025
HISTORY_DAYS = 14


@dataclass(frozen=True)
class Splits:
    days: dict[str, np.ndarray]      # split -> target day indices (into the day axis)
    train_sites: np.ndarray          # site indices
    unseen_sites: np.ndarray

    @property
    def seen_mask(self) -> np.ndarray:
        n = len(self.train_sites) + len(self.unseen_sites)
        m = np.zeros(n, bool)
        m[self.train_sites] = True
        return m

    def sites_for(self, split: str) -> np.ndarray:
        if split == "test":
            return np.sort(np.concatenate([self.train_sites, self.unseen_sites]))
        return self.train_sites

    def site_days(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """(site, day) index pairs of every window in a split."""
        s, d = np.meshgrid(self.sites_for(split), self.days[split], indexing="ij")
        return s.ravel(), d.ravel()


def time_split(day_dates: pd.DatetimeIndex, history_days: int = HISTORY_DAYS) -> dict[str, np.ndarray]:
    """Target-day indices per split for a daily axis (one entry per window start)."""
    dates = pd.DatetimeIndex(day_dates)
    years = dates.year.to_numpy()
    week = dates.isocalendar().week.to_numpy().astype(int)
    idx = np.arange(len(dates))
    ok = idx >= history_days
    days = {
        "train": idx[ok & np.isin(years, TRAIN_YEARS)],
        "val": idx[ok & (years == VAL_YEAR) & (week % 4 != 3)],
        "cal": idx[ok & (years == VAL_YEAR) & (week % 4 == 3)],
        "test": idx[ok & (years == TEST_YEAR)],
    }
    if any(len(v) == 0 for v in days.values()):
        avail = idx[ok]
        n = len(avail)
        a, b, c = int(n * 0.55), int(n * 0.7), int(n * 0.8)
        days = {"train": avail[:a], "val": avail[a:b], "cal": avail[b:c], "test": avail[c:]}
    return days


def site_split(site_split_col) -> tuple[np.ndarray, np.ndarray]:
    """(train site indices, unseen site indices) from the sites table's `split` column."""
    s = np.asarray(site_split_col).astype(str)
    bad = set(s) - {"train", "unseen"}
    if bad:
        raise ValueError(f"unknown site split values {bad}")
    return np.flatnonzero(s == "train"), np.flatnonzero(s == "unseen")


def make_splits(day_dates: pd.DatetimeIndex, site_split_col, history_days: int = HISTORY_DAYS) -> Splits:
    tr, un = site_split(site_split_col)
    return Splits(time_split(day_dates, history_days), tr, un)
