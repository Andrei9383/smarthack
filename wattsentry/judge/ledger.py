"""Trust ledger: how far each witness can be believed, per situation (team/judge-design.md §6).

Input: graded outcome rows (§8 `outcomes`) with bucket attributes, written by the grader (J1, #146):

    run_id, model, model_version, asset_id, day, score_name, score, weight, source, observed_at, gated
    + bucket attributes: either a ready `bucket` key (wattsentry/judge/buckets.py) or the columns
      kind (site|turbine|zone|feeder), seen (bool), battery (bool), quality ('clean'|'gappy' or bool gappy),
      ood_high (bool); season is derived from `day` unless a `season` column is given.

`gated = 1` means the asset-day failed the §5.4 quality gate: it is ungraded and ignored. Rows with
`score_name = "norm_err"` are not trust scores: they carry normalised forecast errors for the band scale (§6.4),
e = (q50 − y)/(q50 − q10) below the median, (y − q50)/(q90 − q50) above it, any number of rows per asset-day.

Output: one row per (day, model, model_version, bucket, score_name) in the §8 `trust` shape (`trust_rows`).

The rules (§6), all evaluated causally on each day D:
- only outcomes with observed_at ≤ D are used (§5.5);
- all asset-day scores of one (witness, version, score, bucket, day) are averaged into one bucket-day (§5.7); its
  weight is min(1, mean outcome weight), so a day counts as at most one unit and action-caused visits (weight 0.5)
  count half; inverse-probability weights > 1 only weight the mean within the day;
- trust = (n0·prior + Σ wᵢ·sᵢ)/(n0 + Σ wᵢ), n0 = 7 bucket-days, wᵢ = 0.5^(age/14) × bucket-day weight, where age is
  D minus the outcome's day; n_eff = Σ wᵢ (≤ ≈ 20.7 at this half-life, so trust can move far from the prior);
- below 5 effective bucket-days a bucket reads its parent (conditions dropped last to first); below 5 at the root
  (all of the model's evidence) it reads the prior; the prior comes from the model version's own validation
  priors (`Priors`), so a new version starts from its own prior and **no record → trust = prior**;
- 80 % interval from Beta(trust·(n0+n_eff), (1−trust)·(n0+n_eff));
- band scale: weighted split-conformal 0.80 quantile of the normalised errors (row mass = weight × decay), with
  the same causality and backoff (≥ 5 effective bucket-days, one unit per bucket-day with norm_err rows),
  bounded to 0.5–3 (1 when there is no evidence);
- `brier` rows (M7 detail, lower is better) are not trust scores and are ignored;
- health alert per bucket series: raised when trust < prior − 0.2 for 3 consecutive days with n_eff ≥ 5, cleared
  when trust > prior − 0.1.

Implementation: the decay is a common factor, Σ 0.5^((D−dᵢ)/14)·xᵢ = 0.5^(D/14)·Σ 2^(dᵢ/14)·xᵢ, so every
bucket-day becomes a fixed-weight event at its observation day, and the ledger over all days is a cumulative sum
over observation days on a dense (series × day) grid. A bucket-day whose composition changes when a late outcome
arrives (a crew report after the telemetry) enters as the delta of its contribution. No per-day Python loop.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from scipy.special import betaincinv

from wattsentry.judge.buckets import CONDITIONS, ROOT, ancestors, truncate

N0 = 7.0                # prior strength in bucket-days (§6.3)
HALF_LIFE = 14.0        # days (§6.3)
MIN_EVIDENCE = 5.0      # effective bucket-days before a bucket is read instead of its parent (§6.1)
ALERT_DROP = 0.2        # raise below prior − 0.2 ...
ALERT_DAYS = 3          # ... for 3 consecutive days ...
ALERT_CLEAR = 0.1       # ... clear above prior − 0.1 (§6.5)
BAND_TARGET = 0.80
BAND_MIN, BAND_MAX = 0.5, 3.0
BAND_SCORE = "norm_err"
DETAIL_SCORES = frozenset({"brier"})  # graded details (lower is better), not trust scores
DEFAULT_PRIOR = 0.5     # only for a model with no priors at all (should not happen once #147's priors ship)
PRIOR_MIN_N = 20        # validation asset-days for a bucket's own prior to count as "covered" (§6.2)
INTERVAL = (0.10, 0.90)

_BAND_EDGES = np.geomspace(BAND_MIN, BAND_MAX, 65)  # 64 log bins inside the bounds (2.8 % resolution)
TRUST_COLUMNS = ["run_id", "day", "model", "model_version", "bucket", "score_name", "prior", "n_eff", "trust",
                 "lo80", "hi80", "band_scale", "health_alert"]
_STR = ("model", "model_version", "score_name", "bucket")


# ---------------------------------------------------------------- priors (§6.2)

@dataclass
class Priors:
    """Validated priors: {(model, model_version or None): {score_name: {bucket key: (prior, n)}}}.

    `get` returns the most specific covered bucket figure (n ≥ PRIOR_MIN_N), walking parents up to the model's
    overall figure "*"; version-specific priors win over the model's generic ones.
    """

    table: dict = field(default_factory=dict)
    default: float = DEFAULT_PRIOR
    min_n: int = PRIOR_MIN_N

    def add(self, model: str, version: str | None, scores: dict) -> None:
        dst = self.table.setdefault((model, version), {})
        for score, by_key in scores.items():
            for key, v in by_key.items():
                prior, n = (v.get("prior"), v.get("n", self.min_n)) if isinstance(v, dict) else (v, self.min_n)
                if prior is not None and math.isfinite(float(prior)):
                    dst.setdefault(score, {})[key] = (float(prior), int(n))

    def get(self, model: str, version: str | None, score: str, key: str = ROOT) -> float:
        for tab in (self.table.get((model, version)), self.table.get((model, None))):
            by_key = (tab or {}).get(score)
            if not by_key:
                continue
            chain = ancestors(key)
            for k in chain:
                if k in by_key and by_key[k][1] >= self.min_n:
                    return by_key[k][0]
            for k in chain:  # thinly covered everywhere: still the most specific validated figure
                if k in by_key:
                    return by_key[k][0]
        return self.default

    @classmethod
    def from_dir(cls, models_dir: str | Path = "models", **kw) -> "Priors":
        """Read `priors` from every *.calibration.json and *.priors.json in models_dir, plus judge_priors.json."""
        p, d = cls(**kw), Path(models_dir)
        for f in sorted(d.glob("*.calibration.json")) + sorted(d.glob("*.priors.json")):
            try:
                cal = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            pri = cal.get("priors") or (cal if "scores" in cal else None)
            model = cal.get("model") or f.name.split(".")[0]
            if pri and pri.get("scores"):
                p.add(model, cal.get("model_version"), pri["scores"])
                if (model, None) not in p.table:  # ensemble members share the ensemble's priors
                    p.add(model, None, pri["scores"])
        jp = d / "judge_priors.json"
        if jp.exists():
            for model, m in json.loads(jp.read_text()).get("models", {}).items():
                p.add(model, None, m.get("scores", {}))
                if m.get("model_version"):
                    p.add(model, m["model_version"], m.get("scores", {}))
        return p


# ---------------------------------------------------------------- input normalisation (integer codes)

LOCAL_TZ = "Europe/Bucharest"  # replay days are local days
_EPOCH = dt.date(1970, 1, 1)
_MISSING = -(10 ** 9)


def _ts_day(t: dt.datetime) -> int:
    """A timestamp's Europe/Bucharest local date as days since 1970-01-01 (naive = UTC)."""
    return ((t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).astimezone(ZoneInfo(LOCAL_TZ)).date() - _EPOCH).days


def _local_day(v: str) -> int:
    """'YYYY-MM-DD' → that day; an ISO timestamp → its Europe/Bucharest local date (naive = UTC)."""
    v = v.strip()
    if len(v) <= 10:
        return (dt.date.fromisoformat(v) - _EPOCH).days
    return _ts_day(dt.datetime.fromisoformat(v.replace("Z", "+00:00")))


def _daynum_np(s: pl.Series) -> np.ndarray:
    """Day numbers (days since 1970-01-01) of a date / timestamp column. Timestamps are taken in local time
    (Europe/Bucharest): J1 writes observed_at as the UTC instant of the next local midnight, e.g.
    2024-06-02T21:00:00Z = 2024-06-03 00:00 EEST, which must count as 06-03, not as the graded day 06-02."""
    if s.dtype == pl.Date or s.dtype.is_integer():
        return s.cast(pl.Int32).to_numpy()
    if isinstance(s.dtype, pl.Datetime):
        if s.dtype.time_zone is None:
            s = s.dt.replace_time_zone("UTC")
        return s.dt.convert_time_zone(LOCAL_TZ).dt.date().cast(pl.Int32).to_numpy()
    s = s.cast(pl.Utf8)
    uniq = s.unique().drop_nulls().to_list()  # few distinct values (one per day or visit): parse each once
    lut = {u: _local_day(u) for u in uniq}
    return s.replace_strict(lut, default=None, return_dtype=pl.Int32).fill_null(_MISSING).to_numpy()


def _cats(df: pl.DataFrame, cols: list[str]) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Dense int32 codes 0..k−1 + labels for string columns; the per-row hashing runs in one parallel select
    (a column that is already Categorical/Enum is not re-hashed)."""
    def as_cat(c):
        dt_ = df.schema[c]
        e = pl.col(c)
        return (e if dt_ in (pl.Categorical,) or isinstance(dt_, pl.Enum)
                else e.cast(pl.Utf8).fill_null("").cast(pl.Categorical)).alias(c)
    cat = df.select([as_cat(c) for c in cols])
    out = {}
    for c in cols:
        phys = cat[c].to_physical().to_numpy()
        u = cat.select(pl.col(c).unique()).with_columns(pl.col(c).to_physical().alias("p"))
        p, lab = u["p"].to_numpy().astype(np.int64), u[c].cast(pl.Utf8).to_list()
        order = np.argsort(p)
        lut = np.full(int(p.max()) + 1 if len(p) else 1, -1, np.int32)
        lut[p[order]] = np.arange(len(p), dtype=np.int32)
        out[c] = (lut[phys], [lab[i] for i in order])
    return out


def _flag_np(df: pl.DataFrame, col: str, yes: str, default: bool) -> np.ndarray:
    if col not in df.columns:
        return np.full(df.height, int(default), np.int32)
    s = df[col]
    if s.dtype == pl.Utf8:
        s = s.str.to_lowercase().is_in([yes, "1", "true", "yes", "y"])
    return s.cast(pl.Boolean).fill_null(default).cast(pl.Int32).to_numpy()


def _dense(code: np.ndarray, labels_of) -> tuple[np.ndarray, list]:
    """Compact a sparse non-negative int code to 0..k−1 (via a lookup table, no sort of the rows)."""
    if len(code) == 0:
        return code, []
    present = np.flatnonzero(np.bincount(code))
    lut = np.full(int(code.max()) + 1, -1, np.int32)
    lut[present] = np.arange(len(present), dtype=np.int32)
    return lut[code], [labels_of(int(c)) for c in present]


@dataclass
class Coded:
    """Outcome rows as integer arrays. base = (model, model_version, score_name); bucket = full bucket key."""
    base: np.ndarray
    bucket: np.ndarray
    day: np.ndarray
    obs: np.ndarray
    score: np.ndarray
    weight: np.ndarray
    base_labels: list      # code -> (model, version, score)
    bucket_labels: list    # code -> key (may include buckets seen only on dropped rows)
    keep: np.ndarray | None = None  # rows to use (not gated, finite score, weight > 0); applied by take()

    def take(self, m: np.ndarray) -> "Coded":
        if self.keep is not None:
            m = m & self.keep
        return Coded(self.base[m], self.bucket[m], self.day[m], self.obs[m], self.score[m], self.weight[m],
                     self.base_labels, self.bucket_labels)


def prepare(outcomes) -> Coded:
    """Outcome rows → integer-coded arrays; gated / non-finite / zero-weight rows dropped, obs ≥ day."""
    df = outcomes if isinstance(outcomes, pl.DataFrame) else pl.from_pandas(outcomes) \
        if hasattr(outcomes, "to_numpy") else pl.DataFrame(outcomes)
    if df.height == 0:
        e = np.zeros(0, np.int64)
        return Coded(e, e, e, e, e.astype(float), e.astype(float), [], [])
    day = _daynum_np(df["day"])
    # an outcome is known on its local observation day; without one, the earliest it can be known is day + 1
    obs = _daynum_np(df["observed_at"]).copy() if "observed_at" in df.columns else day + 1
    obs = np.where(obs <= _MISSING // 2, day + 1, obs)
    np.maximum(obs, day, out=obs)
    score = df["score"].cast(pl.Float64).to_numpy()
    w = df["weight"].cast(pl.Float64).fill_null(1.0).to_numpy() if "weight" in df.columns \
        else np.ones(df.height)
    keep = np.isfinite(score) & (w > 0)
    if "gated" in df.columns:
        keep &= df["gated"].cast(pl.Int8).fill_null(0).to_numpy() == 0
    has_bucket = "bucket" in df.columns
    cats = _cats(df, ["model", "model_version", "score_name"] + (["bucket"] if has_bucket else
                                                                 ["kind"] if "kind" in df.columns else []))
    (m, ml), (v, vl), (s, sl) = cats["model"], cats["model_version"], cats["score_name"]
    nv, ns = len(vl), len(sl)
    base, base_labels = _dense((m * nv + v) * ns + s,
                               lambda c: (ml[c // (nv * ns)], vl[(c // ns) % nv], sl[c % ns]))
    if has_bucket:
        bucket, bucket_labels = cats["bucket"]
    else:
        k, kl = cats.get("kind", (np.zeros(df.height, np.int32), ["site"]))
        seen = _flag_np(df, "seen", "seen", True)
        batt = _flag_np(df, "battery", "batt", False)
        if "season" in df.columns:
            warm = (df["season"].cast(pl.Utf8).str.to_lowercase() == "warm").fill_null(False)
        else:
            month = pl.Series(day).cast(pl.Date).dt.month()
            warm = (month >= 4) & (month <= 9)
        warm = warm.cast(pl.Int32).to_numpy()
        if "quality" in df.columns and df.schema["quality"] == pl.Utf8:
            gappy = (df["quality"].str.to_lowercase() == "gappy").fill_null(False).cast(pl.Int32).to_numpy()
        else:
            gappy = _flag_np(df, "quality" if "quality" in df.columns else "gappy", "gappy", False)
        ood = _flag_np(df, "ood_high", "ood_hi", False)
        code = (((((k * 2 + seen) * 2 + batt) * 2 + warm) * 2 + gappy) * 2 + ood)

        def lab(c: int) -> str:
            bits = [(c >> i) & 1 for i in range(5)]  # ood, gappy, warm, batt, seen
            return "|".join((kl[c >> 5], "seen" if bits[4] else "unseen", "batt" if bits[3] else "nobatt",
                             "warm" if bits[2] else "cold", "gappy" if bits[1] else "clean",
                             "ood_hi" if bits[0] else "ood_lo"))
        bucket, bucket_labels = _dense(code, lab)
    return Coded(base, bucket, day, obs, score, w, base_labels, bucket_labels, keep)


# ---------------------------------------------------------------- dense causal accumulation

_BAND_MAX_LAG = 16  # band evidence: observation lags beyond 15 days are counted at 15 (norm_err is telemetry, d+1)


def _band_units(c: Coded, n_buckets: int) -> pl.DataFrame | None:
    """(base, b, day, obs) → Σweight, row count for the band rows by one dense bincount (no row hashing)."""
    if len(c.day) == 0:
        return None
    bb, bb_lab = _dense(c.base.astype(np.int64) * n_buckets + c.bucket, lambda x: x)
    dmin = int(c.day.min())
    di = (c.day - dmin).astype(np.int64)
    nd, L = int(di.max()) + 1, _BAND_MAX_LAG
    if len(bb_lab) * nd * L > 50_000_000:
        return None  # too sparse for a dense grid: fall back to the polars group-by
    key = (bb.astype(np.int64) * nd + di) * L + np.minimum(c.obs - c.day, L - 1)
    n = np.bincount(key, minlength=len(bb_lab) * nd * L)
    sw = np.bincount(key, weights=c.weight, minlength=len(n))
    nz = np.flatnonzero(n)
    lab = np.asarray(bb_lab, np.int64)[nz // (nd * L)]
    day = (nz // L) % nd + dmin
    return pl.DataFrame({"base": (lab // n_buckets).astype(np.int32), "b": (lab % n_buckets).astype(np.int32),
                         "day": day.astype(np.int32), "obs": (day + nz % L).astype(np.int32),
                         # norm_err rows split their asset-day's weight (J1: 1/k each), so their mean weight says
                         # nothing about the unit: each bucket-day with band rows counts as one unit
                         "sw": np.where(sw[nz] > 0, 1.0, 0.0) * n[nz],
                         "n": n[nz].astype(np.uint32)})


def _unit_events(agg: pl.DataFrame, value: bool) -> pl.DataFrame:
    """(sid, day, obs) sums sw, n [, sws] → delta events of one bucket-day unit: du [, dc = Δ(u·s̄)]."""
    agg = agg.sort(["sid", "day", "obs"])
    grp = ["sid", "day"]
    cols = [pl.col("sw").cum_sum().over(grp).alias("csw"), pl.col("n").cum_sum().over(grp).alias("cn")]
    if value:
        cols.append(pl.col("sws").cum_sum().over(grp).alias("csws"))
    agg = agg.with_columns(cols)
    u = pl.min_horizontal(pl.lit(1.0), pl.col("csw") / pl.col("cn"))
    out = [u.alias("u")]
    if value:
        out.append((u * pl.col("csws") / pl.col("csw")).alias("c"))
    agg = agg.with_columns(out)
    d = [(pl.col("u") - pl.col("u").shift(1).over(grp).fill_null(0.0)).alias("du")]
    if value:
        d.append((pl.col("c") - pl.col("c").shift(1).over(grp).fill_null(0.0)).alias("dc"))
    return agg.with_columns(d)


def _accumulate(sid: np.ndarray, day: np.ndarray, obs: np.ndarray, vals: np.ndarray, n_series: int,
                d0: int, n_days: int) -> np.ndarray:
    """Σ over events with obs ≤ D of 0.5^((D − day)/HL)·val, for every series and grid day D = d0 .. d0+n_days−1.
    vals: (k, n_events) → (k, n_series, n_days)."""
    j = np.clip(obs - d0, 0, None)
    ok = j < n_days
    f = np.exp2((day[ok] - d0) / HALF_LIFE)
    flat = sid[ok].astype(np.int64) * n_days + j[ok]
    decay = np.exp2(-np.arange(n_days) / HALF_LIFE)
    out = np.empty((vals.shape[0], n_series, n_days))
    for k in range(vals.shape[0]):
        acc = np.bincount(flat, weights=vals[k, ok] * f, minlength=n_series * n_days).reshape(n_series, n_days)
        out[k] = np.cumsum(acc, axis=1) * decay
    return out


def _alert_state(trust: np.ndarray, prior: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    """Hysteresis over days (axis 1): raise after ALERT_DAYS consecutive days below prior − DROP with enough
    evidence, clear above prior − CLEAR, otherwise keep the previous state."""
    low = (trust < prior - ALERT_DROP) & (n_eff >= MIN_EVIDENCE)
    c = np.cumsum(low, axis=1)
    run = c - np.maximum.accumulate(np.where(~low, c, 0), axis=1)
    ev = np.where(run >= ALERT_DAYS, 1, np.where(trust > prior - ALERT_CLEAR, 0, -1)).astype(np.int8)
    idx = np.where(ev >= 0, np.arange(ev.shape[1]), 0)
    idx = np.maximum.accumulate(idx, axis=1)
    st = np.take_along_axis(ev, idx, axis=1)
    return (st == 1).astype(np.int8)


@dataclass
class _Level:
    keys: list                # sid -> (model, version, score, bucket key)
    index: dict               # (model, version, score, key) -> sid
    trust: np.ndarray         # (S, D)
    n_eff: np.ndarray
    prior: np.ndarray         # (S, 1)
    alert: np.ndarray


@dataclass
class _BandLevel:
    index: dict               # (model, version, key) -> row
    scale: np.ndarray         # (S, D), NaN where no evidence
    n_eff: np.ndarray


class Ledger:
    """A replayed ledger over a day grid. `trust_rows()` gives the §8 table; `lookup()` answers any bucket."""

    def __init__(self, outcomes, priors: Priors | None = None, start=None, end=None, run_id: str = ""):
        self.priors = priors or Priors()
        self.run_id = run_id
        c = prepare(outcomes)
        n = len(c.day)
        self.d0 = _daynum(start) if start is not None else int(c.day.min()) if n else 0
        d1 = _daynum(end) if end is not None else int(max(c.obs.max(), c.day.max())) if n else self.d0
        self.n_days = max(1, d1 - self.d0 + 1)
        self.full = c.bucket_labels
        # parent key code of every full bucket at every level (level index 0 = full ... 6 = root)
        self._par = []
        for lv in range(len(CONDITIONS), -1, -1):
            keys = [truncate(b, lv) for b in self.full]
            uniq = sorted(set(keys))
            pos = {k: i for i, k in enumerate(uniq)}
            self._par.append((np.array([pos[k] for k in keys], np.int64), uniq))
        is_band = np.isin(c.base, [i for i, (_, _, s) in enumerate(c.base_labels) if s == BAND_SCORE])
        is_detail = np.isin(c.base, [i for i, (_, _, s) in enumerate(c.base_labels) if s in DETAIL_SCORES])
        self.levels = self._trust_levels(c.take(~is_band & ~is_detail))
        self.bands = self._band_levels(c.take(is_band)) if is_band.any() else []

    def _series(self, base: np.ndarray, par: np.ndarray, n_par: int, labels) -> tuple[np.ndarray, list]:
        return _dense(base * n_par + par, lambda x: labels(x // n_par, x % n_par))

    def _level_sums(self, c: Coded, value: bool, agg0: pl.DataFrame | None = None):
        """Per level: (sid, day, obs) sums of the bucket-day units → (events, series keys)."""
        if agg0 is None:
            cols = {"base": c.base, "b": c.bucket, "day": c.day, "obs": c.obs, "w": c.weight}
            if value:
                cols["ws"] = c.weight * c.score
            aggs = [pl.col("w").sum().alias("sw"), pl.len().alias("n")]
            if value:
                aggs.append(pl.col("ws").sum().alias("sws"))
            agg0 = pl.DataFrame(cols).group_by(["base", "b", "day", "obs"]).agg(aggs)
        b0, base0 = agg0["b"].to_numpy(), agg0["base"].to_numpy()
        sums = ["sw", "n"] + (["sws"] if value else [])
        for par, uniq in self._par:
            sid, keys = self._series(base0, par[b0], len(uniq),
                                     lambda bi, pi: (*c.base_labels[bi], uniq[pi]))
            a = (agg0.select(["day", "obs", *sums]).with_columns(pl.Series("sid", sid))
                 .group_by(["sid", "day", "obs"]).agg([pl.col(x).sum() for x in sums]))
            yield _unit_events(a, value), keys

    # ---- trust
    def _trust_levels(self, c: Coded) -> list[_Level]:
        if len(c.day) == 0:
            z = np.zeros((0, self.n_days))
            return [_Level([], {}, z, z, np.zeros((0, 1)), z.astype(np.int8)) for _ in self._par]
        levels = []
        for ev, keys in self._level_sums(c, value=True):
            S = len(keys)
            sw, w = _accumulate(ev["sid"].to_numpy(), ev["day"].to_numpy(), ev["obs"].to_numpy(),
                                np.stack([ev["dc"].to_numpy(), ev["du"].to_numpy()]), S, self.d0, self.n_days)
            w = np.maximum(w, 0.0)
            prior = np.array([self.priors.get(*k) for k in keys], dtype=np.float64)[:, None]
            trust = (N0 * prior + sw) / (N0 + w)
            levels.append(_Level(keys, {k: i for i, k in enumerate(keys)}, trust, w, prior,
                                 _alert_state(trust, prior, w)))
        return levels  # levels[0] = full buckets ... levels[-1] = root

    # ---- band scale (§6.4)
    def _band_levels(self, c: Coded) -> list[_BandLevel]:
        """Recency-weighted conformal quantile per bucket from a dense (series × observation day × bin) histogram
        of the normalised errors; each row has mass weight × 2^(day/HL) (the common decay cancels in a quantile);
        parents sum their children. Evidence (n_eff) counts bucket-day units exactly as for trust."""
        nb = len(_BAND_EDGES) + 1
        # bin = searchsorted(_BAND_EDGES, |e|, "right") on the log grid: 0 below, nb−1 above the bounds
        with np.errstate(divide="ignore"):
            pos = np.log(np.abs(c.score) / BAND_MIN) * ((nb - 2) / math.log(BAND_MAX / BAND_MIN))
        b = np.clip(np.floor(pos) + 1, 0, nb - 1).astype(np.int64)
        par0, uniq0 = self._par[0]
        sid, keys0 = self._series(c.base, par0[c.bucket], len(uniq0),
                                  lambda bi, pi: (*c.base_labels[bi][:2], uniq0[pi]))
        S, D = len(keys0), self.n_days
        j = np.clip(c.obs - self.d0, 0, None)
        ok = j < D
        mass = c.weight[ok] * np.exp2((c.day[ok] - self.d0) / HALF_LIFE)
        flat = (sid[ok].astype(np.int64) * D + j[ok]) * nb + b[ok]
        cube = np.bincount(flat, weights=mass, minlength=S * D * nb).reshape(S, D, nb)
        np.cumsum(cube, axis=1, out=cube)
        out = []
        fpos = {k: i for i, k in enumerate(self.full)}
        for (par, uniq), (ev, keys) in zip(self._par, self._level_sums(c, False, _band_units(c, len(self.full)))):
            # level series (model, version, key) and the full series that roll up into each
            lvl_keys = sorted({(m, v, uniq[par[fpos[k6]]]) for m, v, k6 in keys0})
            pos = {k: i for i, k in enumerate(lvl_keys)}
            up = np.array([pos[(m, v, uniq[par[fpos[k6]]])] for m, v, k6 in keys0], np.int64)
            if len(lvl_keys) == S and np.array_equal(up, np.arange(S)):
                lc = cube
            else:
                lc = np.zeros((len(lvl_keys), D, nb))
                np.add.at(lc, up, cube)
            scale = _hist_quantile(lc, BAND_TARGET)
            # evidence per (model, version, key): units from the band rows (score name is the same for all)
            (w,) = _accumulate(ev["sid"].to_numpy(), ev["day"].to_numpy(), ev["obs"].to_numpy(),
                               ev["du"].to_numpy()[None], len(keys), self.d0, D)
            n_eff = np.zeros((len(lvl_keys), D))
            for i, (m, v, _s, k) in enumerate(keys):
                n_eff[pos[(m, v, k)]] += np.maximum(w[i], 0.0)
            out.append(_BandLevel(pos, scale, n_eff))
        return out

    # ---- reading
    def _resolve(self, keys: list[tuple[str, str, str, str]], di: np.ndarray):
        """Backoff for many (model, version, score, bucket) series over day indexes di, vectorised across series
        → arrays (K, len(di)): trust, n_eff, prior, alert, read level index (−1 = prior), lo80, hi80."""
        K, n = len(keys), len(di)
        trust, n_eff, prior = np.full((K, n), np.nan), np.zeros((K, n)), np.full((K, n), np.nan)
        alert, read, root_w = np.zeros((K, n), np.int8), np.full((K, n), -1), np.zeros((K, n))
        nl = len(self.levels)
        for lv, L in enumerate(self.levels):
            depth = nl - 1 - lv  # conditions kept at this level
            sid = np.array([L.index.get((m, v, s, truncate(b, depth)), -1) for m, v, s, b in keys], np.int64)
            has = sid >= 0
            if not has.any():
                continue
            rows = np.flatnonzero(has)
            w = L.n_eff[sid[rows]][:, di]
            if depth == 0:
                root_w[rows] = w
            take = (read[rows] < 0) & (w >= MIN_EVIDENCE)
            r_idx, d_idx = np.nonzero(take)
            g = rows[r_idx]
            trust[g, d_idx] = L.trust[sid[g], di[d_idx]]
            n_eff[g, d_idx] = w[r_idx, d_idx]
            prior[g, d_idx] = L.prior[sid[g], 0]
            alert[g, d_idx] = L.alert[sid[g], di[d_idx]]
            read[g, d_idx] = lv
        miss = read < 0
        if miss.any():
            p0 = np.array([self.priors.get(*k) for k in keys])[:, None]
            trust = np.where(miss, p0, trust)
            prior = np.where(miss, p0, prior)
            n_eff = np.where(miss, root_w, n_eff)
        lo, hi = beta_interval(trust, np.where(miss, 0.0, n_eff))
        return trust, n_eff, prior, alert, read, lo, hi

    def _band(self, keys: list[tuple[str, str, str]], di: np.ndarray) -> np.ndarray:
        """Band scale for many (model, version, bucket) over day indexes di, with the same backoff; 1 = none."""
        out = np.full((len(keys), len(di)), np.nan)
        nl = len(self.bands)
        for lv, B in enumerate(self.bands):
            depth = nl - 1 - lv
            r = np.array([B.index.get((m, v, truncate(b, depth)), -1) for m, v, b in keys], np.int64)
            rows = np.flatnonzero(r >= 0)
            if len(rows) == 0:
                continue
            sc, w = B.scale[r[rows]][:, di], B.n_eff[r[rows]][:, di]
            take = np.isnan(out[rows]) & (w >= MIN_EVIDENCE) & np.isfinite(sc)
            out[rows] = np.where(take, sc, out[rows])
        return np.where(np.isnan(out), 1.0, out)

    def lookup(self, model: str, version: str, score: str, bucket: str, day) -> dict:
        """Trust of any (witness, version, score, bucket) on one day, including buckets with no record."""
        di = np.array([min(max(_daynum(day) - self.d0, 0), self.n_days - 1)])
        t, n, p, a, lv, lo, hi = (x[0, 0] for x in self._resolve([(model, version, score, bucket)], di))
        return dict(model=model, model_version=version, score_name=score, bucket=bucket, prior=float(p),
                    trust=float(t), n_eff=float(n), lo80=float(lo), hi80=float(hi),
                    band_scale=float(self._band([(model, version, bucket)], di)[0, 0]), health_alert=int(a),
                    read_level=int(lv))

    def frame(self, extra: list[tuple[str, str, str, str]] = ()) -> pl.DataFrame:
        """Every graded (model, version, score, full bucket) series on every grid day, plus `extra` series
        (e.g. a new version or a bucket with no record). `read_conditions` = how many bucket conditions the value
        was read with (6 = its own bucket, 0 = the model's root, −1 = the prior)."""
        keys = list(self.levels[0].keys) + list(extra)
        num = ("prior", "n_eff", "trust", "lo80", "hi80", "band_scale", "health_alert", "read_level")
        if not keys:
            return pl.DataFrame(schema={c: pl.Float64 for c in ("model", "model_version", "score_name",
                                                                "bucket", "day", *num)})
        di = np.arange(self.n_days)
        t, n, p, a, lv, lo, hi = self._resolve(keys, di)
        bkeys = sorted({(m, v, b) for m, v, _s, b in keys})
        bpos = {k: i for i, k in enumerate(bkeys)}
        band = self._band(bkeys, di)[[bpos[(m, v, b)] for m, v, _s, b in keys]]
        nl = len(self.levels)
        read_cond = np.where(lv < 0, -1, nl - 1 - lv)
        kf = pl.DataFrame({c: [k[i] for k in keys] for i, c in enumerate(_STR)}).with_row_index("k")
        out = pl.DataFrame({"k": np.repeat(np.arange(len(keys), dtype=np.uint32), self.n_days),
                            "day": np.tile(di + self.d0, len(keys)).astype(np.int32),
                            **{c: x.ravel() for c, x in zip(num[:-1], (p, n, t, lo, hi, band, a))},
                            "read_conditions": read_cond.ravel()})
        return (out.join(kf, on="k", how="left")
                .select(*_STR, pl.col("day").cast(pl.Date), *num[:-1], "read_conditions")
                .with_columns(pl.col("health_alert").cast(pl.Int8)))

    def trust_rows(self, extra=()) -> pl.DataFrame:
        """The §8 `trust` table."""
        f = self.frame(extra)
        return f.with_columns(pl.lit(self.run_id).alias("run_id"),
                              pl.col("day").cast(pl.Utf8)).select(TRUST_COLUMNS)

    def alerts(self) -> pl.DataFrame:
        """Health-alert episodes per series at every bucket level (e.g. "pvnet degraded on site|unseen since
        2025-06-04"): model, model_version, score_name, bucket, since, until (null = still raised)."""
        rows = []
        for L in self.levels:
            for r in np.flatnonzero(L.alert.any(axis=1)):
                a = np.concatenate([[0], L.alert[r], [0]]).astype(np.int8)
                starts, ends = np.flatnonzero(np.diff(a) == 1), np.flatnonzero(np.diff(a) == -1)
                m, v, s, k = L.keys[int(r)]
                for s0, e0 in zip(starts, ends):
                    rows.append(dict(model=m, model_version=v, score_name=s, bucket=k, since=self.d0 + int(s0),
                                     until=None if e0 >= self.n_days else self.d0 + int(e0)))
        schema = {"model": pl.Utf8, "model_version": pl.Utf8, "score_name": pl.Utf8, "bucket": pl.Utf8,
                  "since": pl.Int32, "until": pl.Int32}
        out = pl.DataFrame(rows, schema=schema)
        return out.with_columns(pl.col("since").cast(pl.Date), pl.col("until").cast(pl.Date))


# ---------------------------------------------------------------- helpers

def _beta_exact(t: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.clip(t, 1e-6, 1 - 1e-6)
    a, b = t * n, (1 - t) * n
    return betaincinv(a, b, INTERVAL[0]), betaincinv(a, b, INTERVAL[1])


_BETA_T = np.linspace(0.0, 1.0, 401)
_BETA_N = np.arange(N0, N0 + 24.0, 0.5)  # n0 + n_eff, n_eff ≤ ≈ 20.7


@functools.lru_cache(maxsize=1)
def _beta_table() -> tuple[np.ndarray, np.ndarray]:
    t, n = np.meshgrid(_BETA_T, _BETA_N, indexing="ij")
    return _beta_exact(t, n)


def beta_interval(trust: np.ndarray, n_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """80 % interval of Beta(trust·(n0+n_eff), (1−trust)·(n0+n_eff)). Large arrays are interpolated
    (bilinear) from a cached quantile table (error < 1e-3); small ones call betaincinv directly."""
    t = np.asarray(trust, float)
    n = N0 + np.asarray(n_eff, float)
    if t.size < 5000 or (n > _BETA_N[-1]).any():
        return _beta_exact(t, n)
    lo_t, hi_t = _beta_table()
    x = np.clip(t, 0.0, 1.0) * (len(_BETA_T) - 1)
    y = (np.clip(n, _BETA_N[0], _BETA_N[-1]) - _BETA_N[0]) / 0.5
    i0 = np.minimum(x.astype(np.int64), len(_BETA_T) - 2)
    j0 = np.minimum(y.astype(np.int64), len(_BETA_N) - 2)
    fx, fy = x - i0, y - j0

    def interp(tab):
        return ((1 - fx) * (1 - fy) * tab[i0, j0] + fx * (1 - fy) * tab[i0 + 1, j0]
                + (1 - fx) * fy * tab[i0, j0 + 1] + fx * fy * tab[i0 + 1, j0 + 1])
    return interp(lo_t), interp(hi_t)


def _hist_quantile(cube: np.ndarray, q: float) -> np.ndarray:
    """Weighted q-quantile from bin masses (..., bins): bin 0 = below BAND_MIN, last = above BAND_MAX, the rest
    log-spaced in between (log-linear interpolation inside a bin). NaN where the mass is zero."""
    tot = cube.sum(-1)
    cdf = np.cumsum(cube, -1) / np.where(tot > 0, tot, 1.0)[..., None]
    k = np.minimum((cdf < q).sum(-1), cube.shape[-1] - 1)  # first bin with cdf ≥ q
    prev = np.where(k > 0, np.take_along_axis(cdf, np.maximum(k - 1, 0)[..., None], -1)[..., 0], 0.0)
    cur = np.take_along_axis(cdf, k[..., None], -1)[..., 0]
    frac = np.clip((q - prev) / np.maximum(cur - prev, 1e-12), 0, 1)
    lo_e = _BAND_EDGES[np.clip(k - 1, 0, len(_BAND_EDGES) - 1)]
    hi_e = _BAND_EDGES[np.clip(k, 0, len(_BAND_EDGES) - 1)]
    val = lo_e * (hi_e / lo_e) ** frac
    val = np.where(k == 0, BAND_MIN, np.where(k >= cube.shape[-1] - 1, BAND_MAX, val))
    return np.where(tot > 0, val, np.nan)


def _daynum(d) -> int:
    """A (local) day as days since 1970-01-01; timestamps are converted to their Europe/Bucharest date."""
    if isinstance(d, (int, np.integer)):
        return int(d)
    if isinstance(d, dt.datetime):
        return _ts_day(d)
    if isinstance(d, dt.date):
        return (d - _EPOCH).days
    return _local_day(str(d))


def replay(outcomes, priors: Priors | None = None, run_id: str = "", start=None, end=None) -> pl.DataFrame:
    """Replay the ledger over the outcomes' days → §8 `trust` rows."""
    return Ledger(outcomes, priors, start, end, run_id).trust_rows()


def main() -> None:
    """python -m wattsentry.judge.ledger outcomes.parquet --out trust.parquet [--models models] [--alerts a.csv]"""
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Replay the Judge trust ledger (§6) over graded outcome rows.")
    ap.add_argument("outcomes", type=Path, help="parquet/csv with §8 outcomes rows + bucket columns")
    ap.add_argument("--out", type=Path, required=True, help="trust rows (.parquet or .csv)")
    ap.add_argument("--models", type=Path, default=Path("models"), help="dir with calibration JSONs + priors")
    ap.add_argument("--alerts", type=Path, help="optional csv of health-alert episodes")
    ap.add_argument("--run-id", default="")
    a = ap.parse_args()
    t = time.perf_counter()
    df = pl.read_parquet(a.outcomes) if a.outcomes.suffix == ".parquet" else pl.read_csv(a.outcomes)
    led = Ledger(df, Priors.from_dir(a.models), run_id=a.run_id)
    rows = led.trust_rows()
    (rows.write_parquet if a.out.suffix == ".parquet" else rows.write_csv)(a.out)
    if a.alerts:
        led.alerts().write_csv(a.alerts)
    print(f"LAB_METRIC t_ledger_s={time.perf_counter() - t:.3f}")
    print(f"{df.height} outcome rows -> {rows.height} trust rows, {led.alerts().height} alert episodes")


if __name__ == "__main__":
    main()
