"""Trust ledger: how far each witness can be believed, per situation (team/judge-design.md §6).

Input: graded outcome rows (§8 `outcomes`) with bucket attributes, written by the grader (J1, #146):

    run_id, model, model_version, asset_id, day, score_name, score, weight, source, observed_at, gated
    + bucket attributes: either a ready `bucket` key (sentinel/judge/buckets.py) or the columns
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
- band scale: weighted split-conformal 0.80 quantile of the normalised errors, with the same decay, bucket-day
  units, causality and backoff, bounded to 0.5–3 (1 when there is no evidence);
- health alert per bucket series: raised when trust < prior − 0.2 for 3 consecutive days with n_eff ≥ 5, cleared
  when trust > prior − 0.1.

Implementation: the decay is a common factor, Σ 0.5^((D−dᵢ)/14)·xᵢ = 0.5^(D/14)·Σ 2^(dᵢ/14)·xᵢ, so every
bucket-day becomes a fixed-weight event at its observation day, and the ledger over all days is a cumulative sum
over observation days on a dense (series × day) grid. A bucket-day whose composition changes when a late outcome
arrives (a crew report after the telemetry) enters as the delta of its contribution. No per-day Python loop.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import betaincinv

from sentinel.judge.buckets import CONDITIONS, ROOT, ancestors, truncate

N0 = 7.0                # prior strength in bucket-days (§6.3)
HALF_LIFE = 14.0        # days (§6.3)
MIN_EVIDENCE = 5.0      # effective bucket-days before a bucket is read instead of its parent (§6.1)
ALERT_DROP = 0.2        # raise below prior − 0.2 ...
ALERT_DAYS = 3          # ... for 3 consecutive days ...
ALERT_CLEAR = 0.1       # ... clear above prior − 0.1 (§6.5)
BAND_TARGET = 0.80
BAND_MIN, BAND_MAX = 0.5, 3.0
BAND_SCORE = "norm_err"
DEFAULT_PRIOR = 0.5     # only for a model with no priors at all (should not happen once #147's priors ship)
PRIOR_MIN_N = 20        # validation asset-days for a bucket's own prior to count as "covered" (§6.2)
INTERVAL = (0.10, 0.90)

_BAND_EDGES = np.geomspace(BAND_MIN, BAND_MAX, 65)  # 64 log bins inside the bounds (2.8 % resolution)
TRUST_COLUMNS = ["run_id", "day", "model", "model_version", "bucket", "score_name", "prior", "n_eff", "trust",
                 "lo80", "hi80", "band_scale", "health_alert"]
_SERIES = ["model", "model_version", "score_name", "key"]
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

_KINDS = ("site", "turbine", "zone", "feeder")


def _daynum_np(s: pl.Series) -> np.ndarray:
    if s.dtype == pl.Date:
        x = s
    elif isinstance(s.dtype, pl.Datetime):
        x = s.dt.date()
    elif s.dtype.is_integer():
        return s.cast(pl.Int32).to_numpy()
    else:
        x = s.cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d")
    return x.cast(pl.Int32).to_numpy()


def _cat(s: pl.Series) -> tuple[np.ndarray, list[str]]:
    """Dense integer codes 0..k−1 + labels of a string column (hashing only the few distinct values)."""
    c = s.cast(pl.Utf8).fill_null("").cast(pl.Categorical)
    phys = c.to_physical().to_numpy().astype(np.int64)
    u = pl.DataFrame({"c": c}).unique().with_columns(pl.col("c").to_physical().alias("p"))
    p, lab = u["p"].to_numpy().astype(np.int64), u["c"].cast(pl.Utf8).to_list()
    order = np.argsort(p)
    lut = np.full(int(p.max()) + 1 if len(p) else 1, -1, np.int64)
    lut[p[order]] = np.arange(len(p))
    return lut[phys], [lab[i] for i in order]


def _flag_np(df: pl.DataFrame, col: str, yes: str, default: bool) -> np.ndarray:
    if col not in df.columns:
        return np.full(df.height, int(default), np.int64)
    s = df[col]
    if s.dtype == pl.Utf8:
        return s.str.to_lowercase().is_in([yes, "1", "true", "yes", "y"]).fill_null(default).to_numpy() \
            .astype(np.int64)
    return s.cast(pl.Boolean).fill_null(default).to_numpy().astype(np.int64)


def _dense(code: np.ndarray, labels_of) -> tuple[np.ndarray, list]:
    """Compact a sparse non-negative int code to 0..k−1 (via a lookup table, no sort of the rows)."""
    if len(code) == 0:
        return code, []
    present = np.flatnonzero(np.bincount(code))
    lut = np.full(int(code.max()) + 1, -1, np.int64)
    lut[present] = np.arange(len(present))
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
    bucket_labels: list    # code -> key

    def take(self, m: np.ndarray) -> "Coded":
        return Coded(self.base[m], self.bucket[m], self.day[m], self.obs[m], self.score[m], self.weight[m],
                     self.base_labels, self.bucket_labels)


def prepare(outcomes) -> Coded:
    """Outcome rows → integer-coded arrays; gated / non-finite / zero-weight rows dropped, obs ≥ day."""
    df = outcomes if isinstance(outcomes, pl.DataFrame) else pl.from_pandas(outcomes) \
        if hasattr(outcomes, "to_numpy") else pl.DataFrame(outcomes)
    if df.height == 0:
        e = np.zeros(0, np.int64)
        return Coded(e, e, e, e, e.astype(float), e.astype(float), [], [])
    if "weight" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("weight"))
    keep = pl.col("score").cast(pl.Float64).is_finite() & (pl.col("weight").cast(pl.Float64).fill_null(1.0) > 0)
    if "gated" in df.columns:
        keep = keep & (pl.col("gated").cast(pl.Int8).fill_null(0) == 0)
    df = df.filter(keep)
    day = _daynum_np(df["day"])
    obs = _daynum_np(df["observed_at"]).copy() if "observed_at" in df.columns else day.copy()
    np.maximum(obs, day, out=obs)
    score = df["score"].cast(pl.Float64).to_numpy()
    w = df["weight"].cast(pl.Float64).fill_null(1.0).to_numpy()
    m, ml = _cat(df["model"])
    v, vl = _cat(df["model_version"])
    s, sl = _cat(df["score_name"])
    nv, ns = len(vl), len(sl)
    base, base_labels = _dense((m * nv + v) * ns + s,
                               lambda c: (ml[c // (nv * ns)], vl[(c // ns) % nv], sl[c % ns]))
    if "bucket" in df.columns:
        bucket, bucket_labels = _cat(df["bucket"])
    else:
        if "kind" in df.columns:
            k, kl = _cat(df["kind"])
        else:
            k, kl = np.zeros(df.height, np.int64), ["site"]
        seen = _flag_np(df, "seen", "seen", True)
        batt = _flag_np(df, "battery", "batt", False)
        if "season" in df.columns:
            warm = (df["season"].cast(pl.Utf8).str.to_lowercase() == "warm").fill_null(False).to_numpy()
        else:
            month = day.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
            warm = (month >= 4) & (month <= 9)
        if "quality" in df.columns and df.schema["quality"] == pl.Utf8:
            gappy = (df["quality"].str.to_lowercase() == "gappy").fill_null(False).to_numpy()
        else:
            gappy = _flag_np(df, "quality" if "quality" in df.columns else "gappy", "gappy", False)
        ood = _flag_np(df, "ood_high", "ood_hi", False)
        code = (((((k * 2 + seen) * 2 + batt) * 2 + warm.astype(np.int64)) * 2 + gappy.astype(np.int64)) * 2
                + ood)

        def lab(c: int) -> str:
            bits = [(c >> i) & 1 for i in range(5)]  # ood, gappy, warm, batt, seen
            return "|".join((kl[c >> 5], "seen" if bits[4] else "unseen", "batt" if bits[3] else "nobatt",
                             "warm" if bits[2] else "cold", "gappy" if bits[1] else "clean",
                             "ood_hi" if bits[0] else "ood_lo"))
        bucket, bucket_labels = _dense(code, lab)
    return Coded(base, bucket, day.astype(np.int64), obs.astype(np.int64), score, w, base_labels, bucket_labels)


# ---------------------------------------------------------------- dense causal accumulation

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
        band_scores = {i for i, (_, _, s) in enumerate(c.base_labels) if s == BAND_SCORE}
        is_band = np.isin(c.base, list(band_scores)) if band_scores else np.zeros(n, bool)
        self.levels = self._trust_levels(c.take(~is_band))
        self.bands = self._band_levels(c.take(is_band)) if is_band.any() else []

    def _series(self, base: np.ndarray, par: np.ndarray, n_par: int, labels) -> tuple[np.ndarray, list]:
        return _dense(base * n_par + par, lambda x: labels(x // n_par, x % n_par))

    def _level_sums(self, c: Coded, value: bool):
        """Per level: (sid, day, obs) sums of the bucket-day units → (events, series keys)."""
        cols = {"base": c.base, "b": c.bucket, "day": c.day, "obs": c.obs, "w": c.weight}
        if value:
            cols["ws"] = c.weight * c.score
        aggs = [pl.col("w").sum().alias("sw"), pl.len().alias("n")]
        if value:
            aggs.append(pl.col("ws").sum().alias("sws"))
        agg0 = pl.DataFrame(cols).group_by(["base", "b", "day", "obs"]).agg(aggs)
        b0, base0 = agg0["b"].to_numpy(), agg0["base"].to_numpy()
        for par, uniq in self._par:
            sid, keys = self._series(base0, par[b0], len(uniq),
                                     lambda bi, pi: (*c.base_labels[bi], uniq[pi]))
            sums = ["sw", "n"] + (["sws"] if value else [])
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
        flat = (sid[ok] * D + j[ok]) * nb + b[ok]
        cube = np.bincount(flat, weights=mass, minlength=S * D * nb).reshape(S, D, nb)
        np.cumsum(cube, axis=1, out=cube)
        out = []
        fpos = {k: i for i, k in enumerate(self.full)}
        for (par, uniq), (ev, keys) in zip(self._par, self._level_sums(c, value=False)):
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
    def _resolve(self, model: str, version: str, score: str, bucket: str, di: np.ndarray):
        """Backoff for one series over day indexes di → (trust, n_eff, prior, alert, read level, lo, hi)."""
        n = len(di)
        trust = np.full(n, np.nan)
        n_eff, alert = np.zeros(n), np.zeros(n, np.int8)
        prior = np.full(n, np.nan)
        read = np.full(n, -1)
        root_w = np.zeros(n)
        chain = ancestors(bucket)
        nl = len(self.levels)
        for k in chain:
            lv = nl - 1 - (0 if k == ROOT else len(k.split("|")))
            L = self.levels[lv] if 0 <= lv < nl else None
            sid = L.index.get((model, version, score, k)) if L is not None else None
            if sid is None:
                continue
            w = L.n_eff[sid, di]
            if k == ROOT:
                root_w = w
            take = (read < 0) & (w >= MIN_EVIDENCE)
            trust[take], n_eff[take] = L.trust[sid, di][take], w[take]
            prior[take], alert[take] = L.prior[sid, 0], L.alert[sid, di][take]
            read[take] = lv
        miss = read < 0
        trust[miss] = prior[miss] = self.priors.get(model, version, score, bucket)
        n_eff[miss] = root_w[miss]
        return trust, n_eff, prior, alert, read, *beta_interval(trust, np.where(miss, 0.0, n_eff))

    def _band(self, model: str, version: str, bucket: str, di: np.ndarray) -> np.ndarray:
        out = np.full(len(di), np.nan)
        nl = len(self.bands)
        for k in ancestors(bucket):
            lv = nl - 1 - (0 if k == ROOT else len(k.split("|")))
            B = self.bands[lv] if 0 <= lv < nl else None
            r = B.index.get((model, version, k)) if B is not None else None
            if r is None:
                continue
            take = np.isnan(out) & (B.n_eff[r, di] >= MIN_EVIDENCE) & np.isfinite(B.scale[r, di])
            out[take] = B.scale[r, di][take]
        return np.where(np.isnan(out), 1.0, out)

    def lookup(self, model: str, version: str, score: str, bucket: str, day) -> dict:
        """Trust of any (witness, version, score, bucket) on one day, including buckets with no record."""
        di = np.array([min(max(_daynum(day) - self.d0, 0), self.n_days - 1)])
        t, n, p, a, lv, lo, hi = self._resolve(model, version, score, bucket, di)
        return dict(model=model, model_version=version, score_name=score, bucket=bucket, prior=float(p[0]),
                    trust=float(t[0]), n_eff=float(n[0]), lo80=float(lo[0]), hi80=float(hi[0]),
                    band_scale=float(self._band(model, version, bucket, di)[0]), health_alert=int(a[0]),
                    read_level=int(lv[0]))

    def frame(self, extra: list[tuple[str, str, str, str]] = ()) -> pl.DataFrame:
        """Every graded (model, version, score, full bucket) series on every grid day, plus `extra` series
        (e.g. a new version or a bucket with no record), with the bucket each value was read from."""
        L = self.levels[0]
        keys = list(L.keys) + list(extra)
        di = np.arange(self.n_days)
        num = ("prior", "n_eff", "trust", "lo80", "hi80", "band_scale", "health_alert", "read_level")
        cols = {c: [] for c in num}
        band_cache = {}
        for m, v, s, b in keys:
            t, n, p, a, lv, lo, hi = self._resolve(m, v, s, b, di)
            if (m, v, b) not in band_cache:
                band_cache[(m, v, b)] = self._band(m, v, b, di)
            for c, x in zip(num, (p, n, t, lo, hi, band_cache[(m, v, b)], a, lv)):
                cols[c].append(x)
        if not keys:
            return pl.DataFrame(schema={c: pl.Float64 for c in ("model", "model_version", "score_name",
                                                                "bucket", "day", *num)})
        lvl_name = {i: len(CONDITIONS) - i for i in range(len(self.levels))}
        kf = pl.DataFrame({c: [k[i] for k in keys] for i, c in enumerate(_STR)}).with_row_index("k")
        out = pl.DataFrame({"k": np.repeat(np.arange(len(keys), dtype=np.uint32), self.n_days),
                            "day": np.tile(di + self.d0, len(keys)),
                            **{c: np.concatenate(x) for c, x in cols.items()}})
        out = out.join(kf, on="k", how="left").select(*_STR, "day", *num)
        return out.with_columns(pl.col("day").cast(pl.Int32).cast(pl.Date),
                                pl.col("read_level").replace_strict(lvl_name, default=-1)
                                .alias("read_conditions"), pl.col("health_alert").cast(pl.Int8)) \
            .drop("read_level")

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

def beta_interval(trust: np.ndarray, n_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """80 % interval of Beta(trust·(n0+n_eff), (1−trust)·(n0+n_eff))."""
    t = np.clip(np.asarray(trust, float), 1e-6, 1 - 1e-6)
    n = N0 + np.asarray(n_eff, float)
    a, b = t * n, (1 - t) * n
    return betaincinv(a, b, INTERVAL[0]), betaincinv(a, b, INTERVAL[1])


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
    import datetime as dt
    if isinstance(d, (int, np.integer)):
        return int(d)
    if isinstance(d, dt.datetime):
        d = d.date()
    elif not isinstance(d, dt.date):
        d = dt.date.fromisoformat(str(d)[:10])
    return (d - dt.date(1970, 1, 1)).days


def replay(outcomes, priors: Priors | None = None, run_id: str = "", start=None, end=None) -> pl.DataFrame:
    """Replay the ledger over the outcomes' days → §8 `trust` rows."""
    return Ledger(outcomes, priors, start, end, run_id).trust_rows()
