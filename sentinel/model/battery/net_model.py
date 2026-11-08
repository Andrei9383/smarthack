"""Net import per home and quarter-hour, including the battery (card #174, spec I step 5).

(a) ``NetQuantileModel`` - the fallback: LightGBM quantile (0.1 / 0.5 / 0.9) directly on the home's net import
    (import - export, kWh per 15 min) from ``features.ROW_FEATURES``.
(b) ``simulate_guess`` - stage 2 of the explainable variant: expected PV + expected load + tomorrow's prices go through
    the K8 kernel (``sentinel.kernels.battery_strategies``, Rust when built, numpy reference otherwise) with the
    classifier's strategy and estimated parameters. ``mix`` blends the per-strategy simulations with the classifier's
    probabilities (soft) or takes the most likely one (hard).

Sign convention everywhere: net > 0 = the home imports (consumption sign, as sentinel.eval.imbalance).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from .classifier import PRIOR
from .data import DT_H, World
from .features import ROW_FEATURES

TAUS = (0.1, 0.5, 0.9)
SIM_CODES = (1, 2)   # strategies that need the kernel; 0 = no battery (net = load - pv); 3 (enrolled) runs as 1


def kernel():
    """K8 dispatcher (Rust when built) or the numpy reference if the dispatcher is not registered."""
    try:
        from sentinel.kernels import battery_strategies

        return battery_strategies
    except ImportError:  # pragma: no cover
        from sentinel.kernels.strategies import battery_strategies_py

        return battery_strategies_py


# ============================================================================================ (b) simulation
def simulate(pv: np.ndarray, load: np.ndarray, code: np.ndarray, params: Mapping[str, np.ndarray],
             price: np.ndarray, rank: np.ndarray, med: np.ndarray, day: np.ndarray,
             soc0: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """One K8 run. pv, load (n, T) kW; code (n,) strategy codes; params: cap_kwh, p_kw, reserve_frac, k_top (n,);
    price / rank / med / day (T,) for the steps. Every home complies every day (the guess is the behaviour).
    Returns (net import kW (n, T), soc at the end of the run (n,))."""
    n, T = pv.shape
    cap = np.asarray(params["cap_kwh"], float)
    res = np.asarray(params["reserve_frac"], float)
    if soc0 is None:
        soc0 = cap * (res + 0.5 * (1 - res))
    soc0 = np.clip(soc0, res * cap, cap)
    d = np.asarray(day, np.int64)
    d = d - d.min() if len(d) else d
    comply = np.ones((n, int(d.max()) + 1 if len(d) else 1), np.uint8)
    batt, soc = kernel()(pv, load, np.asarray(code, np.int8), cap, np.asarray(params["p_kw"], float), res,
                         np.full(n, PRIOR["eta"]), np.asarray(params["k_top"], np.int32), soc0,
                         price, rank, med, d.astype(np.int32), comply, DT_H)
    net = (load - pv + batt).astype(np.float32)
    return net, (soc[:, -1] if T else soc0)


def simulate_guess(world: World, steps: np.ndarray, pv: np.ndarray, load: np.ndarray,
                   month_params: list[tuple[np.ndarray, Mapping[str, np.ndarray]]]) -> dict[int, np.ndarray]:
    """Per-strategy net import (code -> (n, len(steps)) kW) over contiguous ``steps``, month by month with the
    parameters estimated at the start of each month (``month_params`` = [(step mask within steps, params)]);
    the SoC carries over between months. code 0 = no battery."""
    out = {0: (load - pv).astype(np.float32)}
    n = pv.shape[0]
    for code in SIM_CODES:
        net = np.empty_like(out[0])
        soc = None
        for sel, prm in month_params:
            idx = np.flatnonzero(sel)
            if not len(idx):
                continue
            st = steps[idx]
            net[:, idx], soc = simulate(pv[:, idx], load[:, idx], np.full(n, code, np.int8), prm,
                                        world.price[st], world.rank[st], world.med[st], world.cal["day_index"][st],
                                        soc)
        out[code] = net
    return out


def mix(variants: Mapping[int, np.ndarray], probs: np.ndarray, step_day: np.ndarray, hard: bool = False) -> np.ndarray:
    """Blend per-strategy nets with per-day probabilities probs (n, D, 4) (enrolled counts as self-consumption);
    step_day (T,) indexes D. hard=True takes the most likely strategy of each home-day."""
    p = probs[:, step_day]                                          # (n, T, 4)
    w = np.stack([p[..., 0], p[..., 1] + p[..., 3], p[..., 2]], -1)
    if hard:
        w = (w == w.max(-1, keepdims=True)).astype(np.float32)
        w /= w.sum(-1, keepdims=True)
    return (w[..., 0] * variants[0] + w[..., 1] * variants[1] + w[..., 2] * variants[2]).astype(np.float32)


CAL_DAYS = 35   # history simulated before a forecast period, for the residual calibration


def residual_bias(world: World, steps: np.ndarray, resid: np.ndarray, lookback: int = 28, lag: int = 2,
                  min_obs: int = 20) -> np.ndarray:
    """Per home and local clock hour: the mean of (meter - two-stage forecast) over meter days [D-lag-lookback+1,
    D-lag] (data PPC has when forecasting day D), as an additive correction (n, len(steps)). 0 where fewer than
    ``min_obs`` readings (profile meters, first days). Corrects the stage-2 inputs' level errors (proxy M1/M2)
    while the simulated battery shape stays the explanation."""
    from scipy import sparse

    n, T = resid.shape
    day = world.cal["day_index"][steps]
    day = day - day[0]
    nd = int(day[-1]) + 1
    hr = np.minimum(world.cal["hour"][steps].astype(np.int64), 23)
    key = day * 24 + hr
    M = sparse.csr_matrix((np.ones(T), (np.arange(T), key)), shape=(T, nd * 24))
    ok = np.isfinite(resid)
    Mt = M.T.tocsr()
    S = np.asarray(Mt @ np.where(ok, resid, 0.0).astype(np.float64).T).T.reshape(n, nd, 24)
    C = np.asarray(Mt @ ok.astype(np.float64).T).T.reshape(n, nd, 24)
    z = np.zeros((n, 1, 24))
    cs, cc = np.concatenate([z, S.cumsum(1)], 1), np.concatenate([z, C.cumsum(1)], 1)
    d = np.arange(nd)
    hi = np.clip(d - lag + 1, 0, nd)
    lo = np.clip(d - lag + 1 - lookback, 0, nd)
    ws, wc = cs[:, hi] - cs[:, lo], cc[:, hi] - cc[:, lo]
    bias = np.where(wc >= min_obs, ws / np.maximum(wc, 1), 0.0)
    return bias[:, day, hr].astype(np.float32)


def portfolio_behaviour(code: np.ndarray, params: Mapping[str, np.ndarray], price: np.ndarray, rank: np.ndarray,
                        med: np.ndarray, day: np.ndarray) -> Callable:
    """A ``behaviour(pv, load, battery, soc0)`` for I5's ``sentinel.eval.portfolio.set_behaviour`` on data-v2:
    pv / load (..., n, T) scenarios -> net import kW, same shape, with each home's GUESSED strategy and parameters.
    soc0 is a fraction of capacity (I5's convention) or None. I5 calls it per chunk of homes with
    ``fleet.battery.take(slice)``: when n < N the chunk is located by matching ``battery.capacity_kwh`` /
    ``p_max_kw`` against ``guessed_battery_arrays`` (build I5's Fleet with those, see BatteryModel.portfolio_inputs)."""
    code = np.asarray(code, np.int8)
    N = len(code)
    cap_enc, p_enc = guessed_battery_arrays(code, params)[:2]
    state = {"next": 0}

    def offset(n: int, battery) -> int:
        if n == N:
            return 0
        if battery is None:
            raise ValueError(f"behaviour: got {n} homes of {N} and no battery arrays to locate them")
        cb, pb = np.asarray(battery[0], float), np.asarray(battery[1], float)
        hint = state["next"] if state["next"] + n <= N else 0
        for o in (hint, *range(0, N - n + 1)):
            if np.array_equal(cap_enc[o:o + n], cb) and np.array_equal(p_enc[o:o + n], pb):
                state["next"] = (o + n) % N
                return o
        raise ValueError("behaviour: home chunk not found; build the Fleet's battery with guessed_battery_arrays")

    def fn(pv, load, battery=None, soc0=None):
        pv = np.asarray(pv, np.float64)
        load = np.broadcast_to(np.asarray(load, np.float64), pv.shape)
        n, T = pv.shape[-2:]
        o = offset(n, battery)
        sl = slice(o, o + n)
        lead = pv.shape[:-2]
        K = int(np.prod(lead)) if lead else 1
        tile = lambda a: np.tile(np.asarray(a)[sl], K)  # noqa: E731
        prm = {k: tile(params[k]) for k in ("cap_kwh", "p_kw", "reserve_frac", "k_top")}
        s0 = None
        if soc0 is not None:
            frac = np.broadcast_to(np.asarray(soc0, float), lead + (n,)).reshape(K * n)
            s0 = frac * prm["cap_kwh"]
        net, _ = simulate(pv.reshape(K * n, T), load.reshape(K * n, T), tile(code), prm, price, rank, med, day, s0)
        return net.reshape(pv.shape)

    return fn


def guessed_battery_arrays(code: np.ndarray, params: Mapping[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """I5's BatteryArrays fields (capacity, p_max, eta_c, eta_d, e_lo, e_hi) from the GUESS (capacity 0 = no battery
    guessed), so I5's Fleet, warm_soc0 and chunking work unchanged on data-v2."""
    code = np.asarray(code)
    cap = np.where(code > 0, np.asarray(params["cap_kwh"], float), 0.0)
    p = np.where(code > 0, np.asarray(params["p_kw"], float), 0.0)
    eta = np.full(len(code), PRIOR["eta"])
    return cap, p, eta, eta.copy(), np.asarray(params["reserve_frac"], float) * cap, cap.copy()


# ============================================================================================ (a) quantile LightGBM
class NetQuantileModel:
    def __init__(self, boosters=None, meta: dict | None = None):
        self.boosters = list(boosters or [])
        self.meta = meta or {}

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None, *, rounds: int = 400, lr: float = 0.05,
            leaves: int = 63, min_leaf: int = 200, seed: int = 0, threads: int = 0, log=print) -> "NetQuantileModel":
        import lightgbm as lgb

        boosters = []
        dtr = lgb.Dataset(X, y, feature_name=list(ROW_FEATURES), free_raw_data=False)
        for tau in TAUS:
            valid, cb = [], []
            if X_val is not None and len(X_val):
                valid = [lgb.Dataset(X_val, y_val, reference=dtr)]
                cb = [lgb.early_stopping(30, verbose=False)]
            b = lgb.train(dict(objective="quantile", alpha=tau, metric="quantile", learning_rate=lr,
                               num_leaves=leaves, min_data_in_leaf=min_leaf, feature_fraction=0.9,
                               bagging_fraction=0.8, bagging_freq=1, seed=seed, verbose=-1, num_threads=threads),
                          dtr, rounds, valid_sets=valid, callbacks=cb)
            log(f"[battery] net quantile tau={tau}: {b.best_iteration or b.current_iteration()} rounds")
            boosters.append(b)
        return cls(boosters, {"features": list(ROW_FEATURES), "taus": list(TAUS), "target": "net import kWh/15min",
                              "seed": seed})

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(N, 3) kWh per 15 min, sorted (no quantile crossing)."""
        q = np.stack([b.predict(X, num_threads=0) for b in self.boosters], axis=-1)
        return np.sort(q, axis=-1).astype(np.float32)

    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for tau, b in zip(TAUS, self.boosters):
            b.save_model(str(d / f"battery_net_q{int(tau * 100):02d}.txt"))
        (d / "battery_net.json").write_text(json.dumps(self.meta, indent=1))

    @classmethod
    def load(cls, d: str | Path) -> "NetQuantileModel":
        import lightgbm as lgb

        d = Path(d)
        return cls([lgb.Booster(model_file=str(d / f"battery_net_q{int(t * 100):02d}.txt")) for t in TAUS],
                   json.loads((d / "battery_net.json").read_text()))
