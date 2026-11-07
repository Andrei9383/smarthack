# Feature I (Imbalance): prosumers no longer cause imbalance

**Prosumer Sentinel · Smarthack XI · PPC**
New app mode: `?mode=imbalance`. This is the core of the pitch.

> English translation of the user's spec `feature-imbalance-prosumatori.md` (card #156). Step numbers ("Spec I step N")
> are identical to the original. Numbers, formulas, sources and assumptions are unchanged; only paths are mapped to this repo:
>
> | Original path | This repo | Card |
> |---|---|---|
> | `ml/data/opcom_pi.py` | `sentinel/data/opcom_pi.py` | I1 #171 |
> | `ml/data/prices.py` | `sentinel/data/prices.py` | X1 #158 |
> | `data/prices.parquet` | `data/raw/prices.parquet` (PZU, exists) + `data/raw/prices_pi.parquet` (PI, I1), joined by `sentinel/data/prices.py` | X1, I1 |
> | `ml/sim/prosumers.py`, `ml/sim/battery_strategies.py`, `ml/sim/config.yaml` | `sentinel/sim/prosumers_v2/**` (strategies also as Rust kernel K8 in `sentinel/kernels/**` + `rust/sentinel_core/src/**`) | I2 #172 |
> | `ml/sim/metering.py` | `sentinel/sim/prosumers_v2/` (metering layer) | I2 #172 |
> | `data-v2/*.parquet` | outputs of the data-v2 generation (new data version; current fleet `sentinel/sim/fleet/` and replay#506 stay byte-identical) | I2 #172 |
> | `ml/eval/baselines_portfolio.py` | `sentinel/eval/baselines_portfolio.py` | I3 #173 |
> | `ml/models/battery/{features,classifier,net_model}.py` | `sentinel/model/battery/` | I4 #174 |
> | `ml/eval/portfolio.py` | `sentinel/eval/portfolio.py` | I5 #175 |
> | `ml/eval/imbalance.py` | `sentinel/eval/imbalance.py` (`a`, `b` frozen in `models/imbalance_ab.json`) | X1 #158 |
> | `ml/replay.py` | `replay.py` → `sentinel/agent/replay.py` | I6 #176 |
> | `db/sentinel.sqlite` | `demo/sentinel.sqlite` (demo stays on replay#506 unless the user approves) | — |
> | `packages/db/src/schema/imbalance.ts`, `packages/api/src/routers/imbalance.ts` | same paths (new files), mode in `apps/web/src/modes/imbalance/**` + `apps/web/src/modes/registry.ts` | I7 #177 |
> | Level 3 LP (step 10) | `sentinel/opt/**` (same LP as G4 #162) | I8 #178 |
>
> M1 PVNet / M2 LoadNet / M4 SiteHealth live in `sentinel/model/` (`forecast.py`, `sitehealth.py`); model reads go
> through `sentinel/judge/adapter.py` (C1 #157). The ~200-scenario rollout is the Rust rollout kernel in `sentinel/kernels/`.

---

## 0. In short

PPC buys today the energy for tomorrow, based on an estimate. Its prosumers (homes with panels, some also with a battery) do things PPC does not see in time. The result: PPC buys too much or too little and pays the difference.

**Sentinel predicts what PPC's prosumers will do tomorrow, for every quarter hour.** That way PPC buys the right quantity already on the day before, corrects in the afternoon at the real price and, optionally, uses the batteries of enrolled customers for what remains.

It needs no hardware and asks nothing of customers. It uses the smart-meter data PPC already receives daily.

---

## 1. The problem, in plain words

### The 18:00 story

1. Yesterday, PPC estimated how much its customers would consume today and bought the energy on the day-ahead market (PZU).
2. The estimate uses a "typical home" curve (standard load profile). The curve does not know that some homes have panels and batteries.
3. Today at 18:00, PPC sees it is short of energy and buys expensively, last minute, on the intraday market (PI).
4. At the same time, prosumers' batteries discharge in the evening and cover part of the same gap.
5. **PPC bought twice.** It sells the surplus cheaply or pays it as imbalance.

### Why now

- **PPC MaxControl** customers get the PZU price for every quarter hour, both for consumption and for energy delivered to the grid (minimum 0, maximum 0.999 lei/kWh, smart meter only). This makes them discharge their battery in the evening, when the price is high. The behaviour exists and is growing.
- In April 2026, ANRE reported **332,684 prosumers, with 3,789 MW** installed.
- ANRE estimated that about **66,000 prosumers have a battery**, with over 600 MW in total.
- The president of AFEER (the suppliers' association) said that every MWh injected into the grid by prosumers brings **costs of 200–375 lei** that suppliers pass on to all customers.

### The real cause

Prosumers are not "doing something wrong". **PPC estimates them as standard homes.** The problem is one of information, so it is solved with a better forecast.

---

## 2. How the solution works: 3 levels

```
Weather (Open-Meteo) ────┐
PZU prices (OPCOM) ──────┼──► M1 solar + M2 load ──► Battery model ──► forecast per home (scenarios)
Meter history (D+1) ─────┘                                                   │
M4: broken homes removed from the calculation ──────────────────────────────┤
                                                                             ▼
                                                  Σ the PPC portfolio (P10 / P50 / P90)
                                                                             │
                     Level 1: PPC buys a day ahead, on PZU, the forecast quantity
                     Level 2: at 15:00 it corrects on PI with the updated forecast
                     Level 3: (optional) discharges the batteries of enrolled customers
                                                                             │
                                                                             ▼
                                     Avoidable cost (lei), Sentinel vs today's method
```

| Level | What it does | What it asks of customers | Status |
|---|---|---|---|
| **1. We predict** | the portfolio forecast for tomorrow, over 96 intervals | nothing | **MUST** |
| **2. We correct** | update at 15:00 and trading on PI | nothing (or a sample of inverters) | SHOULD |
| **3. We coordinate** | the LP discharges the batteries of enrolled customers | consent + bonus | OPTIONAL |

---

## 3. Mini-glossary

| Term | Meaning |
|---|---|
| **PZU** | The Day-Ahead Market (OPCOM). PPC buys here a day ahead. It now has 15-minute prices. |
| **PI** | The Intraday Market (OPCOM). Here you buy "last minute", on the same day. |
| **Position** | How much PPC bought for a 15-minute interval (MWh). |
| **Imbalance** | The difference between what the portfolio consumed and what PPC bought. Paid at the imbalance price. |
| **Avoidable cost** | How much more PPC pays compared with knowing the future exactly. **This is what we reduce.** |
| **P10 / P50 / P90** | Pessimistic / most likely / optimistic. Only 10% of cases fall below P10 or above P90. |
| **Smart meter** | Measures every 15 minutes. The data reaches the supplier the next day (D+1). |
| **Profile meter** | Old meter: only the monthly total, split over intervals by the standard profile. |
| **MaxControl** | PPC contract for prosumers, with the PZU price for every 15 minutes. |

---

## 4. What we reuse from what we already have

| Existing piece | Role in the feature |
|---|---|
| **M1 PVNet** | how much each home's panels produce (P10/P50/P90) |
| **M2 LoadNet** | how much each home consumes (P10/P50/P90) |
| **M4 SiteHealth** | removes homes with a broken panel or inverter from the forecast |
| **The ~200-scenario rollout** (from the P10 commitment) | correct aggregation of uncertainty over the whole portfolio |
| **The simulator's battery model** (`sentinel/sim/fleet/battery.py`) | the basis for the new strategies |
| **OPCOM PZU prices** | the day-ahead purchase price |
| **LP + P10** (`sentinel/opt/`) | level 3 (optional) |
| **The "baseline vs Sentinel" replay** | the demo format |

**What is new:** the battery strategies in the simulator, the metering layer, the battery model, the portfolio aggregation, the avoidable cost and the UI mode.

---

## 5. Implementation, step by step

Each step has: the goal, the files, what you do, what comes out and **how you know it is done**.

### Step 1: Real price data (P1, ~1 h)

**Goal:** real prices for the day-ahead purchase (PZU) and for the same-day one (PI).

**Files:** `sentinel/data/opcom_pi.py` (I1 #171), `sentinel/data/prices.py` (X1 #158)

**1a. PZU per 15 minutes.** OPCOM now publishes PZU per 15 minutes (96 values a day). If for older periods you only have hourly values, repeat each value over the 4 quarters and note this in the README.

**1b. Intraday prices (PI).** OPCOM has no export button, but the "Tranzacții PI" (PI trades) page can be read day by day. We checked that it works for days in 2025 too. The form has the fields `_token`, `ziua`, `tip` (60 or 15 min), `action=trading_for`, `limba=ro`.

```python
# sentinel/data/opcom_pi.py
import time, requests, pandas as pd
from bs4 import BeautifulSoup

URL = "https://www.opcom.ro/rapoarte-pi-rezultate-pi/ro"

def session_with_token():
    s = requests.Session()
    html = s.get(URL, timeout=30).text
    token = BeautifulSoup(html, "html.parser").find("input", {"name": "_token"})["value"]
    return s, token

def fetch_day(s, token, day: str, tip: str = "15") -> pd.DataFrame:
    """day = 'dd/mm/yyyy'. Returns the trades: interval, side, price, volume."""
    r = s.post(URL, data={"_token": token, "ziua": day, "tip": tip,
                          "action": "trading_for", "limba": "ro", "buton": "Refresh"}, timeout=30)
    soup = BeautifulSoup(r.text, "html.parser")
    rows, current = [], None
    for tr in soup.select("table tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) == 4:            # row with the interval label (rowspan)
            current, side, price, vol = tds
        elif len(tds) == 3 and current:
            side, price, vol = tds
        else:
            continue
        try:
            rows.append({"interval": current, "side": side,
                         "price": float(price.replace(".", "").replace(",", ".")),
                         "volume": float(vol.replace(".", "").replace(",", "."))})
        except ValueError:
            pass
    df = pd.DataFrame(rows)
    df["day"] = day
    return df

def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-weighted average price per interval = the cost of the last-minute purchase."""
    g = df.groupby(["day", "interval"])
    return g.apply(lambda x: (x.price * x.volume).sum() / x.volume.sum()).rename("pi_vwap").reset_index()

# run: 365 days, with a pause between requests (without loading the OPCOM server)
# s, tok = session_with_token()
# for d in pd.date_range("2025-01-01", "2025-12-31"):
#     fetch_day(s, tok, d.strftime("%d/%m/%Y")); time.sleep(1)
```

Check on 2–3 days that the parser catches the interval label correctly: the rows use `rowspan`, and the label appears only on the first row of each block. If the 15-min product has no data for 2025, use `tip="60"`.

**Intervals without trades:** set `pi_vwap = pzu`. The correction then costs 0 relative to PZU, a conservative assumption.

**Output:** `data/prices.parquet` with the columns `ts, pzu_lei_mwh, pi_vwap_lei_mwh, pi_volume_mwh`. (In this repo: the PI download is committed as `data/raw/prices_pi.parquet` + a `data/raw/MANIFEST.json` entry so `make reproduce` stays offline; `sentinel/data/prices.py` serves the joined 15-min table from it and `data/raw/prices.parquet`.)

**Done when:** a test year has 35,040 rows (96 × 365), with no gaps in PZU.

### Step 2: The prosumer simulator, extended (P2, ~1 h)

**Goal:** realistic prosumers, with different contracts and behaviours, some hidden from the model.

**Files:** `sentinel/sim/prosumers_v2/` — prosumers, battery strategies, config (I2 #172; the original names were `prosumers.py`, `battery_strategies.py`, `config.yaml`)

**2a. New attributes on each home:**

| Attribute | Values | Does PPC know it? |
|---|---|---|
| `kwp` | installed power | **yes** (from the connection) |
| `contract` | `standard` / `maxcontrol` | **yes** (it is its contract) |
| `meter` | `smart` / `profile` | **yes** |
| `battery_declared` | true / false | **partly** (not everyone declares) |
| `has_battery` | true / false | **no** |
| `strategy` | `none` / `self_consumption` / `price_driven` / `enrolled` | **no** |
| `cap_kwh`, `p_kw`, `reserve_frac`, `k_top` | the battery parameters | **no** |

**2b. Configuration, with real percentages:**

```yaml
# sentinel/sim/prosumers_v2/config.yaml
n_homes: 5000
share_battery: 0.20          # ~66,000 / 332,684 per ANRE; replace with your figure
share_maxcontrol_of_battery: 0.25   # unknown, we also run 0.10 and 0.50
share_smart_meter: 0.60      # Rețele Electrice: 50–59% (2024 targets)
share_battery_declared: 0.70
share_enrolled: 0.10         # only for level 3
battery_cap_kwh: [5, 15]
battery_p_kw: [3, 5]
reserve_frac: [0.10, 0.40]
k_top_intervals: [8, 20]     # how many "expensive" quarters a MaxControl customer targets
noncompliance_prob: 0.10     # days when the customer does not follow the strategy
```

**2c. The battery strategies** (powers in kW, 15-minute step):

```python
# sentinel/sim/prosumers_v2/battery_strategies.py  (numpy reference; Rust kernel K8 copy per team/infra.md)
DT = 0.25  # hours

def self_consumption(pv, load, soc, cap, p_max, reserve, eta=0.95):
    """Standard contract: charge from surplus, discharge into the home. Does not export from the battery."""
    surplus = pv - load
    if surplus > 0:
        c = min(surplus, p_max, (cap - soc) / (eta * DT))
        return c, 0.0
    d = min(-surplus, p_max, max(soc - reserve * cap, 0) * eta / DT)
    return 0.0, d

def price_driven(pv, load, soc, cap, p_max, reserve, price, top_mask, med_price, eta=0.95):
    """MaxControl: charge from the sun, discharge at maximum in the expensive quarters (export included)."""
    surplus = pv - load
    if surplus > 0:                       # the sun goes into the battery first
        c = min(surplus, p_max, (cap - soc) / (eta * DT))
        return c, 0.0
    if top_mask:                          # expensive quarter: discharge at maximum, and export
        d = min(p_max, max(soc - reserve * cap, 0) * eta / DT)
        return 0.0, d
    if price >= med_price:                # price above the median: cover only the home
        d = min(-surplus, p_max, max(soc - reserve * cap, 0) * eta / DT)
        return 0.0, d
    return 0.0, 0.0                       # low price: take from the grid, keep the battery for the evening

def grid_flows(pv, load, c, d):
    net = load - pv + c - d               # > 0 import, < 0 export
    return max(net, 0.0), max(-net, 0.0)
```

`top_mask` is computed once a day from tomorrow's PZU prices: the most expensive `k_top` quarters for that home. On "noncompliance" days, the home uses `self_consumption`.

**Output:** `data-v2/homes.parquet` and `data-v2/home_15min.parquet`, with the columns `ts, home_id, pv_kw, load_kw, batt_kw, soc_kwh, import_kwh, export_kwh`.

**Done when:** in the chart of a summer day, MaxControl homes visibly export in 18–21, and standard ones do not.

### Step 3: The metering layer, i.e. what PPC sees (P2, ~30 min)

**Goal:** the model must see **only** what PPC sees in reality.

**File:** `sentinel/sim/prosumers_v2/` metering module (I2 #172; originally `metering.py`)

- `smart` homes give the 15-minute curve (import, export), available **the next day** (`available_at = day + 1, 00:00`).
- `profile` homes give only the monthly import and export totals.
- **At settlement** (step 7), `profile` homes are settled on the standard profile curve, not on what they actually did. This is the real rule: where the meter does not record per interval, the distributor splits consumption by profile. So improvements on them **do not show up in money**, and the reporting stays honest.

**Output:** `data-v2/meter_observed.parquet` (smart only) and `data-v2/meter_monthly.parquet`.

**Done when:** no training file contains `pv_kw`, `batt_kw`, `soc_kwh` or `strategy`. Add a test that checks this (`tests/test_data_v2.py`).

### Step 4: The baselines, i.e. PPC today (P4, ~1 h)

**File:** `sentinel/eval/baselines_portfolio.py` (I3 #173)

| Policy | How it computes the position | What it shows |
|---|---|---|
| `profile` | standard load profile × expected consumption, minus a typical monthly solar curve × kWp, no weather | PPC today |
| `persistence` | "tomorrow like a week ago", from the meter data (D+1) | simple baseline with data |
| `forecast_only` | M1 + M2 per home, without the battery model | what the forecast alone brings |
| `sentinel` | M1 + M2 + the battery model + M4 | the complete solution |

The standard profile coefficients ("Household, urban zone" or "rural") are published by the distributors. For the Rețele Electrice area, look for their profile. If you do not find it quickly, use the public example from Distribuție Energie Electrică România.

**Done when:** all 4 policies write a position for every interval of the test year.

### Step 5: The battery model (P3, ~2 h)

**Goal:** for each home and each quarter hour of tomorrow, how much it will take from or give to the grid, including the battery's effect.

**Files:** `sentinel/model/battery/features.py`, `sentinel/model/battery/classifier.py`, `sentinel/model/battery/net_model.py` (I4 #174)

**5a. The fallback first (fastest, works for sure).** A quantile LightGBM (0.1 / 0.5 / 0.9), directly on the home's net export.

Features:
- M1 P50 (solar) and M2 P50 (load) for that interval
- the interval's PZU price and **its rank within the day**
- hour, day of week, month
- `kwp`, `contract`, `battery_declared`
- net import and export from the previous day and from a week ago, at the same hour (smart homes only)

**5b. Then the main, two-stage, explainable variant.**

1. **Strategy classifier** (LightGBM), on the last 28 days of meter data. Per-home features:
   - the correlation between export and the PZU price
   - what share of the day's export falls in 18–21
   - midday export vs the expected solar surplus (a battery "eats" it)
   - evening import vs expected consumption (a battery reduces it)
   - the gap between the hour of peak export and the hour of the highest price
   - `contract`, `battery_declared`, `kwp`

   **Output:** the strategy + the confidence.

   **For training** use the homes in the train split. In a real pilot, labels come partly from what PPC knows (the MaxControl contract, declared batteries) and from enrolled customers.
2. **Simulation with the guessed strategy:** M1 + M2 + tomorrow's PZU prices go through the functions in `battery_strategies.py` (the K8 kernel), with the estimated parameters. The output is the net import and export over 96 intervals, for each scenario.

**Splits (no leakage):** train 2022–2023, validation 2024, test 2025. 30% of homes never appear in train.

**Metrics:**
- `strategy_f1` (macro) on unseen homes
- `net_nmae`, per home and **for the portfolio**
- separately, `net_nmae_17_21`

**Done when:** at portfolio level, the battery variant beats `forecast_only` in the 17–21 window.

### Step 6: Portfolio aggregation (P4, ~1.5 h)

**Goal:** PPC's total position for every quarter hour, with the correct uncertainty.

**File:** `sentinel/eval/portfolio.py` (I5 #175)

1. **Do not add the quantiles directly.** The sum of the P90s is not the P90 of the sum.
2. Generate K scenarios (200, or 50 if slow):
   - for solar, **a single quantile level shared per region** (the sun is the same for neighbouring homes)
   - for load, independent levels for each home
   - each scenario goes through the battery model
3. In each scenario: `portfolio[t] = Σ (import − export)` over all homes, in MWh.
4. From the scenarios take P10 / P50 / P90 for each interval.
5. **Conformal calibration on the total:** on the validation days, measure how often reality falls outside the band and widen the band until it covers about 80%.
6. **How much PPC buys a day ahead:** the quantile `q* = a / (a + b)` of the scenarios, where `a` and `b` are the penalties from step 7. With `a = b`, this means P50.
7. **The 15:00 correction (level 2):** redo steps 2–6 for the 16:00–24:00 intervals, with the updated weather (less noise at a short horizon). The quantity traded on PI is `traded_id[t] = new_quantile[t] − bought_da[t]`.

The scenario part is the same kind of computation as the P10 rollout, so the Rust kernel can be reused if it is ready.

**Output:** `portfolio_forecast.parquet` (`issue_time, ts, policy, p10, p50, p90`) and `positions.parquet` (`ts, policy, bought_da, traded_id, actual`).

**Done when:** on validation, the portfolio P10–P90 band catches 78–82% of the real values.

### Step 7: The avoidable cost (P4, ~45 min)

**Goal:** a single number in lei, comparable across policies. **You use the same function for wind too.**

**File:** `sentinel/eval/imbalance.py` (X1 #158)

**The logic, in words:**
- PPC pays PZU on what it bought a day ahead.
- It pays the PI price on what it corrected on the day.
- What remains uncovered goes to imbalance: the shortfall is paid at PZU × (1 + a), the surplus is sold at PZU × (1 − b).

We subtract the "perfect" cost (if it had known the future and bought everything on PZU). **What remains is the avoidable cost.**

```python
# sentinel/eval/imbalance.py
import numpy as np

def avoidable_cost(actual, bought_da, traded_id, pzu, pi, a=0.5, b=0.5):
    """All are vectors over 15-min intervals. Quantities in MWh, prices in lei/MWh.
    actual    = the portfolio's real net consumption (import − export)
    bought_da = how much PPC bought on PZU a day ahead
    traded_id = how much it bought (+) or sold (−) on PI on the day
    """
    dev = actual - bought_da - traded_id          # what remained uncovered
    short = np.clip(dev, 0, None)                  # PPC had too little
    long = np.clip(-dev, 0, None)                  # PPC had too much
    intraday = traded_id * (pi - pzu)              # what the correction cost relative to PZU
    imbalance = a * pzu * short + b * pzu * long   # the imbalance penalty (proxy)
    return {
        "avoidable_lei": float(intraday.sum() + imbalance.sum()),
        "intraday_lei": float(intraday.sum()),
        "imbalance_lei": float(imbalance.sum()),
        "abs_dev_mwh": float(np.abs(dev).sum()),
    }
```

**Calibrating `a` and `b`:**
1. Choose `a ≥ b`, because a shortfall is usually more expensive than a surplus.
2. Tune them so that the `profile` policy comes out at **200–375 lei per MWh injected by prosumers**, the range publicly reported by suppliers.
3. Then **freeze them** and do not touch them again. (In this repo: `models/imbalance_ab.json`.)

**Why it is fair:** with `a = b`, the reduction percentage does not depend on the chosen value. With `a ≠ b`, it depends only on their ratio. Put the percentage on the slide as the headline number.

**Additional metrics:**
- `abs_dev_mwh`: the wrong energy, without any price, so the most solid number
- `double_cover_mwh_18_21`: how much PPC bought beyond need between 18 and 21 (exactly the 18:00 story)
- `lei_per_mwh_injected`: the avoidable cost divided by the MWh exported by prosumers

**Done when:** the function has tests on small hand-computed cases (perfect forecast = 0 lei, shortfall only, surplus only).

### Step 8: Replay and metrics (P4 + P5, ~45 min)

**File:** `replay.py` → `sentinel/agent/replay.py` (extended; I6 #176)

```bash
python replay.py --mode imbalance \
  --policy profile,persistence,forecast_only,sentinel \
  --out demo/sentinel.sqlite
```

(In this repo `replay.py` does not yet have `--mode`. Write to `$LAB_OUTPUT_DIR`, not over the demo DB, unless the user
approves.)

New `LAB_METRIC` lines:

```
LAB_METRIC avoidable_cost_lei=...
LAB_METRIC lei_per_mwh_injected=...
LAB_METRIC abs_dev_mwh=...
LAB_METRIC double_cover_mwh_18_21=...
LAB_METRIC strategy_f1=...
LAB_METRIC portfolio_nmae=...
LAB_METRIC portfolio_coverage_p10_p90=...
```

**Sensitivity analysis:** run with `share_maxcontrol_of_battery` = 0.10 / 0.25 / 0.50. On the slide show that Sentinel wins in all three. (One lab grid.)

### Step 9: Database, API and UI (P5, ~2 h, in parallel from the start with mock data)

(I7 #177; schema change = contract change: `lab announce` + `team/decisions.md`)

**9a. Drizzle** (`packages/db/src/schema/imbalance.ts`):

```ts
export const portfolioForecast = sqliteTable("portfolio_forecast", {
  issueTime: text("issue_time").notNull(),
  ts: text("ts").notNull(),
  policy: text("policy").notNull(),          // profile | persistence | forecast_only | sentinel
  p10: real("p10"), p50: real("p50"), p90: real("p90"),
});

export const positions = sqliteTable("positions", {
  ts: text("ts").notNull(),
  policy: text("policy").notNull(),
  boughtDa: real("bought_da"), tradedId: real("traded_id"), actual: real("actual"),
  devMwh: real("dev_mwh"), intradayLei: real("intraday_lei"), imbalanceLei: real("imbalance_lei"),
});

export const siteStrategy = sqliteTable("site_strategy", {
  siteId: text("site_id").primaryKey(),
  strategy: text("strategy"),                // none | self_consumption | price_driven
  confidence: real("confidence"),
});
// + the `contract` and `meter` columns on the existing assets table
```

**9b. oRPC** (`packages/api/src/routers/imbalance.ts`):
- `getImbalanceDay({ date, policy })`: the 96-interval series (band, bought, actual) and the totals in lei
- `getPolicyComparison({ from, to })`: the cost per policy (for the waterfall)
- `getStrategyBreakdown()`: how many homes have no battery, how many use it for self-consumption, how many follow the price

**9c. The `?mode=imbalance` mode** (`apps/web/src/modes/imbalance/**`, registered in `apps/web/src/modes/registry.ts`):
1. **Main chart:** the P10–P90 band, the "bought" line (stepped) and the "actual" line. The 18–21 window highlighted.
2. **KPIs:** avoidable cost today, lei/MWh injected, MWh bought for nothing in 18–21. Each shows "today's method" vs "Sentinel", with the global Without/With toggle.
3. **Waterfall:** `profile` → + the M1/M2 forecast → + the battery model → + the 15:00 correction → (+ enrolled batteries). Shows what each step brings.
4. **"What Sentinel learned":** how many homes are in each strategy, with the mean confidence.
5. **Overview:** the Imbalance card (now with `[X]` / `[Y]`) gets the real numbers.

**Done when:** switching the Without/With toggle changes all the numbers on the page in under 1 second.

### Step 10 (optional): Levels 2 and 3, in full

- **Live sample (level 2):** 5% of homes give inverter data every 15 minutes. From them, estimate the portfolio in real time (upscaling per region and strategy), then correct on PI at 15:00 with real data, not just with updated weather.
- **Enrolled batteries (level 3):** the existing LP (`sentinel/opt/`, I8 #178) gets a new objective: minimise the portfolio's avoidable cost. The feeder limit and the customer reserve remain rules that are never broken. When the 15:00 forecast shows a shortfall at 18–21, the LP schedules the discharge of enrolled batteries instead of buying on PI. The commitment stays at P10.

---

## 6. The data: what is real and what is synthetic

| Data | Source | Type |
|---|---|---|
| Weather | Open-Meteo | real |
| PZU per 15 min | OPCOM | real |
| PI (intraday) | OPCOM, "Tranzacții PI" (scraper) | real |
| Standard load profile | coefficients published by distributors | real |
| Number of prosumers, MW, % with battery | ANRE | real, for calibration |
| % smart meters | Rețele Electrice | real, for calibration |
| MaxControl rules | the PPC offer | real |
| Imbalance price | PZU × (1 ± a), calibrated on the AFEER figure | **proxy**, marked as such |
| Each home's curves | the simulator | **synthetic** |
| Contract, strategy, meter per home | sampled from the real percentages | **synthetic** |

**For a real pilot:** Rețele Electrice already makes data from over 1.6 million smart meters available to suppliers daily. In the same format, they replace the synthetic data without code changes.

---

## 7. ~1-minute demo (within the 3-minute pitch)

1. **Friday, 18:00, the `profile` policy.** The "bought" line is below "actual" all evening. PPC buys last minute on PI, then the batteries discharge and PPC ends up long. On screen: `[X] lei`.
2. **Switch to Sentinel.** The morning forecast already showed the batteries discharging at 18:00, and the "bought" line follows reality. On screen: `[Y] lei` and "−[Z]%".
3. **The waterfall:** where the gain comes from (forecast, battery model, correction).
4. **"What Sentinel learned":** "of 1,000 homes with a battery, 250 follow the price, and we found them from meter data alone".
5. **The line:** *"PPC cannot control 330,000 prosumers, but it can predict them."*

---

## 8. Jury questions, with answers

**"You invented the behaviour and then learned it. Isn't that circular?"**
The rules come from the real MaxControl offer and from battery physics. Each home has random parameters and days when it does not follow the rule. The model does not see the strategy, only the meter (D+1). The result holds at 10%, 25% and 50% MaxControl customers. In a pilot, real smart-meter data goes straight in.

**"Where do you get the imbalance price?"**
The PZU and PI prices are real, from OPCOM. Only the final penalty is approximated, calibrated on the cost publicly reported by suppliers. The reduction percentage does not depend on the chosen level.

**"Can you control customers' batteries?"**
Levels 1 and 2 control nothing, they only predict. Level 3 is optional, only with the customer's consent and with a bonus, through the inverter manufacturer's app. PPC already uses the same notification channel (the myPPC app, e.g. in the free-hours pilot).

**"What do you do with homes without a smart meter?"**
They are settled on the profile, so we do not count them in the gain. We include them in the physical forecast, but report money only on smart-metered homes.

**"Isn't someone already doing this?"**
Battery aggregators (e.g. Delta Green, with E.ON in Czechia) control enrolled batteries in real time, with a device installed in each home. Sentinel **predicts the whole portfolio without hardware**, including homes without a battery, and only then coordinates the batteries.

---

## 9. To decide or to ask the PPC mentors

- [ ] The share of homes with a battery in the PPC portfolio (the team knows it)
- [ ] How many PPC prosumers are on MaxControl
- [ ] Whether PPC already uses the daily smart-meter curves in its forecast or still the standard profile
- [ ] Whether the PPC wind parks and the supplier are in the same balancing portfolio (for the link with the wind pillar)
- [ ] The time at which PPC usually makes its intraday correction

---

## 10. Final checklist

- [ ] `data/prices.parquet` with PZU + PI for the whole test year (here: `data/raw/prices.parquet` + `data/raw/prices_pi.parquet`)
- [ ] `data-v2` with contracts, strategies and meters; the "no hidden columns" test passes
- [ ] The 4 policies write positions for the whole test year
- [ ] The battery model: the fallback done; the classifier if time remains
- [ ] The portfolio band calibrated (78–82% coverage on validation)
- [ ] `avoidable_cost` with tests; `a` and `b` calibrated and frozen
- [ ] Full replay + `LAB_METRIC`s + MaxControl sensitivity
- [ ] The `?mode=imbalance` mode + the Overview card with real numbers
- [ ] Every number on a slide has a run ID and the label "synthetic RO / real RO / proxy"

---

## Sources

- [PPC MaxControl: PZU price per 15 minutes, minimum 0, maximum 0.999 lei/kWh](https://stiricraiova.ro/2026/07/15/ppc-energie-lanseaza-ppc-maxcontrol-oferta-pentru-prosumatori-cu-plata-soldului-la-30-de-zile-si-protectie-impotriva-preturilor-negative/)
- [The PPC contract indexed to PZU, smart meters only](https://economedia.ro/?p=359078)
- [ANRE: 332,684 prosumers, 3,789 MW (April 2026)](https://www.capital.ro/legea-prosumatorului-intra-in-linie-dreapta-spre-promulgare-ccr-a-publicat-motivarea-deciziei-iar-noile-reguli-aduc-schimbari-importante.html)
- [Romanian prosumers' batteries (ANRE estimate ~66,000)](https://balkangreenenergynews.com/prosumers-in-romania-have-much-more-battery-power-than-energy-companies/)
- [AFEER: 200–375 lei per MWh injected by prosumers](https://financialintelligence.ro/laurentiu-urluescu-afeer-pentru-fiecare-mwh-introdus-in-retea-de-prosumatori-exista-costuri-intre-200-si-375-lei-care-sunt-socializate-ultima-luna-complet-confirmata-pentru-deco/)
- [Rețele Electrice: daily meter data for suppliers](https://economedia.ro/?p=224249)
- [Rețele Electrice: smart meter programme 2025](https://reteleelectrice.ro/wp-content/uploads/comunicat-rer-program-smart-meter-2025.pdf)
- [Smart meter targets by region (2024)](https://economedia.ro/?p=186991)
- [The load-profile rule for meters without interval recording](https://alba24.ro/regulamentul-de-furnizare-a-energiei-electrice-va-fi-modificat-de-anre-facturare-lunara-esalonari-la-plata-si-alte-noutati-953975.html)
- [Example of a specific load profile (PDF)](https://distributie-energie.ro/wp-content/uploads/2024/08/Profil-specific-de-consum-PSC-tip-Casnic-Zona-Urban.pdf)
- [PPC Energie: the free-hours pilot, announced in myPPC](https://economedia.ro/?p=346674)
- [Delta Green: battery aggregation for suppliers](https://www.energynomics.ro/en/romanias-imbalance-problem-is-becoming-a-household-opportunity-how-can-delta-green-help)
- [OPCOM: PI trades](https://www.opcom.ro/rapoarte-pi-rezultate-pi/ro)
