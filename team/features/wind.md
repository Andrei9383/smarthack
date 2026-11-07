# Feature W (Wind): the PPC park knows what it produces and what is broken

**Prosumer Sentinel · Smarthack XI · PPC Renewables**
The wind pillar. Extends the app's **Production** and **O&M** modes.

> English translation of the user's spec `feature-eolian-ppc-renewables.md` (card #156). Step numbers ("Spec W step N")
> are identical to the original. Numbers, formulas, sources and assumptions are unchanged; only paths are mapped to this repo:
>
> | Original path | This repo | Card |
> |---|---|---|
> | `ml/models/windpower/` (M5), `binned.py` | `sentinel/model/windpower/` (incl. `binned.py`) | W1 #164 |
> | `ml/models/windpower/availability.py` | `sentinel/model/windpower/availability.py` | W2 #165 |
> | `ml/eval/wind_portfolio.py` | `sentinel/eval/wind_portfolio.py` | W3 #166 |
> | `ml/agent/workorders.py` | `sentinel/agent/triage.py` (extended) | W4 #167 |
> | `ml/agent/maintenance_window.py` | `sentinel/agent/maintenance_window.py` | W6 #169 |
> | `ml/opt/park_battery.py` | `sentinel/opt/park_battery.py` | W5 #168 |
> | `ml/eval/imbalance.py` (`avoidable_cost`) | `sentinel/eval/imbalance.py` | X1 #158 |
> | `data/prices.parquet` | `data/raw/prices.parquet`, read via `sentinel/data/prices.py` | X1 #158, I1 #171 |
> | M6 WindNBM | `sentinel/model/wind/` (`nbm.py`, `windmon_rules.py`) | — |
> | M7 WindFailure | `sentinel/model/windfailure/` | T6 #144 |
> | Wind simulator | `sentinel/sim/wind/`, faults in `sentinel/sim/wind_faults/` | — |
> | CARE-to-Compare benchmark | `sentinel/eval/care/` | — |
> | ENTSO-E national wind | `data/raw/entsoe_ro.parquet` | — |
> | `ml/replay.py` | `replay.py` → `sentinel/agent/replay.py` | — |
> | `db/sentinel.sqlite` | `demo/sentinel.sqlite` (demo stays on replay#506 unless the user approves) | — |
> | Drizzle tables / UI | `packages/db/src/schema/**`, `packages/api/src/routers/{production,om}.ts`, `apps/web/src/modes/{production,om}/**` | W7 #170 |
>
> Model reads (M5/M6/M7) go through `sentinel/judge/adapter.py` (C1 #157).

---

## 0. In short

PPC has large wind parks in Dobrogea and Moldova. Every day, PPC announces how much they will produce tomorrow and sells the energy. If the wind or the turbines do not do what was announced, it pays the difference. In addition, broken turbines lose production for as long as nobody notices, and operators are flooded with false alerts.

**Sentinel tells the PPC parks four things:**
1. **How much to sell tomorrow**, without counting on the broken turbines.
2. **Which turbine to repair first**, by money lost.
3. **On which day to repair it**, so the stop costs as little as possible.
4. **How to use the park's battery** to cover what the forecast missed.

It uses the same logic and the same cost function as the prosumer feature. It does not control any turbine: it only predicts, classifies and recommends.

---

## 1. The problem, in plain words

### What PPC has

According to the v3 document:
- The Fântânele-Cogealac-Grădina park (Constanța): **600 MW**, from 240 GE turbines of 2.5 MW.
- An **84 MW** park in Dobrogea, bought from Lukoil.
- The Deleni park (Vaslui): **140 MW**, from 23 GE Vernova turbines of 6.1 MW.
- PPC is adding a **45.72 MW / 91.44 MWh** battery to a park in Dobrogea.

### Three real problems

1. **Imbalance.** The producer announces a day ahead how much it will produce. The real wind differs, and the difference is paid. The Romanian Wind Energy Association (RWEA) said as early as 2017 that Romania has the highest balancing costs for wind in the EU. In 2024 it said these costs are still very high.
2. **The broken turbine is counted.** If a turbine is stopped and the forecast still counts it, PPC promises energy that does not come. It pays twice: the lost production and the imbalance.
3. **Too many alerts.** Rules of the "high wind + zero power" kind fire on any stop, including intentional ones (curtailment) and maintenance. In the WindMon test (the previous PPC hackathon), the rules gave **~15,000 alerts in 16 days**, and 70% were data gaps from only 4 offline turbines.

### The real cause

Nobody knows what each turbine **should** produce with the wind it has. With a model for that, the forecast, the alerts and the planning become measurable.

---

## 2. How it works: 4 decisions

```
Wind 100 m, direction, temperature, pressure (Open-Meteo) ──► M5: how much each turbine produces (P10/P50/P90)
SCADA data every 15 min (power, wind, temperatures) ────────► M6: is the turbine normal? if not, what is wrong?
History of M6 residuals ────────────────────────────────────► M7: failure risk in 7 / 14 days
                                                                    │
                                     availability of each turbine tomorrow (M6 + M7)
                                                                    │
    ┌─────────────────────────────┬────────────────────────────────┼──────────────────────────────┐
    ▼                             ▼                                ▼                              ▼
D1. How much to sell    D2. What to repair first      D3. Which day to repair      D4. The park battery
(M5 × availability)     (M6 + lei lost/day)           (M7 + M5 + prices)           (LP on the forecast error)
    │                                                                                         │
    └──────────────────────────► Avoidable cost (lei), same function as for prosumers ◄────────┘
```

| Decision | Models | Status | Priority |
|---|---|---|---|
| **D1. How much to sell tomorrow** | M5, M6, M7 | M5 is "optional", must be done | **MUST** |
| **D2. What to repair first** | M6 | almost done | **MUST** (~30 min) |
| **D3. Which day to repair** | M7, M5, PZU | M7 optional | SHOULD, only if M7 works out |
| **D4. The park battery** | M5, LP | the LP exists on the feeder | SHOULD |

---

## 3. Mini-glossary

| Term | Meaning |
|---|---|
| **Notification / position** | How much PPC announced and sold for each 15-minute interval. |
| **Imbalance** | The difference between what the park produced and what PPC sold. It is paid. |
| **Avoidable cost** | How much PPC loses compared with knowing production exactly. **This is what we reduce.** |
| **Power curve** | How much power the turbine gives at each wind speed. |
| **Availability** | What part of the turbine can produce tomorrow: 1 = healthy, 0 = stopped. |
| **Curtailment** | The turbine is intentionally limited (by the dispatcher). Not a fault. |
| **Icing** | Ice on the blades: produces less or stops. |
| **Yaw** | Rotating the turbine head into the wind. If misaligned, it loses production without an error. |
| **Residual** | The difference between what it *should* do (from the model) and what it *does*. A slowly growing residual announces a fault. |
| **Work order** | The technician's ticket: turbine, probable fault, confidence, lei lost per day. |

---

## 4. What we reuse

| Existing piece | Role |
|---|---|
| **M6 WindNBM** (`sentinel/model/wind/`) | the state of each turbine and the problem type (precision 0.85 / recall 0.82, vs 0.53 / 0.44 for the WindMon rules; alerts per turbine-day 0.698 → 0.114) |
| **M5 WindPower** (`sentinel/model/windpower/`) | the per-turbine and per-park forecast. *Must be finished.* |
| **M7 WindFailure** (`sentinel/model/windfailure/`) | failure risk. *Only if the early warning is real.* |
| **The wind simulator** (`sentinel/sim/wind/`) | 3 parks at real coordinates, ~120 turbines, the PPC SCADA schema, faults injected with the exact start time |
| **`sentinel/eval/imbalance.py`** | `avoidable_cost`, the same function as for prosumers |
| **`data/raw/prices.parquet`** (via `sentinel/data/prices.py`) | PZU + PI (OPCOM), from the prosumer feature |
| **The LP (cvxpy + HiGHS)** | the basis for the battery plan |
| **The triage formula from v3** | score = P(fault) × lei lost × days until visit |

---

## 5. Implementation, step by step

### Step 1: Prices (P1, 0 min, already done)

Use `data/raw/prices.parquet` (through `sentinel/data/prices.py`) from the prosumer feature (PZU + PI from OPCOM).

For the days after tomorrow (step 6), PZU is not yet known. Use the monthly forward prices from the OPCOM front page (ROPEX_FM), or the profile of the same day last week.

### Step 2: M5 WindPower, finished (P3, ~1.5 h)

**Goal:** for each turbine and each quarter hour of the next 1–24 h, how much it produces (P10/P50/P90).

**Files:** `sentinel/model/windpower/` (card W1 #164)

**Inputs:**
- the 100 m wind (u, v) brought to hub height with the shear exponent
- air density from temperature and pressure: `v_norm = v_hub × (ρ / ρ0)^(1/3)`
- the direction sector, a turbulence proxy, the season
- a per-turbine embedding

**Model:** probabilistic MLP per turbine type (2.5 MW or 6.1 MW), with pinball loss.

**Fallback (30 min, if there is no time):** the empirical power curve on wind bins (0.5 m/s step), with the 10 / 50 / 90 quantiles computed from the training data per bin and per turbine type.

```python
# sentinel/model/windpower/binned.py (fallback)
import numpy as np, pandas as pd

def fit_binned(df, bin_ms=0.5):
    """df: v_norm (m/s), power_kw, model (2.5MW / 6.1MW), only steps with the turbine healthy."""
    df = df.assign(bin=(df.v_norm // bin_ms) * bin_ms)
    q = df.groupby(["model", "bin"]).power_kw.quantile([0.1, 0.5, 0.9]).unstack()
    q.columns = ["p10", "p50", "p90"]
    return q.reset_index()

def predict_binned(curves, model, v_norm, bin_ms=0.5):
    b = (np.asarray(v_norm) // bin_ms) * bin_ms
    c = curves[curves.model == model].set_index("bin")
    return c.reindex(b, method="nearest")[["p10", "p50", "p90"]].to_numpy()
```

**Train only on the steps where M6 says "normal".** Otherwise the model also learns the faults.

**Baselines:**
- `manufacturer_curve`: the manufacturer's generic curve applied to the forecast wind
- `persistence`: the power 24 h ago

**Metrics:** `val_nmae`, `val_pinball` per turbine and per park. Separately, validation on real national wind production (ENTSO-E, already in the repo: `data/raw/entsoe_ro.parquet`).

**Done when:** at park level, M5 beats both baselines on validation.

### Step 3: Tomorrow's availability (P4, ~45 min)

**Goal:** the forecast no longer counts the broken turbines.

**File:** `sentinel/model/windpower/availability.py` (card W2 #165)

| What M6 / M7 says | Availability for tomorrow |
|---|---|
| normal | 1.0 |
| **fault (stopped)** + open work order | **0** until the estimated repair time, then 1 |
| underperformance / yaw | 1 − the measured mean loss (e.g. 0.9) |
| icing risk (from the weather forecast: temp. < ~1 °C + high humidity or precipitation) | reduced only on the risk intervals |
| high risk from M7 | 1 − P(fault in the next 24 h) |
| curtailment | 1 (physically fine; the limiting comes from the dispatcher and is handled separately) |
| broken wind sensor / missing data | 1, but use the weather wind; marked "uncertain" |

**The park forecast:**
1. Generate scenarios with **one wind quantile level shared across the park**, because neighbouring turbines see the same wind.
2. In each scenario: turbine production × its availability, then the sum over the park, plus the wake correction between turbines per direction sector.
3. From the scenarios take P10 / P50 / P90 for the park.
4. Calibrate conformally **on the park total**.

**Done when:** on the test days with stopped turbines, `sentinel` has a lower error than `windpower_only` (M5 without availability).

### Step 4 (D1): How much to sell tomorrow + avoidable cost (P4, ~45 min)

**File:** `sentinel/eval/wind_portfolio.py` (card W3 #166)

**Policies compared:**

| Policy | How much it sells a day ahead |
|---|---|
| `manufacturer_curve` | manufacturer curve × forecast wind, all turbines available |
| `persistence` | the production 24 h ago |
| `windpower_only` | M5, all turbines available |
| `sentinel` | M5 × the availability from step 3 + a 15:00 correction on PI |

**How much it sells:** the quantile `q* = a / (a + b)` of the scenarios (with `a = b`, P50). At 15:00, with updated wind, it sells or buys the difference on PI.

**The cost:** the same function as for prosumers. **Watch the sign.** The function is written for consumption, so production goes in **with a minus**, as negative demand:

```python
from sentinel.eval.imbalance import avoidable_cost

res = avoidable_cost(
    actual=-prod_mwh,          # the park's real production, with a minus
    bought_da=-sold_da_mwh,    # what it sold on PZU, with a minus
    traded_id=-sold_id_mwh,    # what it sold extra (+) or bought back (−) on PI, with a minus
    pzu=pzu, pi=pi, a=a, b=b,  # the same a, b as for prosumers
)
# dev > 0 means PPC sold more than it produced (short)
```

**Metrics:**
- `wind_avoidable_cost_lei`
- `lei_per_mwh_produced`
- `abs_dev_mwh`: without prices, the most solid number
- `promised_but_not_delivered_mwh`: energy sold on turbines that were already stopped

**Done when:** the perfect-forecast test gives 0 lei, and the signs are checked on a small hand-computed case.

### Step 5 (D2): What to repair first (P4, ~30 min)

**Goal:** the work-order list sorted by money.

**File:** `sentinel/agent/triage.py` (extended; card W4 #167)

```python
def lei_per_day(turbine, status, m5, m6, pzu_next24):
    if status == "fault_stop":                       # stopped: loses everything it would have produced
        lost_mwh = m5.p50_next24h_mwh(turbine)
    else:                                            # running badly: loses the difference
        lost_mwh = m6.expected_minus_actual_mwh_last24h(turbine)
    return lost_mwh * pzu_next24.mean()

def triage_score(p_fault, lei_day, days_until_visit):
    return p_fault * lei_day * days_until_visit       # the v3 formula, without the flexibility term
```

**Rules:**
- Sensor faults and data gaps get a **remote check**, never a crew sent to the turbine.
- Curtailment does **not** create a work order. It is only logged as curtailed energy.
- Top-K work orders per day, where K = the number of available crews.

**Output:** every work order has `lei_per_day`, `triage_score` and the evidence chart (the power curve or the temperature residual).

**Done when:** the list in the O&M mode is sorted by `triage_score`, and T06 (stopped) appears at the top, with lei per day.

### Step 6 (D3): Which day to repair (P4, ~1 h, only if M7 is good)

**Goal:** for a high-risk turbine, the day when the repair stop costs the least.

**File:** `sentinel/agent/maintenance_window.py` (card W6 #169)

```python
def best_repair_day(turbine, days, repair_hours, p_fail_by_day, wind_limit_ms,
                    m5, price_for_day, unplanned_extra_days=3):
    best = None
    for d in days:                                     # e.g. the next 10 days
        if m5.max_wind_in_work_hours(d) > wind_limit_ms:
            continue                                   # wind too high for work at height
        planned = m5.p50_mwh(turbine, d, repair_hours) * price_for_day(d)
        # if you wait, you risk it failing first: a longer, unplanned stop
        risk = p_fail_by_day[d] * unplanned_extra_days * 24 * m5.mean_mw(turbine) * price_for_day(d)
        total = planned + risk
        if best is None or total < best[1]:
            best = (d, total)
    return best   # (the day, the estimated cost in lei)
```

**Prices:** real PZU for tomorrow; after that, the OPCOM monthly forward or the same day last week.

**The wind limit** for climbing the turbine is a configurable parameter.

**Comparison:**
- **reactive:** repair after it fails, with a long stop
- **fixed calendar:** repair on the planned date, regardless of wind
- **Sentinel:** the cheapest day before failure

**Metric:** `maint_window_savings_lei` on the faults injected in the test.

**Condition:** M7 must show **real early warning** on the test faults (`lead_time_h` > 0 on unseen events) and a reasonable CARE score on the public benchmark. Otherwise, D3 is not presented.

### Step 7 (D4): The park battery covers the errors (P3 or P4, ~1.5 h)

**Goal:** the 45.72 MW / 91.44 MWh (2 hours) battery reduces the difference between what PPC sold and what the park delivered.

**File:** `sentinel/opt/park_battery.py` (card W5 #168)

**The idea:**
- The park delivers **production − charge + discharge**.
- When delivery would be below what was sold, the battery discharges. When it would be above, it charges.
- The plan is redone **every hour**, with the updated forecast for the rest of the day (model predictive control).

**Three variants compared:**
1. **No battery.**
2. **Simple rule (baseline):** in each quarter hour cover as much of the difference as it can; when there is no difference, drift back towards 50% charge.
3. **LP (Sentinel):** plans over the rest of the day, so it keeps energy for the hours where the updated forecast sees a shortfall.

```python
# sentinel/opt/park_battery.py
import cvxpy as cp, numpy as np

def plan(prod_fc_mwh, sold_mwh, pzu, e0, a=0.5, b=0.5,
         p_mw=45.72, e_mwh=91.44, eta=0.94, c_deg=20.0, dt=0.25):
    T = len(prod_fc_mwh)
    C = cp.Variable(T, nonneg=True); D = cp.Variable(T, nonneg=True); e = cp.Variable(T + 1)
    short = cp.Variable(T, nonneg=True); long = cp.Variable(T, nonneg=True)
    delivered = prod_fc_mwh - C + D
    cons = [e[0] == e0,
            e[1:] == e[:-1] + eta * C - D / eta,
            e >= 0.1 * e_mwh, e <= 0.9 * e_mwh,
            C <= p_mw * dt, D <= p_mw * dt,
            short >= sold_mwh - delivered,
            long >= delivered - sold_mwh]
    cost = cp.sum(cp.multiply(a * pzu, short) + cp.multiply(b * pzu, long)) + c_deg * cp.sum(C + D)
    cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.HIGHS)
    return C.value, D.value     # only the first 4 quarters are applied, then the plan is redone
```

`c_deg` (lei/MWh shifted) is the battery's wear cost. Put it as an assumption in the README.

**Honestly:** the LP beats the simple rule only if the updated forecast is better than persistence. Report the difference, whatever it is.

**Metrics:**
- `battery_cover_share`: what part of the absolute deviation the battery covered
- the reduction of the avoidable cost
- cycles per day

### Step 8: Replay and metrics (P4 + P5, ~30 min)

```bash
python replay.py --mode wind \
  --policy manufacturer_curve,persistence,windpower_only,sentinel \
  --battery none,rule,lp \
  --out demo/sentinel.sqlite
```

(In this repo `replay.py` does not yet have `--mode` or `--battery`. Write to `$LAB_OUTPUT_DIR`, not over the demo DB,
unless the user approves.)

New `LAB_METRIC` lines:

```
LAB_METRIC wind_avoidable_cost_lei=...
LAB_METRIC lei_per_mwh_produced=...
LAB_METRIC wind_abs_dev_mwh=...
LAB_METRIC promised_but_not_delivered_mwh=...
LAB_METRIC alerts_per_turbine_day=...
LAB_METRIC battery_cover_share=...
LAB_METRIC maint_window_savings_lei=...
```

**Caution:** the current M6 results were obtained on an older version of the wind simulator. Rerun M6 on the current simulator before putting the numbers on a slide.

### Step 9: Database and UI (P5, ~1.5 h)

(card W7 #170; schema in `packages/db/src/schema/**` — a contract change: `lab announce` + `team/decisions.md`)

**Drizzle:**
- `wind_positions`: ts, park_id, policy, sold_da, traded_id, actual, dev_mwh, cost_lei
- `turbine_availability`: ts, turbine_id, availability, reason
- `work_orders`: new columns `lei_per_day`, `triage_score`, `best_repair_day`
- `park_battery`: ts, policy, charge, discharge, soc

**Production mode** (`apps/web/src/modes/production/**`):
- per-park chart: the P10–P90 band, "sold" and "actual"
- marks on the intervals where a stopped turbine was removed from the calculation
- KPI: avoidable lei today, manufacturer curve vs Sentinel

**O&M mode** (`apps/web/src/modes/om/**`):
- the work-order queue sorted by lei per day
- in the turbine drawer: the power curve (real points over the learned curve), the temperature residual trend, the M7 risk and **"the cheapest repair day"** with a small cost calendar

**Battery panel:**
- state of charge over the day
- what it covered, compared: no battery / rule / LP

**Overview:** the Wind O&M card also shows lei lost avoided.

### Step 10 (optional): The link with the prosumers

If the PPC parks and the PPC supplier are in the same balancing portfolio, their deviations can offset each other: wind that is too weak can be covered by prosumer solar that is too high, and vice versa. Run `avoidable_cost` on the sum of the two positions and compare with the sum of the separate costs. The difference is the value of a joint forecast for the whole group.

---

## 6. The data

| Data | Source | Type |
|---|---|---|
| Wind, temperature, pressure | Open-Meteo (ERA5), at the park coordinates | real |
| Park location, number of turbines, power | public data (see v3) | real |
| National wind production | ENTSO-E (already in the repo) | real, for validating M5 |
| PZU, PI, monthly forward | OPCOM | real |
| Real turbine faults | CARE-to-Compare (95 datasets, 36 turbines, 44 events) | real, **non-RO**, only for M6 / M7 |
| Per-turbine SCADA, faults with exact time | the simulator, in the PPC schema | **synthetic** |
| Power curves | generic for the 2.5 / 6.1 MW classes, **not** the manufacturer's curves | **synthetic** |
| The park battery | public parameters (45.72 MW / 91.44 MWh) | real (parameters), simulated behaviour |
| Imbalance price | PZU × (1 ± a), the same `a`, `b` as for prosumers | **proxy** |

The simulated data follows the PPC SCADA schema (`WP<park>.TID<turbine>.<sensor>`, kW, km/h, integer error codes), so real PPC data can be dropped in without code changes.

---

## 7. ~1-minute demo

1. **O&M mode, the Constanța park.**
   - T06 has good wind and zero power, while its neighbours run: **fault**, with `[lei]` lost per day, first in the queue.
   - A few minutes later, the whole park is limited to the same level: **curtailment**, no work order. The old rules would have fired on both.
2. **Production mode, same day.** The manufacturer curve sold energy on T06 too. Sentinel removed it from the calculation, so PPC sold what it could deliver. On screen: `[X] lei` vs `[Y] lei`.
3. **The park battery.** The wind drops earlier than forecast. The battery covers the gap at 18:00, compared with the simple rule.
4. **(If M7 worked out.)** The T12 gearbox has rising risk. Sentinel recommends the repair on Tuesday (low wind, low price), not Friday.
5. **The line:** *"The park knows what each turbine should produce. From that come the forecast, the alerts and the repairs, all in lei."*

---

## 8. Jury questions, with answers

**"Synthetic faults are too easy to catch."**
They are injected following physics, with random timing and severity. M6 and M7 are also tested on the public CARE-to-Compare benchmark, with real faults (non-RO, marked as such). The baseline is the WindMon rules, written on the real PPC schema.

**"How do you tell curtailment from a fault?"**
By context: if the whole park is limited to the same level at the same time, it is curtailment; if one turbine is stopped and its neighbours run, it is a fault. We show the confusion matrix, mistakes included.

**"Do you command the turbines?"**
No. We only predict, classify and recommend. The battery is simulated in the hackathon.

**"Are the power curves the manufacturer's?"**
No, they are generic per power class. M5 learns each turbine's real curve from data, so in a pilot it adapts by itself.

**"Where does the imbalance price come from?"**
PZU and PI are real (OPCOM). Only the final penalty is approximated, with the same calibration as for prosumers. The reduction percentage does not depend on the chosen level.

---

## 9. To ask the mentors (Laurențiu, O&M)

- [ ] Who is balance-responsible for the PPC parks: their own portfolio or together with the supplier? (decides step 10)
- [ ] What PPC currently produces as a forecast for the parks and how often it is wrong
- [ ] When the Dobrogea battery goes into operation and who commands it
- [ ] How many technician crews are on the park and how long a typical repair takes
- [ ] The wind limit above which nobody climbs the turbine
- [ ] Whether curtailed energy is compensated and how it is reported

---

## 10. Final checklist

- [ ] M5 trained (or the wind-bin fallback), beats the manufacturer curve at park level
- [ ] Tomorrow's availability from M6 / M7; the stopped-turbine test passes
- [ ] The park forecast calibrated (78–82% P10–P90 coverage)
- [ ] `avoidable_cost` on wind, with the sign checked
- [ ] Work orders have lei per day and are sorted by `triage_score`
- [ ] M6 rerun on the current simulator
- [ ] Battery: none / rule / LP compared
- [ ] D3 only if M7 has real early warning
- [ ] The Production and O&M modes show the new numbers
- [ ] Every number on a slide has a run ID and the label "synthetic RO / real RO / non-RO / proxy"

---

## Sources

- [RWEA: Romania, the highest balancing costs for wind in the EU (Bursa, 2017)](https://www.bursa.ro/asociatia-romana-pentru-energie-eoliana-suntem-tara-din-ue-cu-cele-mai-mari-costuri-de-echilibrare-pentru-productia-energiei-eoliene-46939232)
- [RWEA: balancing costs are very high (Agerpres, September 2024)](https://agerpres.ro/economic/2024/09/12/gavrila-rwea-regenerabilele-au-fost-aratate-cu-degetul-ca-principale-provocatoare-de-dezechilibre-in--1353478)
- [OPCOM: PI trades](https://www.opcom.ro/rapoarte-pi-rezultate-pi/ro)
- The PPC parks, the Dobrogea battery, WindMon and CARE-to-Compare: the sources are already cited in the v3 document (economica.net, Balkan Green Energy News, Romania Insider, the WindMon repo).
