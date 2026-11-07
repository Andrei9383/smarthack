# Feature G (Grid): cables stay under the limit without replacing them

**Prosumer Sentinel · Smarthack XI · Rețele Electrice România (the distributor in the PPC group)**
App mode **Feeders**. This is the most advanced part: the numbers already exist.

> English translation of the user's spec `feature-retea-retele-electrice.md` (card #156). Step numbers ("Spec G step N")
> are identical to the original. Numbers, formulas, sources and assumptions are unchanged; only paths are mapped to this repo:
>
> | Original path | This repo | Card |
> |---|---|---|
> | `ml/eval/hosting_capacity.py` | `sentinel/eval/hosting_capacity.py` | G1 #159 |
> | `ml/gridtwin/voltage.py` | `sentinel/gridtwin/voltage.py` (M8 lives in `sentinel/gridtwin/twin.py`) | G2 #160 |
> | `ml/eval/dso_value.py` | `sentinel/eval/dso_value.py` | G3 #161 |
> | LP on the feeder | `sentinel/opt/lp.py` (baseline: `sentinel/opt/policies.py`) | G4 #162 |
> | `ml/replay.py` | `replay.py` → `sentinel/agent/replay.py` | — |
> | `db/sentinel.sqlite` | `demo/sentinel.sqlite` (demo stays on replay#506 unless the user approves) | — |
> | Feeders mode UI | `apps/web/src/modes/feeders/**`, `packages/api/src/routers/feeders.ts`, `packages/db/src/schema/**` | G5 #163 |
> | Supplier objective (step 6) | `sentinel/eval/imbalance.py` (X1 #158) + `sentinel/eval/portfolio.py` (I5 #175) | G4 #162 |
>
> Policy names: the spec's `self_consumption` baseline is `baseline` in the current `replay.py --policy` (implemented by
> `self_consumption_plan` in `sentinel/opt/policies.py`). Model reads go through `sentinel/judge/adapter.py` (C1 #157).

---

## 0. In short

At midday, every panel on the same street produces at maximum at the same time. The cable and the transformer substation cannot carry that much. Voltage rises, inverters trip, and the distributor has to invest in new cables and transformers.

**Sentinel uses the homes' batteries to absorb the midday peak**, instead of letting them fill up by 10:00. It discharges them in the evening, into the home. It also checks whether the network map (GIS) puts each home on the right cable, so the plan does not rest on wrong data.

Problem solved: **more prosumers connected on the same cables, and deferred investment.**

---

## 1. The problem, in plain words

### What happens on a street with prosumers

1. At midday, all homes' panels produce at maximum **at the same time**.
2. Export exceeds what the cable or the transformer substation can carry.
3. **Voltage rises.** Inverters disconnect automatically above **253 V**, so prosumers lose production and complain.
4. Batteries could help, but with the "charge from surplus" rule they are **already full at 10–11**, so at the 12–14 peak they absorb nothing.

### How big the problem is in Romania

- **At another distributor (Distribuție Energie Oltenia, February 2024):**
  - at the peak, voltage rises on some networks by up to **+10%**
  - inverters on the same cable disconnect
  - a transformer substation with 20–28 prosumers would need **about three times the capacity** for about 3 hours a day
  - preparing the network for 100,000–150,000 prosumers was estimated at **~400 million euro**
  - management said battery storage can reduce the need for network reinforcement
- **At Easter 2025**, with very low consumption, the prosumers' association asked its members to switch off their inverters if they left home, so as not to stress the system.
- **Rețele Electrice România** invested about **1.4 billion lei in 2025**: roughly 1,000 new transformer substations and 440 modernised, plus 2 million smart meters.

### The real cause

The batteries exist, but **nobody coordinates them at street level**. Each home optimises only its own bill.

---

## 2. How it works

```
M1 (how much each home produces) + M2 (how much it consumes) ──► export of each street (feeder), per 15 min
M4: broken homes removed from the calculation ───────────────────────┤
M8 GridTwin: is the home on the right cable / phase? (trust score τ) ──► limit used by the LP
                                                             ▼
                      LP per feeder: when each battery charges / discharges
                      ├─ keeps the street's export under the limit (hard constraint)
                      ├─ never goes below the reserve chosen by the customer (hard constraint)
                      └─ discharges into the home in the evening
                                                             ▼
                 P10 commitment: how much flexibility Sentinel can guarantee, measured afterwards
```

**The steps, in words:**
1. **Predict** how much each street will export tomorrow (M1 + M2), without the broken homes (M4).
2. **Check the map:** if the GIS has homes on the wrong cable, the limit is tightened where trust is low (M8).
3. **Plan the batteries:** keep them partly empty in the morning, absorb the midday peak, then discharge in the evening.
4. **Promise only what can be delivered:** flexibility is promised at P10, i.e. the pessimistic case.
5. **Measure:** minutes over the limit, energy over the limit, how much of what was promised was delivered.

---

## 3. Mini-glossary

| Term | Meaning |
|---|---|
| **Feeder** | the cable leaving a transformer substation that supplies a street or neighbourhood (low voltage) |
| **Transformer substation** | the "box" that steps the voltage down for homes |
| **Feeder limit** | how much power the cable can carry without problems |
| **Overvoltage** | voltage rises above the limit (e.g. above 253 V) and inverters stop |
| **Hosting capacity** | how many more prosumers can be connected on a feeder without problems |
| **Flexibility** | how much power the batteries can shift when needed |
| **Nominal / available / promised (P10) / delivered** | theoretical / possible now / guaranteed / measured afterwards |
| **GridTwin (M8)** | checks whether the network map puts each home on the right cable and phase |
| **Trust score τ** | how certain the map is for an area: 1 = certain, 0 = uncertain |

---

## 4. What you already have (results from the replay on the test week)

| Metric | Without Sentinel | With Sentinel |
|---|---|---|
| Minutes over the limit (week) | 6,645 | **1,905 (−71%)** |
| Energy over the limit | 1,397 kWh | **94.5 kWh (−93%)** |
| Export peak | | **−19%** |
| Flexibility delivered out of promised | | **92%** (target ≥ 90%) |
| Times the customer reserve was breached | 1,165 (simple rule) | **0** |
| Customer savings | | **23.7 lei / home / month** |
| Minutes over the limit (Friday) | 1,065 | **225 (−79%)** |

**Pieces ready:** M1, M2, M4, the feeder LP with the P10 commitment, M8 GridTwin, the replay, the Feeders mode.

**What is missing:** the numbers that show the distributor **what it gains**, i.e. extra connections, voltage kept under control and deferred investment. That is what the steps below do.

---

## 5. What is still needed, step by step

### Step 1: Hosting capacity (P4, ~1.5 h), the most important new number

**Goal:** "With Sentinel, feeder F7 can take **N** more prosumers without new cable."

**File:** `sentinel/eval/hosting_capacity.py` (card G1 #159)

**How:**
1. On each feeder, add typical synthetic prosumers (the same kWp and battery distribution as in the simulator).
2. For each number of added homes, run the replay on the test week.
3. Find how many homes can be added **until** minutes over the limit exceed a tolerance (e.g. 0 or 30 minutes per week).
4. Do this without Sentinel (the simple rule) and with Sentinel. **The difference is the gain.**

```python
# sentinel/eval/hosting_capacity.py
def violates(feeder, n_added, policy, tol_min=30):
    homes = feeder.homes + synthetic_homes(feeder, n=n_added)   # same distribution as the simulator
    res = replay_feeder(homes, policy)                            # same test week
    return res.minutes_over_limit > tol_min

def hosting_capacity(feeder, policy, hi=300):
    lo = 0                                                        # binary search
    if violates(feeder, 0, policy):
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if violates(feeder, mid, policy):
            hi = mid - 1
        else:
            lo = mid
    return lo
```

**Assumption to state:** new homes have a battery in the same proportion as existing ones. Also run the variant "new homes without battery".

**Metric:** `hosting_capacity_gain`, i.e. extra prosumers per feeder and in total.

**Done when:** every feeder has two numbers (without / with Sentinel), and with Sentinel it is never lower.

### Step 2: Voltage and inverters that trip (P2, ~1.5 h)

**Goal:** show the physical effect, not just "kW over the limit".

**File:** `sentinel/gridtwin/voltage.py` (card G2 #160; reuses pandapower from M8)

**How:**
1. On the IEEE European LV test feeder (55 customers), place the simulated homes on the 55 load points.
2. Run a 3-phase power flow every 15 minutes, with the injections without Sentinel and with Sentinel.
3. Count:
   - **steps with voltage above 253 V** (1.10 p.u.) at any home
   - **energy lost** by inverters that would have disconnected (kWh): the home's production in the overvoltage steps

**Metrics:** `voltage_violations`, `inverter_trip_kwh`. The loss can be expressed in lei for prosumers: kWh × energy price.

**Caution:** the IEEE feeder is a European test topology, not a Romanian street. Mark this on the slide.

**Done when:** on the same week, Sentinel has fewer overvoltage steps than the simple rule.

### Step 3: Value in lei for the distributor (P4, ~45 min)

**Goal:** a number in lei the distributor understands immediately.

**File:** `sentinel/eval/dso_value.py` (card G3 #161)

**Formula, for the deferred investment:**

```
value = reinforcement_cost × (1 − 1 / (1 + r)^years_deferred)
```

- `reinforcement_cost` = **[SUBSTATION / CABLE COST]**: placeholder; ask the GIS mentor for a figure
- `r` = discount rate (e.g. 7%), as an assumption
- `years_deferred` = how many years later the feeder reaches its limit, from the annual growth of prosumers and the hosting capacity from step 1

**As an order of magnitude** (clearly marked as another source): DEO's estimate of ~400 million euro for 100,000–150,000 prosumers means a few thousand euro of reinforcement per new prosumer.

**On the slide:** the headline number is **"+N prosumers on the same cables"**. The lei come as an estimate, with the assumptions written underneath.

### Step 4: GridTwin in the loop (P2, ~30 min, almost done)

**Goal:** show that checking the map actually changes the decision. (card G3 #161; code in `sentinel/gridtwin/`)

1. Report **what share of the wrong records** (the 10% deliberately corrupted) M8 recovered.
2. Limit used by the LP: `G = τ × G_max + (1 − τ) × G_conservative`. Where the map is uncertain, the limit tightens, so the plan is more cautious.
3. Show a concrete case: the home placed wrongly on F7 in the GIS, M8 moves it to F3, and the plan for F7 changes.

**Metrics:** `gis_recovery_rate` and the difference in minutes over the limit with / without τ.

### Step 5: Who pays for the batteries (~15 min, one slide)

The distributor cannot command customers' batteries without a contract. The realistic model:
- The distributor **buys local flexibility**: it pays the homes (or an aggregator) to absorb at midday on congested feeders.
- The European framework requires that distributors can use flexibility instead of investment (EU Directive 2019/944, art. 32). How it is transposed in Romania must be checked.
- Commands go through the inverter manufacturer's app, only where the customer has given access. In the hackathon, everything is simulated.

### Step 6: Reconciling with the supplier feature (P4, ~30 min)

The same batteries can be wanted both by the supplier (the prosumer / Imbalance feature) and by the distributor. (card G4 #162, `sentinel/opt/lp.py`)

- **The rule:** the feeder limit and the customer reserve are **hard constraints**, and the supplier's cost is the **objective**. The same LP does both.
- **At midday** they usually want the same thing: the battery absorbs the surplus.
- **In the evening** conflicts can appear, and then the cable limit wins.

**Metric:** run the replay with both objectives and show that minutes over the limit do not increase when you add the supplier objective.

### Step 7: UI (P5, ~1 h)

In the **Feeders** mode, in addition to what exists (card G5 #163, `apps/web/src/modes/feeders/**`):
1. **"Hosting capacity" panel:** per feeder, "+N prosumers with Sentinel", with a without / with bar.
2. **"Voltage" panel:** steps above 253 V and kWh lost by inverters, without / with Sentinel (IEEE feeder).
3. **GridTwin layer on the map:** low-trust areas hatched, plus the list of moved homes.
4. **Value:** "deferred investment ≈ `[X]` lei", with the assumptions one click away.
5. **Overview:** the Feeders card can also show "+N possible prosumers".

### Step 8: Replay and metrics (~20 min)

```bash
python replay.py --mode feeders --policy self_consumption,threshold,sentinel \
  --hosting-capacity --voltage --out demo/sentinel.sqlite
```

(In this repo `replay.py` does not yet have `--mode`, `--hosting-capacity` or `--voltage`; today's policy names are
`baseline|threshold|sentinel`, where `baseline` = self-consumption. Write to `$LAB_OUTPUT_DIR`, not over the demo DB,
unless the user approves.)

New `LAB_METRIC` lines:

```
LAB_METRIC hosting_capacity_gain=...
LAB_METRIC voltage_violations=...
LAB_METRIC inverter_trip_kwh=...
LAB_METRIC gis_recovery_rate=...
LAB_METRIC deferred_capex_lei=...      # with the cost placeholder marked
```

`threshold` is the "smarter" per-home rule (from v3), which must be compared honestly. The jury will ask.

---

## 6. The data

| Data | Source | Type |
|---|---|---|
| Weather | Open-Meteo, at the prosumers' localities | real |
| Prices | OPCOM | real |
| Prosumers, batteries, feeders | the simulator, in real Romanian localities | **synthetic** |
| Feeder limits | the simulator | **synthetic** |
| Topology for voltage and GridTwin | the IEEE European LV test feeder (pandapower) | real, **non-RO**, topology only |
| GIS errors | 10% records deliberately corrupted | **synthetic** (labels) |
| Context (overvoltage, reinforcement costs) | public statements by DEO, APCE, Rețele Electrice | real, other operators marked as such |

---

## 7. ~40-second demo

1. **Feeders mode, F7 Oșorhei, Friday.** Without Sentinel, export exceeds the 62 kW limit at midday for 180 minutes. With Sentinel, batteries are kept partly empty in the morning, absorb 11–14 and discharge in the evening.
2. **Capacity panel:** "on the same cables, F7 can take `[N]` more prosumers".
3. **Voltage:** steps above 253 V drop from `[A]` to `[B]`.
4. **GridTwin:** the home wrongly placed on F7 is found and moved, and the plan changes.
5. **The line:** *"The batteries are already in the homes. Sentinel makes them work for the street, not just for the bill."*

---

## 8. Jury questions, with answers

**"The supplier and the distributor are separate companies. Who uses this?"**
This feature is for the distributor (Rețele Electrice). The prosumer one is for the supplier. The same LP respects both, and the cable limit always has priority.

**"Wouldn't a smarter rule on each home do the same?"**
We compare it (`threshold`) and report the difference, whatever it is. A per-home rule does not know what the neighbours do, so it cannot share absorption fairly along the street.

**"The IEEE feeder is not Romanian."**
Correct. We use it only for voltage and for checking the map, as a test topology. A pilot needs the real topology from Rețele Electrice's GIS.

**"How do you control the batteries?"**
Only with a contract and the customer's consent, through the inverter manufacturer's app. Without access, Sentinel only monitors. In the hackathon, control is simulated.

**"Where does the reinforcement cost come from?"**
It is a placeholder until the mentor gives us a figure. The order of magnitude comes from another distributor's (DEO) public statements, marked as such.

---

## 9. To ask the GIS mentor (Ionuț-Valentin Isopescu)

- [ ] How many feeders or transformer substations in the Rețele Electrice area are already at the limit because of prosumers
- [ ] Are connections refused or postponed for lack of capacity?
- [ ] Typical cost of a reinforcement (new substation, new cable per km)
- [ ] How often the home → feeder / phase link is wrong in the GIS
- [ ] Whether Rețele Electrice already buys or tests local flexibility
- [ ] What voltage data they have at substation or smart-meter level

---

## 10. Final checklist

- [ ] Hosting capacity without / with Sentinel on every feeder
- [ ] Voltage: steps above 253 V and kWh lost, on the IEEE feeder
- [ ] Value in lei, with the assumptions written and the cost placeholder marked
- [ ] GridTwin recovery rate + the effect of τ on the plan
- [ ] Comparison with the `threshold` rule, reported honestly
- [ ] Supplier objective added without increasing minutes over the limit
- [ ] The new panels in the Feeders mode
- [ ] Every number has a run ID and the label "synthetic RO / non-RO / proxy"

---

## Sources

- [Distribuție Energie Oltenia: overvoltage, inverters, overloaded substations, ~400 mil. euro (Economica, February 2024)](https://www.economica.net/ce-probleme-a-adus-valul-de-prosumatori-pentru-reteaua-electrica-si-cum-se-pot-rezolva-ce-spune-seful-retelei-cu-cei-mai-multi-prosumatori-din-romania_728762.html/amp)
- [APCE: inverters switched off at Easter, disconnection above 253 V (Economica, April 2025)](https://www.economica.net/apce-indeamna-prosumatorii-sa-opreasca-invertoarele-daca-pleaca-de-acasa-de-paste_829117.html)
- [Rețele Electrice România: ~1.4 bn lei invested in 2025 (Financial Intelligence, April 2026)](https://financialintelligence.ro/retele-electrice-romania-a-investit-circa-14-miliarde-de-lei-in-2025-in-modernizarea-si-digitalizarea-retelelor-electrice-de-distributie/)
- [Rețele Electrice România: the network (134,000 km, 25,000 substations) and smart meters](https://reteleelectrice.ro/wp-content/uploads/comunicat-rer-program-smart-meter-2025.pdf)
