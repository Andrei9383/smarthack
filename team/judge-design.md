# Sentinel Judge: the trust filter

Status: draft for team review, 2026-11-07. Owner: yolooo332 (user decision: this is the project's main feature).
Scope of this document: the **filter**, meaning which information is valid. **What to do with that information (the
actions) is open (§10) and will be specified by the user.**

## 1. Intent

The agent has eight models and acts on their outputs as if all of them were equally right, everywhere, every day.
The failures logged in `team/insights.md` show they are not:
- SiteHealth calls a string outage a "sensor fault" on battery homes.
- WindNBM missed a park-wide power cap.
- LoadNet's bands covered 44 % instead of 80 %.
- PVNet's bands cover 10 % on real sites.

The **Judge** is a layer above the models. Before anything acts on a prediction, the Judge decides whether that
prediction is valid. It answers with one of three verdicts per prediction: **use**, **use with a wider band** or
**don't use** (with the best-trusted substitute named). It bases this on two things:
- the **track record** of that model in the current situation, built by comparing every past prediction with what
  actually happened;
- the **signals of this specific prediction**: how unusual its input is, how much the ensemble disagrees, and
  whether the other models agree with it.

The models, the Judge and outcomes form a loop: models predict, the Judge filters, actions use what passed, reality
is observed, predictions are graded, the track record updates, and the next day's filtering uses it.

**Pitch:** *Prosumer Sentinel is an agent that knows which of its own models to believe. It acts on a prediction
because that model has earned trust in this situation, and it shows what it trusted and what it overruled.*

## 2. Goals, non-goals, anti-goals

Goals
- Every witness prediction gets a verdict, a calibrated probability that it is valid, and a plain-words reason.
- Every prediction is graded against an observed outcome. The grades keep a trust ledger per model and situation.
- **No record → trust the model** (user decision). A model without a live history is used at its validated
  strength. It is never distrusted by default.
- Detects a degrading model within days, does not raise false alarms on healthy ones, and restores trust when a
  model recovers.

Non-goals
- The Judge does not choose actions (§10).
- It does not retrain models. It flags a degraded model; retraining stays a human decision.
- No LLM decides. An LLM may phrase the reasons.

Anti-goals (failure even if it runs)
- **Peeking at the answer:** the live loop must never read `fault_labels` or any simulator truth (§5.1).
- **Trust you can game:** a scoring rule that rewards wide bands, always saying "healthy", or hiding misses (§5.3,
  §5.4).
- **A filter that only removes:** trust must be able to come back, and a model must not be rejected merely for
  being new.

## 3. The witnesses

| Witness | Question | Output the Judge reads | State on main |
|---|---|---|---|
| M1 PVNet | PV production per home | P10/P50/P90 kW, OOD score, ensemble spread | promoted |
| M2 LoadNet | consumption per home | same | promoted |
| M4 SiteHealth | is this home's system healthy | 8 fault-class probabilities + expected PV | promoted |
| M6 WindNBM | is this turbine normal | anomaly class + score + expected kW | promoted |
| M3 DemandNet | zone demand peaks | zone P10/P50/P90 MW, stress, peak hour | code on main |
| M8 GridTwin | are the GIS records right | feeder trust tau, phase/feeder flags | code on main |
| M7 WindFailure | component failure ≤ 14 d | risk per component, alarm | code on main (train#379, synthetic RO parks); weights to promote |
| M5 WindPower | park production | park P10/P50/P90 MW | experiment #42, not merged; skipped until merged |

Each witness has a **substitute**: the baseline it was compared against (plan R4).

| Witness | Substitute |
|---|---|
| M1 | LightGBM / pvlib |
| M2 | LightGBM / persistence |
| M4 | threshold + CUSUM |
| M6 | WindMon rules |
| M3 | persistence |
| M7 | residual trend |
| M8 | GIS as recorded |

Substitutes are graded exactly like witnesses, so the Judge can tell when the baseline is the better source.

## 4. The loop

```
 witnesses + substitutes predict (day d)
            │
            ▼
 ┌─────────────────────── JUDGE (filter) ───────────────────────┐
 │ per prediction:                                              │
 │  ledger lookup  → trust, evidence, band scale  (§6)          │
 │  own signals    → OOD, spread, input quality                 │
 │  cross-checks   → agreement with other witnesses             │
 │  validity model → P(valid)                     (§7)          │
 │  verdict rules  → use / widen / don't use (+ substitute)     │
 └───────────────────────────┬──────────────────────────────────┘
                             ▼
               validated information → ACTIONS (§10, open)
                             │
                             ▼
 outcomes: telemetry (d+1), crew & audit checks (d+1..3), events (≤ d+14)
                             │
                             ▼
 GRADER (§5): independent quality gate → one score per asset-day → ledger update
                             │
                             └──────────────► used for day d+1
```

## 5. The Grader: comparing with what actually happened

### 5.1 Only what PPC can observe

Outcomes come from four observable sources and nothing else:
1. **Telemetry:** measured PV, load, turbine power, voltage, status codes. Every asset reports it, whatever action
   was taken.
2. **Crew reports:** what a technician finds on a visit. In the replay a simulated `CrewReport` reveals the asset's
   true state at visit time, for that asset only, with a 5 % misreport rate (my call). This is the only route by
   which simulator truth reaches the loop.
3. **Audit checks** (§5.6): remote checks the Judge requests itself, so grading does not depend on actions.
4. **Delayed events:** component failures (M7), field verification of GIS records (M8), realised zone load (M3).

### 5.2 Unit of evidence: one asset-day

A home gives 96 readings a day. Counting each reading as evidence would let one day overwhelm everything else.
The unit is therefore **one score per asset, per witness, per day**. The 96 steps are summarised into that day's
score.

### 5.3 Scores: rules that cannot be gamed

All scores lie in 0..1, where higher is better.

**Forecasts (M1, M2, M3, M5): two scores per asset-day.**
- **Calibration** = 1 − |hit − 0.80| / 0.80, where *hit* is the share of the day's production steps where the
  actual value fell inside P10–P90 (for M2, all steps).
  - The target is the team's 80 % (decision 13:52), so both too-narrow and too-wide bands lose.
- **Sharpness and accuracy** = clip(1 − IS_model / IS_substitute, 0, 1), using the interval score
  (Winkler score).
  - The interval score charges for band width plus a penalty for every miss.
  - A huge band can still score well on calibration, but it scores badly here.
  - Comparing against the substitute means "valid" = "better than what we would otherwise use".

**Detectors (M4, M6): split by what the model claimed,** so "always healthy" cannot score well.
- **Alarm precision** (graded when the model calls a fault or anomaly): 1 if a crew or audit check confirms the class
  (M6 curtailment can also be confirmed by the operator's curtailment log), 0.5 if a fault exists but of another
  class, 0 if nothing is found.
- **Miss score** (graded when the model says healthy or normal):
  - 0 if a fault with an onset before that day is found later (by any visit, audit check or the residual check
    below);
  - 1 if an audit check finds the asset healthy, or the residual check passes;
  - ungraded otherwise.
- **Residual check** for unvisited homes: over the next 3 days, measured PV stays inside the **pvlib** band (not
  PVNet's), so a detector's grade never depends on another witness.

**Slow witnesses**
- **M7:** hit if (alarm and failure within 14 days) or (no alarm and no failure). It is graded once the 14 days
  have passed. The Brier score is kept as a detail.
- **M8:** a flag is correct if a field visit confirms it. Unflagged sites that are visited grade the miss side.

### 5.4 Independent data-quality gate

Corrupted telemetry must not grade anyone, but the gate must never use the witness being graded. Otherwise a broken
model's misses would be thrown out as "bad data". The gate uses only physics and the asset's own history:
- PV ≥ 0 and ≤ 1.2 × kWp; PV near 0 at night (pvlib clear-sky = 0)
- load ≤ the connection's fuse limit
- frozen sensor: the same value for ≥ 8 steps while pvlib clear-sky changes
- unit jumps: daily energy outside 0.2–5 × the asset's own 30-day median
- turbine power ≤ 1.1 × rated

A day that fails the gate is **ungraded**. It never counts as a miss, and never as a pass.

### 5.5 Delayed outcomes stay in time order

Each outcome has an `observed_at`. On day d, the ledger uses only outcomes with `observed_at ≤ d`:
- telemetry scores arrive at d+1;
- crew and audit reports arrive when the visit happens;
- M7 scores arrive at d+14.

The ledger never uses an outcome before it could have been known. This applies to training as well (§7.6).

### 5.6 Audit checks: grading that does not depend on actions

Fault calls are confirmed mainly when someone visits. If only acted-on cases were graded, the ledger would learn
about a biased sample. The Judge therefore has its own small **audit budget**: 2 remote checks per crew type per
day (my call), outside any action budget.
- Assets are sampled at random, weighted towards uncertain predictions (P(valid) near the thresholds).
- Every audited outcome carries an **inverse-probability weight** (1 / probability it was sampled), so the ledger's
  estimate stays unbiased.
- Visits caused by actions are also graded, but are flagged and down-weighted in the ledger (my call: weight 0.5),
  because they are not a random sample.

### 5.7 Shared weather counts once

All homes on a feeder share the same sky, so one surprise storm hits PVNet's grades on every site at once. Within
a context bucket, all asset-day scores of one day are **averaged into one bucket-day**, which counts as one unit of
evidence. Independent days are what teach trust, not the number of sites.

## 6. The trust ledger

### 6.1 Context buckets

Trust is kept per **(witness, model version, score, context bucket)**. The bucket conditions:
1. asset kind
2. seen in training or unseen
3. battery yes / no
4. season (warm Apr–Sep / cold Oct–Mar)
5. data quality (clean / gappy: > 10 % missing steps)
6. input unusualness (OOD below / above the model's threshold)

**Backoff:** if a bucket has fewer than 5 effective bucket-days of evidence, read the parent bucket. Conditions are
dropped from last to first; the prior is the root.

### 6.2 Prior: no record means trust the model (user decision)

Every score in every bucket has a **prior**: what that model version achieved on its 2024 validation data. It uses
that bucket's validation figure if validation covered the bucket, otherwise the model's overall figure. Priors ship
inside the model's calibration JSON (extension of `team/contracts/model.md`). Substitutes get priors the same way.

### 6.3 Update rule

```
trust = (n0 · prior + Σ wᵢ · sᵢ) / (n0 + Σ wᵢ)
sᵢ = bucket-day score
wᵢ = 0.5^(age_days / 14) × audit/IPW weight
```

- **`n0 = 7` bucket-days** (my call). n0 must stay well below the most live evidence the decay allows, which is
  about 20 bucket-days at a 14-day half-life. Otherwise trust could never move far from the prior.
- With no evidence, trust = prior (the user's rule).
- The 14-day half-life means recent behaviour dominates and a recovered model regains trust within about two weeks.
- **New model version:** it starts from its own prior. Old outcomes stay with the old version.
- **Uncertainty:** each trust value carries `n_eff` and an 80 % interval from the equivalent Beta distribution, so
  the UI can say "little evidence yet".

### 6.4 Band rescaling

Per bucket, the ledger keeps a **band scale**, so that the live P10–P90 hit rate moves to 0.80. It is computed from
the same causal outcomes as split-conformal quantiles of the normalised errors, bounded to 0.5–3×. This generalises
the global online conformal in `sentinel/opt/commit.py` to per-context. The commit code keeps the existing
data-quality gate (insights #29) and gains the independent gate of §5.4.

### 6.5 Health alerts

If a witness's trust in a bucket stays below **prior − 0.2 for 3 consecutive days**, with at least 5 effective
bucket-days of evidence, the Judge raises a **witness health alert**, for example "PVNet degraded on unseen sites
since 4 Jun", linking to the graded outcomes. The alert clears when trust returns above prior − 0.1. The gap
between the two levels prevents flapping.

## 7. The Judge model (the validity filter)

### 7.1 Why a model on top of the ledger

The ledger knows how reliable a witness is **on average in this situation**. It cannot tell that *this* prediction
is the bad one. Example: PVNet is usually good on battery homes, but today its input is unusual (OOD 4) and it
disagrees with pvlib by 60 %. The Judge model combines the track record with the signals of the individual
prediction.

### 7.2 Question and target

*"Will this prediction pass its grade?"* One row per (witness, asset, day).
- **Forecasts:** valid if the day's hit rate is between 0.60 and 0.95, and sharpness/accuracy > 0, which means
  better than the substitute.
- **Detectors:** valid if the call is confirmed (alarm precision = 1), or the healthy call is not later contradicted
  (miss score = 1).
- Ungraded rows are not used for training.

### 7.3 Inputs

- **Own signals:** OOD score; ensemble spread; band width relative to the bucket's usual width; fallback flag;
  telemetry age; quality-gate flags on the *input* window (gaps, frozen, unit jumps).
- **Ledger:** trust per score, `n_eff`, 3-day trust trend, band scale, health-alert flag; the same features for the
  substitute.
- **Cross-witness agreement:**
  - PVNet P50 vs pvlib (and LightGBM) energy
  - M4's fault vs the PVNet residual
  - the share of the park's turbines in the same M6 class (park consensus, #99)
  - an M8 phase/feeder flag on this site
  - DemandNet's zone vs the sum of LoadNet over the zone
  - M7 risk vs M6 class on the same turbine
- **Context:** asset kind, battery, seen/unseen, season, hour profile of the day (clear / mixed / overcast from
  ERA5 forecast cloud).

### 7.4 Model

- **LightGBM binary classifiers, one per family:**
  - forecasts: M1, M2, M3 (+ M5 when merged)
  - detectors: M4, M6
- Why LightGBM: CPU only, missing inputs handled natively, and feature contributions give the reasons.
- **5 seeds**, averaged. Seed disagreement is reported as the Judge's own uncertainty.
- Probabilities are **calibrated with isotonic regression** on a held-out 30 % of 2024 assets, so P(valid) = 0.8
  means valid about 80 % of the time.
- **M7 and M8** produce too few graded outcomes to train on, so they use **ledger trust only** as P(valid). This is
  said plainly in their verdicts.

### 7.5 Verdict rules

In order; the first match wins:
1. **Bad input:** the input window fails the quality gate, or the telemetry is stale → **don't use**
   ("input unreliable").
2. **Safety pass-through:** an icing or overheating call is **never suppressed**. With low P(valid) it passes as
   **use, low trust**, so a safety signal is never filtered out silently.
3. **No record:** `n_eff = 0` for this witness and bucket, and none of rule 1's red flags → **use**, at the prior
   (the user's rule; the model's opinion is shown but does not override it).
4. **P(valid) ≥ τ_use** → **use**.
5. **Forecasts with τ_widen ≤ P(valid) < τ_use** → **use with wider band** (the bucket's band scale, times an extra
   factor learned on validation).
6. Otherwise → **don't use**. If the substitute's P(valid) in this context is higher, it is named as the
   **substitute**. Otherwise the verdict is "no reliable source", which actions must treat as unknown.

Choosing the thresholds (on 2024 held-out assets; my call):
- τ_use is the lowest value at which ≥ 90 % of "use" predictions are valid.
- τ_widen is the lowest value at which widened bands reach 0.80 calibration.

### 7.6 Training data

- **Year:** 2024, the witnesses' validation year, so their outputs are out-of-sample. On 2022–23 they would look
  too good. Test year: 2025.
- **Weeks:** 12 sampled weeks, one per month, over the demo feeders and all wind parks (keeps lab time bounded).
- **Causal ledger:** ledger features are computed by running the ledger forward in time with outcome delays
  (§5.5), with no look-ahead.
- **Drift augmentation:** in half of the weeks, one random witness is degraded from a random day. The forms of
  degradation are:
  - PV/load bias ±20–40 %
  - bands narrowed ×0.5
  - fault-class confusion (e.g. string outage → sensor fault on battery homes)
  - a frozen output

  Without this, trust barely varies in 2024 and the model never learns what low trust means.
- **Cold starts:** in a quarter of the weeks, the ledger is wiped at a random day, so the model sees no-record rows
  (rule 3 still applies at run time).
- **Entry point:** `train.py --model judge` (`LAB_METRIC` lines, LightGBM text + calibration JSON), submitted to the
  lab only.

### 7.7 Output contract (what actions will read)

Per prediction:
- `verdict` (use | widen | dont_use)
- `p_valid`
- `trust` and `n_eff`
- `band_scale`
- `substitute` (model id or null)
- `reasons`: up to 3 items in plain words, from feature contributions, e.g. "trust 0.41 for battery homes",
  "disagrees with pvlib by 58 %"
- `rule` (which rule in §7.5 fired)
- `judge_version`

Actions never read a raw witness prediction without its verdict.

## 8. Data contract change

New tables in `packages/db/src/schema/sentinel.ts`. The Drizzle schema is the contract (decision 14:15), so this
needs `lab announce` + `team/decisions.md`.
- `outcomes`: `run_id, model, model_version, asset_id, day, score_name, score, weight, source
  (telemetry|crew|audit|event|residual), observed_at, gated (0/1)`
- `trust`: `run_id, day, model, model_version, bucket, score_name, prior, n_eff, trust, lo80, hi80, band_scale,
  health_alert`
- `judge_verdicts`: `run_id, day, asset_id, model, verdict, p_valid, trust, n_eff, band_scale, substitute,
  reasons_json, rule, judge_version`

## 9. Proving the filter works

Evaluated on 2025 (test year), on its own, before any action uses it:

| Check | Metric | Compared with |
|---|---|---|
| Is P(valid) honest? | reliability curve, Brier score | "trust everything" (today), today's static OOD/width rules in `qualify.py`, ledger only |
| Does "use" mean valid? | precision of "use" (target ≥ 0.90), share of predictions kept | same |
| Does it catch bad predictions? | recall of "don't use" on predictions that failed their grade | same |
| Drift drill | days to the health alert after a witness degrades on day 3 | ledger only, static rules (never alert) |
| False alarms | health alerts per witness-month with no drift injected | target ≈ 0 |
| Recovery | days to clear the alert after the witness is restored | — |
| Cold start | a new model version behaves as "trust the model" | — |
| Substitutes | when it names a substitute, is the substitute's grade better? | — |

Downstream KPIs (trucks, P10 hit rate, lei) come once the actions are defined (§10).

## 10. Actions: open

To be specified by the user. The interface is fixed: actions read the verdicts in §7.7 and must treat "no reliable
source" as unknown. One coupling to keep in mind: crew visits caused by actions also produce outcomes. The audit
checks in §5.6 keep grading valid whatever the actions turn out to be.

## 11. Hard situations

- **Witness missing or on fallback:** it is graded and filtered as its fallback (the fallback has its own ledger).
  The verdict says so.
- **No outcome** (gap, gated day): the prediction stays ungraded and trust is unchanged.
- **Long outage of one asset:** its days are gated or ungraded, not counted against the model.
- **All witnesses rejected for an asset:** "no reliable source". Actions decide (likely human review).
- **Judge model cannot load:** ledger-only mode. P(valid) = trust, the same verdict rules apply, and this is logged.
- **Ledger empty** (first day of a run): every verdict is rule 3 or rule 1, so the system behaves like today's
  agent, at validated strength.

## 12. Problems considered and how each is handled

| Problem | Handling |
|---|---|
| Wide bands win "hit rate" | calibration scored as closeness to 0.80 + interval-score skill vs the substitute (§5.3) |
| "Always healthy" looks accurate | alarm precision and miss score kept separately (§5.3) |
| Quality gate hides a broken model's misses | gate uses physics + the asset's own history, never the graded witness (§5.4) |
| 96 readings/day swamp the prior | one score per asset-day (§5.2) |
| One storm hits all homes | bucket-day averaging (§5.7) |
| Prior so heavy trust can't move | n0 = 7 bucket-days < max live evidence ≈ 20 (§6.3) |
| Trust only ever goes down | 14-day decay; the alert clears at a higher level than it is raised (§6.3, §6.5) |
| Peeking at the answer | grader reads only observable sources; truth enters only via simulated crew reports (§5.1) |
| Only acted-on cases get graded | audit checks with inverse-probability weights (§5.6) |
| Using outcomes before they are known | `observed_at` ordering, in training and live (§5.5) |
| Detector graded through another witness | residual check uses pvlib, not PVNet (§5.3) |
| Small buckets give noisy trust | backoff below 5 effective bucket-days + 80 % interval shown (§6.1, §6.3) |
| In-sample witness outputs make training look too easy | train on 2024 validation year, test 2025 (§7.6) |
| Trust never varies in training | drift augmentation (§7.6) |
| No-record case never seen in training | cold-start weeks + hard rule 3 (§7.5, §7.6) |
| Overconfident Judge | isotonic calibration, 5-seed spread reported (§7.4) |
| Filter silences a safety signal | safety pass-through rule (§7.5) |
| New model version inherits old blame | trust keyed by version, starts from its own prior (§6.3) |
| Rare events (M7, M8) can't train a model | ledger-only, stated in the verdict (§7.4) |
| Health alerts flapping | separate raise and clear levels, 3-day persistence, minimum evidence (§6.5) |

What is still not solved:
- Trust can only be as good as the outcomes. On the simulator the crew reports come from our own fault injector, so
  the drift drill shows the loop works, not how real PPC models drift.
- The real-data checks stay labelled non-RO.

## 13. Where it lives

- New package `sentinel/judge/`:
  - `scores.py` (§5.3)
  - `gate.py` (§5.4)
  - `grader.py` (§5.1–5.7)
  - `ledger.py` (§6)
  - `features.py`, `model.py`, `train.py` (§7)
  - `verdict.py` (§7.5)
- Crew reports and audit checks go in `sentinel/agent/world.py`.
- A `judge` slot in `Components` in `sentinel/agent/loop.py`, writing `judge_verdicts` next to predictions.
- `qualify.py` stays until the actions are defined (§10).
- UI: a verdict card (witness claim, trust bar, verdict, reasons) and a trust board (trust over the week per
  witness, health alerts), under `apps/web/src/panels/` following `team/contracts/ui-style.md`.
- `make reproduce` stays offline. Without Judge weights it runs in ledger-only mode.

Suggested cards, in dependency order:
- J1 scores + gate + grader + `outcomes`, with crew reports and audit checks
- J2 ledger + priors in the calibration JSONs + `trust`
- J3 Judge training (`train.py --model judge`) with drift and cold-start augmentation
- J4 verdicts in the loop + `judge_verdicts`
- J5 filter evaluation (§9) + insights
- J6 verdict card + trust board
- then the actions, once specified

## 14. Who decided what

**User decisions:**
- The Judge is the main feature.
- It is a trust layer above the models that decides which information is valid.
- It forms a loop with the witnesses.
- No record → trust the model.
- The actions are specified separately, later.

**My calls** (please check):
- the scores in §5.3 (0.80 calibration target, interval score vs the substitute, the split detector scores)
- the physics-only quality gate
- asset-day and bucket-day evidence units
- `n0 = 7`, 14-day half-life, backoff below 5 bucket-days
- the 2 audit checks per crew type per day, and down-weighting action visits by 0.5
- the 5 % crew misreport rate
- the bucket conditions
- the health alert levels (prior − 0.2 for 3 days, cleared at prior − 0.1)
- LightGBM per family with isotonic calibration
- τ_use for 90 % precision
- training on 12 weeks of 2024 with drift and cold-start augmentation
- ledger-only for M7 and M8
- the safety pass-through
- the file layout

**Left to the builder:**
- exact feature encoding
- the seeds
- wording of the reasons
- the trust board's chart form
- the order of J5 against J6
