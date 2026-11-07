# Demo script: 3 minutes

Open the shared app once beforehand (the first load is slow). The app opens on Overview. Run the steps with the **Demo** button (top right) or keys **1–6**; space plays the timeline.
[Brackets] come from logged runs. Say the label aloud: synthetic RO, real RO or non-RO.

| Time | Click | Show | Say |
|---|---|---|---|
| 0:00–0:15 | Overview | Romania, KPI strip, plain-language "needs attention" list, "118 homes on 5 feeders, 70 turbines in 1 park" | "One map, four problems." |
| 0:15–0:40 | **1 Production** *(SHOULD)* | Real wind field, the Constanța park by expected MWh | "This is where energy will be made." |
| 0:40–0:55 | **2 Demand** *(SHOULD)* | Stressed counties at the evening peak (4 counties have homes in this replay) | "And where it will be needed." |
| 0:55–1:40 | **3 Feeders F7**, toggle Baseline then Sentinel | On Wed 4 Jun export crosses the dashed limit under the baseline (270 min) and stays under with Sentinel (0 min). Don't imply every day is clear: F7 is still over on four days, 3–27 kWh. Then nominal vs available vs committed vs realised | "We never sell the nominal number." |
| 1:40–2:05 | **4 String failure, 1042** | Band diverges → derated → F7's committed MW drops → work order in the top-K with kWh, lei and MW at risk. An OOD site goes to human review | "It knows when it isn't sure." |
| 2:05–2:45 | **5 O&M park** *(SHOULD; fallback: WindMon rules on the same week)* | One turbine at zero power while neighbours produce → fault + work order. Whole park capped → curtailment, no work order; WindMon's rules fired on both | "More faults caught, and no truck for curtailment." |
| 2:45–3:00 | **6 Summary** | Peak reduction, P10 hit rate, comfort breaches, lei per home, alerts per turbine-day, each with its run ID | Read energy over limit (1397 → 94.5 kWh) and P10 hit rate (92.0%), with run 506 |

## If something breaks

- No internet: the app switches to its offline basemap by itself. Force it with `?offline=1`.
- Shared app down: run the laptop copy with `npm start`.
- SHOULD mode not ready: skip its step and add its time to Feeders.
- Summary shows "Not a result": the app is on fixtures. Don't quote those numbers.
- Asked about alerts: the summary's 0.70 → 0.11 per turbine-day is this replay week, raw rules vs agent. The test-set comparison after de-duplication is close, 0.074 vs 0.077 (train#302/303). Our edge is recall and telling curtailment from faults.

## Before presenting

- [x] Real replay promoted (#29); no amber "pending" left in the slides
- [ ] `make reproduce` works on a clean checkout (#116), since slide 11 promises it
- [x] Backup screenshot of every step in `slides/backup/` (replay#506)
- [ ] Check frame rate on the demo laptop, especially Demand (only software rendering was tested)
- [ ] Rehearsal 1 timed
- [ ] Rehearsal 2 timed
