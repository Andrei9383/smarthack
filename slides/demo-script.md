# Demo script: 3 minutes

Open the shared app once beforehand (the first load is slow). Use the Demo row under the map.
[Brackets] come from logged runs. Say the label aloud: synthetic RO, real RO or non-RO.

| Time | Click | Show | Say |
|---|---|---|---|
| 0:00–0:15 | start on the map | Romania, KPI strip, "[N] homes on [n] feeders, [T] turbines in 3 parks" | "One map, four problems." |
| 0:15–0:40 | **1 Production** *(SHOULD)* | Wind over Dobrogea, parks by expected MWh | "This is where energy will be made." |
| 0:40–0:55 | **2 Demand** *(SHOULD)* | Red zones at the evening peak | "And where it will be needed." |
| 0:55–1:40 | **3 Feeders F7**, toggle Baseline then Sentinel | Export crosses the dashed limit under the baseline, stays under with Sentinel; absorbs midday, discharges evening. Then nominal vs available vs committed vs realised | "We never sell the nominal number." |
| 1:40–2:05 | **4 String failure, 1042** | Band diverges → derated → F7's committed MW drops → work order in the top-K with kWh, lei and MW at risk. An OOD site goes to human review | "It knows when it isn't sure." |
| 2:05–2:45 | **5 O&M park** *(SHOULD; fallback: WindMon rules on the same week)* | One turbine at zero power while neighbours produce → fault + work order. Whole park capped → curtailment, no work order; WindMon's rules fired on both | "Fewer false alarms, same faults." |
| 2:45–3:00 | **6 Summary** | Peak reduction, P10 hit rate, comfort breaches, lei per home, alerts per turbine-day, each with its run ID | Read two numbers and their run IDs |

## If something breaks

- No internet: the app switches to its offline basemap by itself. Force it with `?offline=1`.
- Shared app down: run the laptop copy with `npm start`.
- SHOULD mode not ready: skip its step and add its time to Feeders.
- Summary shows "Not a result": the app is on fixtures. Don't quote those numbers.

## Before presenting

- [ ] Real replay promoted (#29); no amber "pending" left in the slides
- [ ] `make reproduce` works on a clean checkout (#35), since slide 11 promises it
- [ ] Backup screenshot of every panel in `slides/backup/`
- [ ] Rehearsal 1 timed
- [ ] Rehearsal 2 timed
