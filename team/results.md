# Results sheet (success criteria #1–#13, plan §2)

For the slides (card #63). Compiled 2026-11-07 from lab runs and card notes only. Copy numbers with their run ID and label.
Labels: **syn RO** = synthetic Romanian fleet/parks (our simulator, real ERA5 weather); **real RO** = real Romanian data
(ENTSO-E / Energy-Charts load, OPCOM prices); **non-RO** = real data from outside Romania (PVDAQ, CARE-to-Compare).
Replay rows use replay#506 (#29, final demo replay: fleet-trained M1/M2/M4 from #14 train#383–392, #24 WindMon rules).

| # | Claim | Ours | Baselines | Target | Status | Label | Run |
|---|---|---|---|---|---|---|---|
| 1 | Feeder export peak shaving (headline) | minutes over limit 1905 (54 min/feeder-day); energy over limit 94.5 kWh; worst overshoot 10.6 kW; peak reduction 19.1 % (24.1 % on congested feeder-days) | self-consumption rule: 6645 min, 1397 kWh, 29.1 kW; tuned threshold rule: not run | zero violations where headroom suffices | **partial** (−71 % minutes, −93 % kWh over; hero day F7 Wed cleared, not zero overall) | syn RO + real OPCOM prices | `replay#506` |
| 2 | Evening import peak reduction per feeder | not reported | not reported | reported | **not measured** | syn RO | — |
| 3 | P10 commitment reliability | 92.0 % of feeder-steps (1845 kWh committed, 16 kWh short) | n/a | ≥ 90 % | **met** | syn RO | `replay#506` |
| 4 | PV nowcast nMAE + 10–90 % coverage | nMAE 0.022–0.024 (4 seeds); coverage 0.76–0.79 (seen 0.82–0.85, unseen 0.60–0.63). PVDAQ: nMAE 0.266, coverage 0.09 | pvlib 0.073, LightGBM 0.044–0.046, persistence 0.279. PVDAQ: pvlib 0.312, LightGBM 0.285, persistence 0.538 | ≥ 20 % below pvlib; coverage 78–82 % | **partial**: syn RO −67–69 % vs pvlib (met); coverage in range for 2 of 4 seeds; PVDAQ only −15 % vs pvlib and coverage 0.09 (missed) | syn RO; non-RO (PVDAQ) | `train#383–386`; `train#377` (PVDAQ) |
| 5 | Household load forecast | nMAE 0.221–0.228 (4 seeds), coverage 0.80–0.81 | persistence D-1 0.360, D-7 mean 0.342; LightGBM 0.259 | beat persistence | **met** (−39 % vs D-1, −15 % vs LightGBM) | syn RO | `train#387–390` |
| 6 | Site faults: PR-AUC + hours-to-detect | PR-AUC 0.69 (seen 0.73, unseen 0.68); 40/40 eval episodes found, median 20 h to detect. PVDAQ+injected: PR-AUC 0.48 | threshold rule PR-AUC 0.23, CUSUM 0.22; threshold+CUSUM 16/40 found, median 35.9 h. PVDAQ+injected: 0.38 | beat both | **met** on syn RO; non-RO PR-AUC only (precision 0.25) | syn RO; non-RO (PVDAQ + injected) | `train#297`; `train#377` (eval.py) |
| 7 | Comfort-reserve breaches; lei/home/month | 0 breaches; 23.7 lei/home/month saved, net of battery degradation | default rule: 1165 breaches | zero breaches; lei reported | **met** | syn RO + real OPCOM prices | `replay#506` |
| 8 | Wind anomalies vs WindMon | precision 0.89, recall 0.91, 0.074 alerts/turbine-day, curtail-vs-fault F1 0.62 (label-free NBM, mean of 2 seeds) | WindMon: precision 0.84, recall 0.75, 0.077 alerts/turbine-day (0.142 before dedup), F1 0.39 | fewer alerts at equal or better recall | **met** (recall +16 pts; alerts only 4 % fewer) | syn RO, held-out park WP2, test 2025 | `train#302/303` (labels: `train#300/301`) |
| 9 | Area demand P50/P90, peak hour | nMAE 0.039; peak-hour error 0.30 h; 10–90 coverage 0.74 | persistence-week 0.067, day 0.069, national shape 0.070; peak-hour: persistence 0.28–0.29 h, shape 0.76 h | beat persistence | **partial**: nMAE met (−42 %); peak hour does not beat persistence; coverage under 78 % | real RO (ENTSO-E load) | no lab run (local) |
| 10 | Grid-twin: corrupted GIS records recovered | phase records 83.3 % recovered (precision 69.2 %, 54 corrupted); feeder records 100 % (precision 100 %, 16 corrupted) | GIS as-is: 0 % | reported | **met** (reported) | syn RO fleet (not the planned IEEE feeder) | `replay#506` |
| 11 | Wind park forecast 24 h | park WP2 nMAE 0.0056 at h24 (0.0043 h1); turbine pinball 0.0063; P10–P90 coverage 0.77–0.80 | binned power curve 0.0090, persistence 0.173 (h24); class curve turbine nMAE 0.036 | beat both | **met** on syn RO; real RO is shape only (hourly corr with ENTSO-E RO wind 0.914 vs binned 0.913) | syn RO; real RO (shape) | `train#418/419` (experiment) |
| 12 | Early failure warning: lead time; CARE | lead time 120 h, recall 0.98, 0.42 false alarms/turbine-year; held-out park 165 h. CARE score 0.40, event recall 0 | fixed residual threshold: 46 h, recall 0.87, 0.39 FA/turbine-year; CARE 0.39–0.41 | real lead time on held-out events | **partial**: met on syn RO; **missed** on CARE (non-RO) | syn RO; non-RO (CARE Farm A) | `train#379` (physics residuals: `train#378`) |
| 13 | UI mode switch < 1 s on demo laptop | not measured | n/a | < 1 s | **not measured** | n/a | manual |

## Caveats and misses

- **#1 is not zero.** 1905 minutes over the limit remain. Hero feeder F7 on Wed 4 Jun goes 90.3 kW / 270 min → 61.7 kW / 0 min (F10 the same day 270 → 0), but F7 is still over on Mon/Tue/Thu/Fri (150–375 min, 2.8–27.3 kWh). Minutes can rise while the overload falls (F7 Mon 285 → 375 min but 79.1 → 27.3 kWh): the LP penalises energy above the limit, so quote kWh over the limit next to minutes.
- **#1 has one baseline, not two.** Only the self-consumption rule ran; the tuned per-home threshold rule was never run.
- **#3** was 82.2 % on the interim weights (replay#380); the fleet-trained LoadNet (coverage 0.81) brings it to 92.0 %, with or without online conformal. Mock-world drill numbers (0.96–0.98) are local diagnostics, not results.
- **#4 on real PV is weak.** On PVDAQ (non-RO) PVNet beats pvlib by only 15 % and its 10–90 band covers 9 % (bands don't transfer). Synthetic gains are partly circular: our simulator uses pvlib physics and the same weather. Unseen-site coverage is 0.60–0.63.
- **#6**: train#297 gives CUSUM a 1 h median detection time, but only on the 10 % of faults it catches within 72 h. eval.py (#377) is the comparison to quote. Soiling AP is 0.29. With the fleet-trained M4 (replay#506) story site 1042's string outage stays `derated` all week (the interim M4 flipped it to sensor_fault).
- **#8**: the alert reduction vs de-duplicated WindMon is small (0.074 vs 0.077). The large gap on the older stand-in data (run#135: 0.101 vs 0.114, precision 0.85 vs 0.53) is not the comparison to quote: WindMon is much stronger on #22 data. The LightGBM classifier and thresholds still use labels. In the demo replay (replay#506, one week, WP01) M6 gives 0.114 alerts/turbine-day vs 0.698 for #24's WindMon rules (not de-duplicated); quote it as the demo-week number, and train#302/303 as the held-out evaluation. No full confusion matrix extracted, only curtail-vs-fault F1.
- **#9** comes from a local CPU run (no lab run ID). Peak-hour error 0.30 h is no better than persistence (0.28–0.29 h). The county split uses synthetic shares (nMAE 0.061 vs 0.077).
- **#10** ran on our synthetic fleet's voltages, not on the IEEE European LV feeder named in the plan. With tau 0.68–0.90 on the demo feeders, the twin tightens G by up to 6 % (F7 65.5 → 61.8 kW).
- **#11** is an experiment card (merge=never). It uses ERA5 as a perfect NWP (no forecast error), and the simulator is clean, so gains are optimistic.
- **#12 does not transfer to real SCADA.** On CARE Farm A (non-RO) M7 scores 0.40, the same as the baseline (0.39–0.41), with event recall 0. About 88 % of synthetic false alarms are drifts the simulator never ended in failure. Scoring rules were changed after the first results, but the same rules apply to both methods.

## Not yet measured

- **#2** Evening import peak per feeder: the replay computes it internally (`peak_import_evening`) but does not print it; no run reports it.
- **#13** UI mode switch < 1 s: no timing recorded on the demo laptop.
- **#1** tuned per-home threshold baseline; **#6** hours-to-detect on PVDAQ + injected; **#9** synthetic-RO and zone P90 numbers from a lab run.
