# Data provenance: what is real and what is synthetic

Sources: [README.md](../README.md) (Results, Limits) and [data/README.md](../data/README.md) (every raw file,
source and licence).

## The project in short

**Prosumer Sentinel** is an AI agent for PPC (Romania), built in 24 h at Smarthack XI. It has two jobs:

- **Low-voltage feeders:** it plans home-battery charging so midday solar export stays under each feeder's limit,
  then releases that energy into the evening peak.
- **Wind fleets:** it tells real turbine faults apart from curtailment, icing and sensor noise, and ranks which
  turbines to inspect.

It only commits flexibility it is fairly sure it can deliver (the P10, a pessimistic estimate). Its outputs are
battery schedules, ranked work orders, alerts and day-ahead demand per zone. Every action is logged with the model
prediction behind it. All models are trained in this repo and run locally as ONNX files.

## Real data (all public)

- **Weather:** ERA5 hourly, 2022–2025, at 33 Romanian towns, 3 wind parks and a grid covering Romania
  (via Open-Meteo).
- **National grid:** Romanian load and generation by type, 2022–2025 (ENTSO-E data, via Energy-Charts).
- **Prices:** OPCOM day-ahead prices for Romania, 2023–2025.
- **Places:** wind park locations (OpenStreetMap turbines), county and country borders (geoBoundaries), and the
  real Romanian towns the simulated homes are placed in.
- **Non-Romanian, for validation only (never used for training):**
  - 4 real PV systems in the US (NREL PVDAQ)
  - real wind turbine SCADA (CARE-to-Compare)

## Synthetic data (our simulators, driven by the real weather above)

- **Homes:** every prosumer household, with its PV output, consumption and battery telemetry. There is no
  personal data.
- **Feeders:** the feeders, their limits, and the GIS records, which are deliberately wrong for about 20 % of
  entries.
- **Wind SCADA:** turbine data for the 3 parks.
- **Faults:** all PV and wind fault labels, which are injected so we know exactly when each one starts.
- **Control:** no real battery or turbine is controlled. Control happens only in the replay.

## Which results use which data

- **Real Romanian data:** only DemandNet (national demand forecast). Its error (nMAE) is 0.039 against 0.067 for
  the persistence baseline.
- **Synthetic Romanian data:** all other headline numbers, including:
  - feeder export peak shaving
  - P10 hit rate of 92 %
  - PV and load forecasts
  - fault detection
  - wind precision and recall
  - money saved, which mixes the simulated homes with real OPCOM prices
- **Real non-Romanian data:**
  - On the US PV systems, PVNet beats the simpler pvlib baseline by only 14 %.
  - On real wind SCADA, the early-failure model does not transfer.

**Caveat:** the PV simulator is built on pvlib, so beating pvlib on synthetic data partly reflects our own
generator.
