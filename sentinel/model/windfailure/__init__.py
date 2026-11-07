"""M7 WindFailure (card #43, OPTIONAL): P(component failure within 7 / 14 days) per turbine-day.

Built on temperature residuals (measured − expected component temperature). Two residual sources:
- `physics` (default, CPU): a label-free linear thermal normal-behaviour model per turbine class and component
  (thermal.py): rise over ambient regressed on a bank of first-order lags of load^γ and ambient, the same structure
  as the simulator's thermal relation (plan §4). Fitted on training parks / training time, every step that has
  data and no error code (no labels, no drift mask).
- `nbm` (torch): M6's WindNBM GRU (sentinel.model.wind.nbm) trained in-process on the thermal objective with the
  same label-free mask (residuals_nbm.py).

Modules: labels.py (failure episodes → daily targets), thermal.py, residuals_nbm.py, features.py (daily per
turbine-component features), metrics.py (alarm episodes, event recall, lead time, false alarms), train.py
(`python -m sentinel.model.windfailure.train`), predictor.py (per-turbine risk for the replay / O&M drawer).

Rule R12 (team/plan.md §8): M7 ships only with real lead time on held-out events; train.py prints `wf_ship`.
"""

COMPONENTS = ["gearbox_t", "gen1_t", "gen2_t", "conv_t", "transformer_t"]  # contract sensor names
COMPONENT_LABELS = {
    "gearbox_t": "gearbox",
    "gen1_t": "generator bearing 1",
    "gen2_t": "generator bearing 2",
    "conv_t": "converter",
    "transformer_t": "transformer",
}
HORIZONS_D = (7, 14)
