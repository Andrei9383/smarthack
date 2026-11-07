"""PV / load / fault baselines (card #15): pvlib (recorded metadata), persistence, LightGBM quantiles, threshold
rule and CUSUM on the pvlib residual. Entry point: python -m sentinel.eval.baselines.pv.run

    from sentinel.eval.baselines.pv import predictors
    P = predictors(fleet_dir)   # {"pvlib": {"pv"}, "persistence": {"pv", "load"}, "lgbm": {"pv", "load"},
                                #  "threshold_cusum": {"sitehealth"}} -> contracts.Predictor objects

Classes: lgbm.LGBMPredictor (M1 / M2 fallback), persistence.PvlibPredictor, persistence.PersistencePredictor,
faults.ThresholdCusumPredictor (M4 fallback). CUSUM numpy reference (kernel K3): cusum.cusum_two_sided.
"""


def predictors(fleet_dir=None, **fit_kw):
    """baseline name -> {kind: Predictor}; see registry.py."""
    from .registry import predictors as _p

    return _p(fleet_dir, **fit_kw)
