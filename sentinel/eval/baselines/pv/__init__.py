"""PV / load / fault baselines (card #15): pvlib (recorded metadata), persistence, LightGBM quantiles, threshold
rule and CUSUM on the pvlib residual. Entry point: python -m sentinel.eval.baselines.pv.run

Predictors (contracts.Predictor): lgbm.LGBMPredictor (M1 / M2 fallback), persistence.PvlibPredictor,
persistence.PersistencePredictor, faults.ThresholdCusumPredictor (M4 fallback). CUSUM reference: cusum.cusum_two_sided.
"""
