import numpy as np

from wattsentry.agent.maintenance_window import QH, best_repair_day, price_profile

D = 10


def _flat(v):
    return np.full((D, QH), float(v))


def test_picks_calm_cheap_day():
    wind, price = _flat(5), _flat(400)
    wind[0] = 20  # day 0 too windy to climb
    power = _flat(2)
    power[3] = 0.1  # low-output day 3 -> cheap downtime
    p = best_repair_day(power, wind, price, [0.0], label="synthetic RO")
    assert p.best_day[0] == 3
    assert np.isinf(p.cost[0, 0])
    assert np.isclose(p.planned[0, 3], 0.1 * 400 * 0.25 * 32)
    assert p.saving_vs_calendar[0] > 0 and p.saving_vs_reactive[0] >= 0


def test_high_risk_pulls_repair_earlier():
    power, wind, price = _flat(2), _flat(5), _flat(400)
    power[6] = 0.1
    calm = best_repair_day(power, wind, price, [0.0])
    risky = best_repair_day(power, wind, price, [0.3], unplanned_extra_days=5)
    assert calm.best_day[0] == 6 and risky.best_day[0] == 0
    assert risky.saving_vs_reactive[0] > 0


def test_vectorised_and_no_climbable_day():
    power = np.stack([_flat(2), _flat(1)])
    wind = np.stack([_flat(20), _flat(5)])
    p = best_repair_day(power, wind, _flat(300), [0.1, 0.1])
    assert p.best_day.tolist()[0] == -1 and p.best_day[1] >= 0
    assert p.best_cost[0] == p.reactive_cost[0] == p.calendar_cost[0]


def test_price_profile_fills_missing_days():
    price = _flat(np.nan)
    price[:2] = 100
    hist = np.tile(np.arange(QH, dtype=float), (7, 1))
    out = price_profile(price, hist)
    assert (out[:2] == 100).all() and np.allclose(out[5], np.arange(QH))
