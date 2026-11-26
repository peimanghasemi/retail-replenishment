import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retailrep import allocation, data, forecast, replenishment  # noqa: E402


def small_data():
    sales, items, stores = data.generate(n_stores=2, n_items=6, days=120, seed=1)
    return sales, items, stores


def test_generator_is_reproducible_and_closed_on_sundays():
    a, _, _ = small_data()
    b, _, _ = small_data()
    pd.testing.assert_frame_equal(a, b)
    sundays = a[a["date"].dt.weekday == 6]
    assert (sundays["demand"] == 0).all() and (sundays["open"] == 0).all()


def test_promotions_raise_demand():
    sales, _, _ = data.generate(n_stores=4, n_items=20, days=200, seed=3)
    opened = sales[sales["open"] == 1]
    assert opened.loc[opened.promo == 1, "demand"].mean() > 1.4 * opened.loc[opened.promo == 0, "demand"].mean()


def test_features_do_not_look_ahead():
    sales, _, _ = small_data()
    df = forecast.add_features(sales)
    g = df[(df.store_id == "S00") & (df.item_id == "I000")].reset_index(drop=True)
    # lag_7 on day t must equal demand on day t-7
    assert np.allclose(g.loc[7:, "lag_7"].to_numpy(), g.loc[:len(g) - 8, "demand"].to_numpy())
    # changing today's demand must not change today's features
    sales2 = sales.copy()
    sales2.loc[sales2.index[-1], "demand"] += 1000
    f1 = forecast.add_features(sales).iloc[-1][forecast.FEATURES[:-3]]
    f2 = forecast.add_features(sales2).iloc[-1][forecast.FEATURES[:-3]]
    assert np.allclose(f1.astype(float).fillna(-1), f2.astype(float).fillna(-1))


def test_perfect_forecast_no_safety_stock_sells_everything_without_waste():
    demand = np.array([5, 7, 6, 0, 8, 9, 4, 6])
    is_open = demand > 0
    r = replenishment.simulate_series(demand, demand.astype(float), np.zeros(len(demand)),
                                      is_open, shelf_life=1, z=0.0)
    assert r.sold == r.demand
    assert r.written_off == 0


def test_short_shelf_life_writes_off_overstock():
    demand = np.array([2, 2, 2, 2, 2, 2])
    forecast_ = np.full(6, 10.0)
    r = replenishment.simulate_series(demand, forecast_, np.zeros(6), np.ones(6, bool),
                                      shelf_life=1, z=0.0)
    assert r.written_off > 0
    assert r.sold == r.demand


def test_stock_balance():
    rng = np.random.default_rng(0)
    demand = rng.poisson(10, 60)
    is_open = np.ones(60, bool)
    r = replenishment.simulate_series(demand, np.full(60, 10.0), np.full(60, 3.0), is_open,
                                      shelf_life=3, z=1.0, initial_stock=0)
    # everything received is sold, written off or still on the shelf at the end
    assert r.sold + r.written_off <= r.received


def test_allocation_rules_use_all_supply():
    mean = np.array([3.0, 10.0, 25.0, 0.5])
    for alloc in (allocation.greedy(30, mean), allocation.proportional(30, mean)):
        assert alloc.sum() == 30 and (alloc >= 0).all()


def test_greedy_prefers_store_with_higher_demand():
    alloc = allocation.greedy(10, np.array([1.0, 20.0]))
    assert alloc[1] > alloc[0]
