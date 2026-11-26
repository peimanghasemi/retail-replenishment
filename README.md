# retail-replenishment

Demand forecasting, daily store replenishment and allocation of short promotion stock for a grocery discounter, with the trade-off that matters most for fresh food: **availability on the shelf versus write-offs**.

The pipeline runs end to end in under a minute on a laptop:

1. **Data.** 8 stores, 40 items, 21 months of daily demand with weekly flyer promotions, Austrian public holidays, Sunday closing, yearly seasonality and overdispersed noise. Four categories with very different shelf lives (bakery 1–2 days, fresh produce 3–5, chilled 7–14, ambient 6–12 months).
2. **Forecast.** One-day-ahead forecasts per store and item. Two simple baselines against a gradient boosting model with Poisson loss and features for promotions, holidays, weekday, season, recent level and each item's past promotion response.
3. **Replenishment.** Daily simulation of every store and item with morning delivery, evening order, FIFO selling and batch-level expiry. Safety stock comes from a forecast error model fitted on a calibration window.
4. **Allocation.** When the distribution centre has less promotion stock than the stores need, the quantity is split by store size, in proportion to the forecast, or greedily by the probability that each next unit sells.
5. **KPIs in SQL.** Fill rate, stockout days, write-off rate and stock cover are computed with DuckDB from the simulation output, in the same way they would be pulled from a warehouse table.

## Results

Test period April to October 2025 (open days only).

**Forecast accuracy**

| Model | WAPE all days | WAPE promotion days | Bias on promotion days |
|---|---|---|---|
| Seasonal naive (same day last week) | 56.4% | 55.8% | −47.8% |
| Same-weekday mean over 4 weeks | 45.0% | 52.7% | −49.5% |
| Gradient boosting (Poisson) | **34.1%** | **31.7%** | **−1.0%** |

The baselines miss promotions badly, under-forecasting them by almost half. The model knows the flyer calendar and how strongly each item reacted to earlier promotions, which removes that bias. The remaining error is close to the noise level of single store-item-days.

**Replenishment policies**

| Policy | Availability (fill rate) | Days with a stockout | Write-off rate | Stock (days of demand) |
|---|---|---|---|---|
| A. Naive forecast, 95% service target everywhere | 94.5% | 8.8% | 15.1% | 1.47 |
| B. ML forecast, 95% service target everywhere | 98.6% | 4.3% | 6.2% | 0.81 |
| C. ML forecast, service target by category | 97.3% | 6.7% | **3.2%** | **0.69** |

* Moving from A to B improves both sides at once. Availability rises by 4 points while write-offs fall from 15% to 6% and stock drops by 45%. A better forecast needs less safety stock to reach the same service level.
* C sets the service target per category from the newsvendor ratio. Bakery and fresh produce lose the whole purchase price when they expire, so their target is lower (75% and 88%) than for chilled (95%) and ambient (98%). This halves write-offs again, from 6.2% to 3.2%, for 1.3 points of availability, almost all of it in bakery.

| Category | Availability B → C | Write-off rate B → C |
|---|---|---|
| Bakery | 98.7% → 93.7% | 24.7% → 14.2% |
| Fresh produce | 98.8% → 97.3% | 2.7% → 1.5% |
| Chilled | 98.6% → 98.6% | 0.0% → 0.0% |
| Ambient | 98.3% → 99.1% | 0.0% → 0.0% |

Whether C is better than B is a business decision. It depends on the real margin of each item and on how much a missing loaf of bread costs in customer goodwill. The point of the simulation is to put a number on that trade-off per category.

**Allocation of short promotion stock** (586 promotion item-days)

| Supply as share of forecast | By store size | Proportional to forecast | Greedy by sell probability |
|---|---|---|---|
| 50% | 98.4% | 98.3% | 98.3% |
| 70% | 94.7% | 94.5% | 94.5% |
| 85% | 90.3% | 90.0% | 90.0% |
| 100% | 84.9% | 84.5% | 84.5% |

This one did not pay off. The three rules sell almost the same share of the allocated units. In this data all stores have the same demand pattern, only scaled by size, so every reasonable rule ends up with nearly the same split. The greedy rule should only make a difference when stores differ in how predictable their demand is, for example small rural stores next to large city stores. I kept the result in because it is a useful finding. The effort is better spent on the forecast and on the replenishment parameters than on a more complex allocation rule.

## Run it

```bash
pip install -e ".[dev]"
python scripts/run_all.py      # writes results/*.csv and results/summary.md
pytest -q
```

## Project layout

```
src/retailrep/
  data.py            synthetic store x item x day demand
  forecast.py        features without look-ahead, baselines, gradient boosting
  uncertainty.py     forecast error model (var = a*f + b*f^2 per category)
  replenishment.py   daily order-up-to policy with FIFO batches and expiry
  allocation.py      store-size, proportional and greedy allocation
sql/kpis.sql         availability and write-off KPIs
scripts/run_all.py   full pipeline
tests/               unit tests (look-ahead, stock balance, allocation)
results/             outputs of the last run
```
