# Results

Data: 8 stores, 40 items, 2024-01-01 to 2025-10-01. Training until 2024-12-31, calibration until 2025-03-31, test after that.

## Forecast accuracy (test period, open days)

| Model | Subset | WAPE | Bias |
|---|---|---|---|
| Seasonal naive (last week) | all days | 56.4% | -0.9% |
| Seasonal naive (last week) | promotion days | 55.8% | -47.8% |
| Seasonal naive (last week) | regular days | 56.5% | +10.1% |
| Same-weekday mean, 4 weeks | all days | 45.0% | -0.7% |
| Same-weekday mean, 4 weeks | promotion days | 52.7% | -49.5% |
| Same-weekday mean, 4 weeks | regular days | 43.2% | +10.7% |
| Gradient boosting (Poisson) | all days | 34.1% | -0.1% |
| Gradient boosting (Poisson) | promotion days | 31.7% | -1.0% |
| Gradient boosting (Poisson) | regular days | 34.6% | +0.1% |

## Replenishment policies (all stores and items)

| Policy | Availability (fill rate) | Days with stockout | Write-off rate | Avg. stock (days of demand) |
|---|---|---|---|---|
| A  naive forecast, 95% service everywhere | 94.5% | 8.8% | 15.1% | 1.47 |
| B  ML forecast, 95% service everywhere | 98.6% | 4.3% | 6.2% | 0.81 |
| C  ML forecast, service target by category | 97.3% | 6.7% | 3.2% | 0.69 |

## By category

| Policy | Category | Availability | Write-off rate |
|---|---|---|---|
| A  naive forecast, 95% service everywhere | ambient | 94.7% | 0.0% |
| A  naive forecast, 95% service everywhere | bakery | 91.8% | 40.1% |
| A  naive forecast, 95% service everywhere | chilled | 96.0% | 1.2% |
| A  naive forecast, 95% service everywhere | fresh_produce | 94.5% | 14.5% |
| B  ML forecast, 95% service everywhere | ambient | 98.3% | 0.0% |
| B  ML forecast, 95% service everywhere | bakery | 98.7% | 24.7% |
| B  ML forecast, 95% service everywhere | chilled | 98.6% | 0.0% |
| B  ML forecast, 95% service everywhere | fresh_produce | 98.8% | 2.7% |
| C  ML forecast, service target by category | ambient | 99.1% | 0.0% |
| C  ML forecast, service target by category | bakery | 93.7% | 14.2% |
| C  ML forecast, service target by category | chilled | 98.6% | 0.0% |
| C  ML forecast, service target by category | fresh_produce | 97.3% | 1.5% |

## Allocation of short promotion supply

586 promotion item-days in the test period. Share of the allocated units that were sold:

| Supply as share of forecast | By store size | Proportional to forecast | Greedy, highest sell probability |
|---|---|---|---|
| 50% | 98.4% | 98.3% | 98.3% |
| 70% | 94.7% | 94.5% | 94.5% |
| 85% | 90.3% | 90.0% | 90.0% |
| 100% | 84.9% | 84.5% | 84.5% |

Run time: 35 s.
