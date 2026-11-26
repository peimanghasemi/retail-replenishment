"""Run the full pipeline: data, forecasts, replenishment simulation, allocation.

Usage:  python scripts/run_all.py
Writes CSV files and results/summary.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retailrep import allocation, data, forecast, replenishment, uncertainty  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

TRAIN_END = "2024-12-31"
CALIB_END = "2025-03-31"


def main() -> None:
    t0 = time.time()
    sales, items, stores = data.generate()
    sales.to_csv(OUT / "sales.csv.gz", index=False)
    df = forecast.add_features(sales)

    train = df[df["date"] <= TRAIN_END]
    model = forecast.fit_gbm(train)

    df["f_naive"] = forecast.seasonal_naive(df)
    df["f_ma"] = forecast.moving_average(df)
    df["f_gbm"] = forecast.predict_gbm(model, df)

    calib = df[(df["date"] > TRAIN_END) & (df["date"] <= CALIB_END)]
    test = df[df["date"] > CALIB_END].copy()

    # forecast accuracy on open days of the test period
    rows = []
    te = test[test["open"] == 1]
    for name, col in [("Seasonal naive (last week)", "f_naive"),
                      ("Same-weekday mean, 4 weeks", "f_ma"),
                      ("Gradient boosting (Poisson)", "f_gbm")]:
        for subset, mask in [("all days", slice(None)), ("promotion days", te["promo"] == 1),
                             ("regular days", te["promo"] == 0)]:
            part = te.loc[mask]
            rows.append({"model": name, "subset": subset,
                         "WAPE": forecast.wape(part["demand"], part[col]),
                         "bias": forecast.bias(part["demand"], part[col])})
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / "forecast_accuracy.csv", index=False)

    # replenishment policies
    err_naive = uncertainty.fit_error_model(calib, "f_naive")
    err_gbm = uncertainty.fit_error_model(calib, "f_gbm")
    shelf = items.set_index("item_id")["shelf_life"]
    z95 = replenishment.z_for(0.95)
    policies = {
        "A  naive forecast, 95% service everywhere": ("f_naive", err_naive, None),
        "B  ML forecast, 95% service everywhere": ("f_gbm", err_gbm, None),
        "C  ML forecast, service target by category": ("f_gbm", err_gbm, replenishment.CATEGORY_SERVICE),
    }
    sim_rows = []
    for pname, (col, err, service) in policies.items():
        for (store, item), g in test.groupby(["store_id", "item_id"], sort=False):
            cat = g["category"].iat[0]
            f = g[col].to_numpy()
            s = uncertainty.sigma(f, cat, err)
            z = replenishment.z_for(service[cat]) if service else z95
            r = replenishment.simulate_series(g["demand"].to_numpy(), f, s,
                                              g["open"].to_numpy().astype(bool),
                                              int(shelf[item]), z)
            sim_rows.append({"policy": pname, "store_id": store, "item_id": item,
                             "category": cat, **r.__dict__})
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT / "replenishment_by_item.csv", index=False)

    # KPIs computed in SQL, the way they would be pulled from a warehouse table
    con = duckdb.connect()
    con.register("sim", sim)
    kpi_sql = (ROOT / "sql" / "kpis.sql").read_text()
    kpi_policy = con.execute(kpi_sql.replace("{group}", "policy")).df()
    kpi_cat = con.execute(kpi_sql.replace("{group}", "policy, category")).df()
    kpi_policy.to_csv(OUT / "kpis_by_policy.csv", index=False)
    kpi_cat.to_csv(OUT / "kpis_by_category.csv", index=False)

    # allocation of short promotion supply, at several supply levels
    k_hat = 8.0
    alloc_rows = []
    promo_days = test[(test["promo"] == 1) & (test["open"] == 1)]
    size = stores.set_index("store_id")["size_factor"]
    for ratio in (0.5, 0.7, 0.85, 1.0):
        for (date, item), g in promo_days.groupby(["date", "item_id"]):
            mean = g["f_gbm"].to_numpy()
            dem = g["demand"].to_numpy()
            supply = int(ratio * mean.sum())
            a_size = allocation.proportional(supply, size.loc[g["store_id"]].to_numpy())
            a_fc = allocation.proportional(supply, mean)
            a_gr = allocation.greedy(supply, mean, k_hat)
            alloc_rows.append({"supply_ratio": ratio, "date": date, "item_id": item,
                               "supply": supply, "demand": dem.sum(),
                               "sold_by_store_size": allocation.sold(a_size, dem),
                               "sold_by_forecast": allocation.sold(a_fc, dem),
                               "sold_greedy": allocation.sold(a_gr, dem)})
    alloc = pd.DataFrame(alloc_rows)
    alloc.to_csv(OUT / "allocation.csv", index=False)

    write_summary(acc, kpi_policy, kpi_cat, alloc, sales, time.time() - t0)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_summary(acc, kpi_policy, kpi_cat, alloc, sales, secs) -> None:
    lines = ["# Results", "",
             f"Data: {sales['store_id'].nunique()} stores, {sales['item_id'].nunique()} items, "
             f"{sales['date'].min():%Y-%m-%d} to {sales['date'].max():%Y-%m-%d}. "
             f"Training until {TRAIN_END}, calibration until {CALIB_END}, test after that.", "",
             "## Forecast accuracy (test period, open days)", "",
             "| Model | Subset | WAPE | Bias |", "|---|---|---|---|"]
    for r in acc.itertuples():
        lines.append(f"| {r.model} | {r.subset} | {pct(r.WAPE)} | {r.bias:+.1%} |")
    lines += ["", "## Replenishment policies (all stores and items)", "",
              "| Policy | Availability (fill rate) | Days with stockout | Write-off rate | Avg. stock (days of demand) |",
              "|---|---|---|---|---|"]
    for r in kpi_policy.itertuples():
        lines.append(f"| {r.policy} | {pct(r.fill_rate)} | {pct(r.stockout_rate)} | "
                     f"{pct(r.writeoff_rate)} | {r.days_of_stock:.2f} |")
    lines += ["", "## By category", "",
              "| Policy | Category | Availability | Write-off rate |", "|---|---|---|---|"]
    for r in kpi_cat.itertuples():
        lines.append(f"| {r.policy} | {r.category} | {pct(r.fill_rate)} | {pct(r.writeoff_rate)} |")
    cols = ["supply", "sold_by_store_size", "sold_by_forecast", "sold_greedy"]
    tot = alloc.groupby("supply_ratio")[cols].sum()
    n_days = alloc.groupby("supply_ratio").size().iloc[0]
    lines += ["", "## Allocation of short promotion supply", "",
              f"{n_days} promotion item-days in the test period. Share of the allocated units that were sold:", "",
              "| Supply as share of forecast | By store size | Proportional to forecast | Greedy, highest sell probability |",
              "|---|---|---|---|"]
    for ratio, r in tot.iterrows():
        lines.append(f"| {ratio:.0%} | {pct(r.sold_by_store_size / r.supply)} | "
                     f"{pct(r.sold_by_forecast / r.supply)} | {pct(r.sold_greedy / r.supply)} |")
    lines += ["", f"Run time: {secs:.0f} s."]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
