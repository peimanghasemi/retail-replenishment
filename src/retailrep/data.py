"""Synthetic store-product-day data.

Real retail sales data is not public at the level needed here (store x item x day
with promotions and shelf life), so the generator below produces data with the
patterns that matter for replenishment: weekly and yearly seasonality, store size,
promotions with strong uplift, holidays and overdispersed daily demand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = {
    # name: (share of items, shelf life range in days, base daily units range)
    "fresh_produce": (0.30, (3, 5), (8, 30)),
    "bakery": (0.15, (1, 2), (10, 40)),
    "chilled": (0.25, (7, 14), (5, 20)),
    "ambient": (0.30, (180, 365), (3, 15)),
}

WEEKDAY_FACTOR = np.array([0.85, 0.90, 0.95, 1.00, 1.20, 1.35, 0.0])  # Mon..Sun, closed Sunday

AUSTRIAN_HOLIDAYS = [
    "01-01", "01-06", "05-01", "08-15", "10-26", "11-01", "12-08", "12-25", "12-26",
]


def make_items(n_items: int, rng: np.random.Generator) -> pd.DataFrame:
    names = list(CATEGORIES)
    shares = np.array([CATEGORIES[c][0] for c in names])
    cats = rng.choice(names, size=n_items, p=shares / shares.sum())
    rows = []
    for i, c in enumerate(cats):
        _, (sl_lo, sl_hi), (b_lo, b_hi) = CATEGORIES[c]
        rows.append({
            "item_id": f"I{i:03d}",
            "category": c,
            "shelf_life": int(rng.integers(sl_lo, sl_hi + 1)),
            "base_demand": float(rng.uniform(b_lo, b_hi)),
            "promo_uplift": float(rng.uniform(1.6, 2.8)),
            "season_amp": float(rng.uniform(0.0, 0.25)),
            "season_peak": int(rng.integers(0, 365)),
        })
    return pd.DataFrame(rows)


def make_stores(n_stores: int, rng: np.random.Generator) -> pd.DataFrame:
    size = rng.lognormal(mean=0.0, sigma=0.35, size=n_stores)
    return pd.DataFrame({"store_id": [f"S{s:02d}" for s in range(n_stores)],
                         "size_factor": size / size.mean()})


def make_promotions(items: pd.DataFrame, dates: pd.DatetimeIndex,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Weekly flyer promotions: each week about 10% of items are on promotion,
    running Thursday to Wednesday as in a typical discounter flyer."""
    rows = []
    week_starts = dates[dates.weekday == 3]
    for ws in week_starts:
        on = rng.random(len(items)) < 0.10
        for item in items.loc[on, "item_id"]:
            for d in pd.date_range(ws, periods=7):
                rows.append((d, item))
    promo = pd.DataFrame(rows, columns=["date", "item_id"]).drop_duplicates()
    promo["promo"] = 1
    return promo


def generate(n_stores: int = 8, n_items: int = 40, start: str = "2024-01-01",
             days: int = 640, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (sales, items, stores). `demand` is the true customer demand."""
    rng = np.random.default_rng(seed)
    items = make_items(n_items, rng)
    stores = make_stores(n_stores, rng)
    dates = pd.date_range(start, periods=days, freq="D")
    promo = make_promotions(items, dates, rng)

    grid = pd.MultiIndex.from_product(
        [dates, stores.store_id, items.item_id], names=["date", "store_id", "item_id"]
    ).to_frame(index=False)
    df = grid.merge(stores, on="store_id").merge(items, on="item_id")
    df = df.merge(promo, on=["date", "item_id"], how="left")
    df["promo"] = df["promo"].fillna(0).astype(int)

    md = df["date"].dt.strftime("%m-%d")
    df["holiday"] = md.isin(AUSTRIAN_HOLIDAYS).astype(int)
    # shops close on public holidays and Sundays; the day before is busier
    before = (df["date"] + pd.Timedelta(days=1)).dt.strftime("%m-%d").isin(AUSTRIAN_HOLIDAYS)
    df["pre_holiday"] = before.astype(int)

    doy = df["date"].dt.dayofyear.to_numpy()
    season = 1 + df["season_amp"] * np.cos(2 * np.pi * (doy - df["season_peak"]) / 365.25)
    weekday = WEEKDAY_FACTOR[df["date"].dt.weekday.to_numpy()]
    mu = (df["base_demand"] * df["size_factor"] * season * weekday
          * np.where(df["promo"] == 1, df["promo_uplift"], 1.0)
          * np.where(df["pre_holiday"] == 1, 1.3, 1.0)
          * np.where(df["holiday"] == 1, 0.0, 1.0))

    # negative binomial noise, dispersion k=8 (variance = mu + mu^2/k)
    k = 8.0
    lam = rng.gamma(shape=k, scale=np.maximum(mu, 1e-9) / k)
    df["demand"] = np.where(mu > 0, rng.poisson(lam), 0)
    df["open"] = (mu > 0).astype(int)

    cols = ["date", "store_id", "item_id", "category", "promo", "holiday",
            "pre_holiday", "open", "demand"]
    return df[cols].sort_values(["store_id", "item_id", "date"]).reset_index(drop=True), items, stores
