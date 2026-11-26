"""One-day-ahead demand forecasts at store x item level."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["weekday", "doy_sin", "doy_cos", "promo", "pre_holiday", "open",
            "lag_7", "lag_14", "same_wd_mean_4", "roll_28", "promo_ratio",
            "store_code", "item_code", "cat_code"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features use only information known at the end of the previous day."""
    out = df.copy()
    out["weekday"] = out["date"].dt.weekday
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    g = out.groupby(["store_id", "item_id"], sort=False)["demand"]
    out["lag_7"] = g.shift(7)
    out["lag_14"] = g.shift(14)
    out["same_wd_mean_4"] = (g.shift(7) + g.shift(14) + g.shift(21) + g.shift(28)) / 4
    # 28-day mean over open, non-promo days, as a clean baseline level
    base = out["demand"].where((out["open"] == 1) & (out["promo"] == 0))
    out["roll_28"] = (base.groupby([out["store_id"], out["item_id"]], sort=False)
                      .transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean()))
    # how strongly this item reacted to its previous promotions
    promo_mean = (out["demand"].where(out["promo"] == 1)
                  .groupby([out["store_id"], out["item_id"]], sort=False)
                  .transform(lambda s: s.shift(1).expanding().mean()))
    out["promo_ratio"] = (promo_mean / out["roll_28"]).clip(0, 5)
    for col, code in [("store_id", "store_code"), ("item_id", "item_code"), ("category", "cat_code")]:
        out[code] = out[col].astype("category").cat.codes
    return out


def seasonal_naive(df: pd.DataFrame) -> pd.Series:
    """Same weekday last week; closed days get zero."""
    return df["lag_7"].fillna(0).where(df["open"] == 1, 0.0)


def moving_average(df: pd.DataFrame) -> pd.Series:
    """Mean of the same weekday over the last four weeks."""
    return df["same_wd_mean_4"].fillna(df["lag_7"]).fillna(0).where(df["open"] == 1, 0.0)


def fit_gbm(train: pd.DataFrame, seed: int = 0) -> HistGradientBoostingRegressor:
    train = train.dropna(subset=["lag_14", "roll_28"])
    train = train[train["open"] == 1]
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=[FEATURES.index(c) for c in ("weekday", "store_code", "item_code", "cat_code")],
        random_state=seed,
    )
    model.fit(train[FEATURES], train["demand"])
    return model


def predict_gbm(model, df: pd.DataFrame) -> pd.Series:
    pred = pd.Series(model.predict(df[FEATURES]), index=df.index)
    return pred.where(df["open"] == 1, 0.0)


def wape(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.abs(actual - pred).sum() / actual.sum())


def bias(actual: pd.Series, pred: pd.Series) -> float:
    return float((pred - actual).sum() / actual.sum())
