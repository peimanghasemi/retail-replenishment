"""Forecast error model used for safety stock.

Daily retail demand is overdispersed, and the error grows faster than linearly with
the forecast on promotion days. The variance of the forecast error is modelled per
category as  var = a * f + b * f^2  and fitted on a calibration window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_error_model(df: pd.DataFrame, pred_col: str) -> dict[str, tuple[float, float]]:
    params = {}
    open_days = df[df["open"] == 1]
    for cat, g in open_days.groupby("category"):
        f = g[pred_col].to_numpy()
        r2 = (g["demand"].to_numpy() - f) ** 2
        X = np.column_stack([f, f ** 2])
        coef, *_ = np.linalg.lstsq(X, r2, rcond=None)
        a, b = np.maximum(coef, [0.5, 0.0])
        params[cat] = (float(a), float(b))
    return params


def sigma(pred: np.ndarray, category: str, params: dict[str, tuple[float, float]]) -> np.ndarray:
    a, b = params[category]
    return np.sqrt(a * pred + b * pred ** 2)
