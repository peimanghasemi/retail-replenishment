"""Daily store replenishment for perishable and ambient items.

Each open day the store receives one delivery in the morning. The order placed in
the evening has to cover demand until the next delivery. Stock is kept in batches
with an expiry day, sold first-in first-out, and units past their expiry are
written off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# Service targets by category from the newsvendor critical ratio
# (margin / (margin + cost of an unsold unit)). Short-life items lose the whole
# purchase price when they expire, so their target is lower.
CATEGORY_SERVICE = {
    "bakery": 0.75,
    "fresh_produce": 0.88,
    "chilled": 0.95,
    "ambient": 0.98,
}


@dataclass
class SimResult:
    demand: float = 0.0
    sold: float = 0.0
    received: float = 0.0
    written_off: float = 0.0
    stockout_days: int = 0
    open_days: int = 0
    inventory_sum: float = 0.0

    def add(self, other: "SimResult") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))


def simulate_series(demand: np.ndarray, forecast: np.ndarray, sigma: np.ndarray,
                    is_open: np.ndarray, shelf_life: int, z: float,
                    initial_stock: float | None = None) -> SimResult:
    """Simulate one store x item over the test horizon.

    forecast[t] and sigma[t] are the one-day-ahead forecast and its standard
    deviation for day t, made in the evening of the previous day.
    """
    n = len(demand)
    res = SimResult()
    # batches as [quantity, last sellable day]
    first = int(np.argmax(is_open))
    start_qty = forecast[first] + z * sigma[first] if initial_stock is None else initial_stock
    batches: list[list[float]] = [[float(np.ceil(start_qty)), first + shelf_life - 1]]

    open_idx = np.flatnonzero(is_open)
    next_open = np.full(n, -1)
    for i in range(len(open_idx) - 1):
        next_open[open_idx[i]] = open_idx[i + 1]

    for t in range(n):
        # remove expired stock at the start of the day
        keep = []
        for q, exp in batches:
            if exp < t:
                res.written_off += q
            else:
                keep.append([q, exp])
        batches = keep

        if not is_open[t]:
            continue

        res.open_days += 1
        d = float(demand[t])
        res.demand += d
        remaining = d
        for b in batches:  # FIFO, oldest first
            take = min(b[0], remaining)
            b[0] -= take
            remaining -= take
            if remaining <= 0:
                break
        batches = [b for b in batches if b[0] > 0]
        res.sold += d - remaining
        if remaining > 0:
            res.stockout_days += 1

        on_hand = sum(q for q, _ in batches)
        res.inventory_sum += on_hand

        # evening order for the next delivery day
        nxt = next_open[t]
        if nxt < 0:
            continue
        usable = sum(q for q, exp in batches if exp >= nxt)
        target = forecast[nxt] + z * sigma[nxt]
        order = max(0.0, np.ceil(target - usable))
        if order > 0:
            batches.append([order, nxt + shelf_life - 1])
            res.received += order
    # stock still on hand at the end is neither sold nor written off
    return res


def z_for(service: float) -> float:
    return float(norm.ppf(service))
