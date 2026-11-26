"""Allocation of a limited promotion quantity across stores.

When the distribution centre has less stock of a promotion item than the stores
would order, the quantity has to be split. Splitting in proportion to the forecast
is common, but it ignores that a unit placed in a store with low and uncertain
demand is less likely to sell. The greedy rule below gives each next unit to the
store where it has the highest probability of being sold, P(D_s >= current + 1),
which maximises expected sales for independent store demands.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.stats import nbinom


def _nb_params(mean: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.maximum(mean, 1e-6)
    p = k / (k + mean)
    return np.full_like(mean, k), p


def greedy(supply: int, mean: np.ndarray, k: float = 8.0) -> np.ndarray:
    n, p = _nb_params(np.asarray(mean, float), k)
    alloc = np.zeros(len(mean), dtype=int)
    # P(D >= 1) for every store
    heap = [(-nbinom.sf(0, n[s], p[s]), s) for s in range(len(mean))]
    heapq.heapify(heap)
    for _ in range(int(supply)):
        neg_gain, s = heapq.heappop(heap)
        alloc[s] += 1
        heapq.heappush(heap, (-nbinom.sf(alloc[s], n[s], p[s]), s))
    return alloc


def proportional(supply: int, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    raw = supply * w
    alloc = np.floor(raw).astype(int)
    # hand out the rounding remainder by largest fractional part
    rest = int(supply - alloc.sum())
    if rest > 0:
        alloc[np.argsort(-(raw - alloc))[:rest]] += 1
    return alloc


def sold(alloc: np.ndarray, demand: np.ndarray) -> int:
    return int(np.minimum(alloc, demand).sum())
