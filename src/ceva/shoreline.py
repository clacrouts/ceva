"""From a cross-shore profile of a water index to a shoreline position.

All functions work on arrays whose last axis is the cross-shore distance, with
every profile running from land to sea (see ``orient_seaward``). Positions are
found by linear interpolation between samples, so they are finer than the
pixel size.
"""
from __future__ import annotations

import numpy as np


def mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified Normalised Difference Water Index (Xu 2006), NaN where undefined.

    Slightly negative reflectances (over-corrected dark water) are set to zero first.
    """
    g = np.maximum(green, 0.0)
    s = np.maximum(swir1, 0.0)
    total = g + s
    with np.errstate(invalid="ignore", divide="ignore"):
        index = (g - s) / total
    return np.where(total > 0, index, np.nan).astype("float32")


def clear_sky(qa: np.ndarray, is_landsat: np.ndarray, landsat_drop_bits: list[int],
              s2_drop_classes: list[int]) -> np.ndarray:
    """True where the provider's flags call the sample usable. ``is_landsat`` is per time step."""
    drop_mask = sum(1 << bit for bit in landsat_drop_bits)
    landsat_ok = (qa & drop_mask) == 0
    s2_ok = ~np.isin(qa, s2_drop_classes)
    return np.where(is_landsat[:, None, None], landsat_ok, s2_ok)


class NoShoreline(ValueError):
    """Nothing to measure on a unit: no transect has land at one end and water at the other (a spit,
    a lagoon, an island narrower than a transect), or no year has a position. Not a failure."""


def orient_seaward(typical: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """For each profile of a long-term median index, +1 if distance increases
    seaward, -1 if it decreases, 0 if the two ends do not differ (both land or
    both water). ``typical`` is [transect, distance]."""
    n, k = typical.shape
    sign = np.zeros(n, dtype="int8")
    for i in range(n):
        valid = np.flatnonzero(np.isfinite(typical[i]))
        if valid.size < 8:
            continue
        quarter = max(valid.size // 4, 2)
        first = np.nanmedian(typical[i, valid[:quarter]])
        last = np.nanmedian(typical[i, valid[-quarter:]])
        if last > threshold >= first:
            sign[i] = 1
        elif first > threshold >= last:
            sign[i] = -1
    return sign


def _interpolate(index: np.ndarray, distance: np.ndarray, k: np.ndarray, threshold: float) -> np.ndarray:
    """Position of the threshold between samples ``k`` and ``k + 1`` (k < 0 gives NaN)."""
    safe = np.clip(k, 0, index.shape[-1] - 2)
    a = np.take_along_axis(index, safe[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(index, safe[..., None] + 1, axis=-1)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = (threshold - a) / (b - a)
    position = distance[safe] + fraction * (distance[safe + 1] - distance[safe])
    return np.where((k >= 0) & np.isfinite(fraction), position, np.nan)


def seaward_edge(index: np.ndarray, distance: np.ndarray, threshold: float, min_land_run_m: float,
                 leading_gap_is_land: bool = False) -> np.ndarray:
    """Seaward edge of the most seaward stretch of land at least ``min_land_run_m`` long.

    This ignores small patches of land offshore (bar crests, vessels, noise) and
    requires water seaward of the edge. Gaps in the profile inherit the state of
    the sample landward of them; a gap at the landward end counts as land only if
    ``leading_gap_is_land`` (true for beach surveys, which start on the dune).
    Returns NaN where no such edge exists.
    """
    spacing = float(distance[1] - distance[0])
    min_run = max(int(round(min_land_run_m / spacing)), 1)
    finite = np.isfinite(index)
    land = np.zeros(index.shape, dtype=bool)
    run = np.zeros(index.shape[:-1], dtype="int32")
    last_edge = np.full(index.shape[:-1], -1, dtype="int64")
    state = np.full(index.shape[:-1], leading_gap_is_land)   # last known state, True = land
    for k in range(index.shape[-1]):
        state = np.where(finite[..., k], index[..., k] <= threshold, state)
        land[..., k] = state
        run = np.where(state, run + 1, 0)
        if k + 1 < index.shape[-1]:
            is_edge = (state & finite[..., k] & finite[..., k + 1] & (index[..., k + 1] > threshold)
                       & (run >= min(min_run, k + 1)))
            last_edge = np.where(is_edge, k, last_edge)
    # Land seaward of the chosen edge (a long emerged bar, or a profile ending on land) invalidates it.
    position = _interpolate(index, distance, last_edge, threshold)
    ks = np.arange(index.shape[-1])
    seaward_land = (land & (ks > last_edge[..., None])).sum(axis=-1)
    return np.where(seaward_land >= min_run, np.nan, position)


def nearest_crossing(index: np.ndarray, distance: np.ndarray, threshold, reference: np.ndarray,
                     max_dist_m: float, min_valid_fraction: float = 0.0) -> np.ndarray:
    """Land-to-water crossing closest to ``reference``, within ``max_dist_m``.

    ``threshold`` is a scalar or an array broadcastable to ``index.shape[:-1]``.
    Observations with too few valid samples inside the search window are rejected.
    """
    thr = np.asarray(threshold, dtype="float32")[..., None] if np.ndim(threshold) else threshold
    a, b = index[..., :-1], index[..., 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        crossing = (a <= thr) & (b > thr)
        position = distance[:-1] + (thr - a) / (b - a) * np.diff(distance)
    offset = np.where(crossing, np.abs(position - reference[..., None]), np.inf)
    best = np.argmin(offset, axis=-1)
    best_offset = np.take_along_axis(offset, best[..., None], axis=-1)[..., 0]
    found = np.take_along_axis(position, best[..., None], axis=-1)[..., 0]
    result = np.where(best_offset <= max_dist_m, found, np.nan)
    if min_valid_fraction > 0:
        window = np.abs(distance - reference[..., None]) <= max_dist_m
        with np.errstate(invalid="ignore"):
            fraction = (np.isfinite(index) & window).sum(axis=-1) / np.maximum(window.sum(axis=-1), 1)
        result = np.where(fraction >= min_valid_fraction, result, np.nan)
    return np.where(np.isfinite(reference), result, np.nan)
