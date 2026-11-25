"""Tide height at each transect for each acquisition.

Uses eo-tides (Bishop-Taylor et al. 2025) on top of pyTMD. The default model,
EOT20, is CC-BY and needs no registration. Heights are in metres above the
model's mean sea level.

Global tide models are built for the open ocean. They are least reliable
inside estuaries and lagoons, which is where results need the most care.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ceva.config import Config

log = logging.getLogger(__name__)

EOT20_SOURCE = "https://doi.org/10.17882/79489"


def model_directory(cfg: Config) -> Path:
    return cfg.data_dir / "tide_models"


def _no_colorama() -> None:
    """Stop eo-tides from calling colorama's ``init()`` on every run.

    Each call wraps ``sys.stdout`` once more when the output is not a terminal (a log file);
    after a few hundred tide runs in one process (a batch), writing to it recursed too deep and
    every unit failed. CEVA prints no colours.
    """
    import eo_tides.utils

    eo_tides.utils.init = _no_op


def _no_op(*args, **kwargs) -> None:
    return None


def tide_heights(cfg: Config, lon: np.ndarray, lat: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """Tide height [time, point] in metres above mean sea level. ``times`` are UTC."""
    from eo_tides.model import model_tides

    _no_colorama()

    directory = model_directory(cfg)
    model = cfg.tides.model
    if not (directory / model).exists() and not any(directory.glob(f"{model}*")):
        raise FileNotFoundError(
            f"Tide model {model} not found in {directory}. For EOT20, download the ocean tide files "
            f"from {EOT20_SOURCE} and unpack them to {directory / 'EOT20' / 'ocean_tides'}."
        )
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    times = pd.DatetimeIndex(times)

    # Model each distinct location once, then map back to the requested points.
    points = np.unique(np.column_stack([lon, lat]).round(6), axis=0)
    frame = model_tides(
        x=points[:, 0], y=points[:, 1], time=times, model=model, directory=str(directory),
        crs="EPSG:4326", mode="one-to-many", output_format="long",
    )
    wide = frame["tide_height"].unstack(["x", "y"])
    wide = wide.reindex(times)
    lookup = {(round(float(x), 6), round(float(y), 6)): i for i, (x, y) in enumerate(wide.columns)}
    order = [lookup[(round(float(x), 6), round(float(y), 6))] for x, y in zip(lon, lat, strict=True)]
    heights = wide.to_numpy(dtype="float32")[:, order]
    if np.isnan(heights).any():
        log.warning("tide model returned %d missing values", int(np.isnan(heights).sum()))
    return heights


class TideTable:
    """Tide heights at fixed points, modelled once per time and kept for the rest of a run.

    Every call to the tide model reads the model's files again (about 15 s for EOT20), however
    few times it is asked for, so a run asks for all the times it will need at once
    (``prefetch``) and looks them up afterwards. A time asked for later is modelled then. The
    height of a time does not depend on the other times modelled with it.
    """

    def __init__(self, cfg: Config, lon: np.ndarray, lat: np.ndarray):
        self.cfg = cfg
        self.lon = np.asarray(lon, dtype="float64")
        self.lat = np.asarray(lat, dtype="float64")
        self._times = pd.DatetimeIndex([])
        self._heights = np.empty((0, len(self.lon)), dtype="float32")

    def prefetch(self, times) -> None:
        """Model the times not yet in the table, in one call."""
        missing = pd.DatetimeIndex(times).unique().difference(self._times)
        if len(missing) == 0:
            return
        heights = tide_heights(self.cfg, self.lon, self.lat, missing)
        times = self._times.append(missing)
        order = np.argsort(times.asi8, kind="stable")
        self._times = times[order]
        self._heights = np.concatenate([self._heights, heights])[order]

    def __call__(self, times) -> np.ndarray:
        """Tide height [time, point], in metres above mean sea level."""
        times = pd.DatetimeIndex(times)
        self.prefetch(times)
        return self._heights[self._times.get_indexer(times)]
