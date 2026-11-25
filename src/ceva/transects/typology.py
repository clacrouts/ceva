"""The coastal type of every GCTS transect: the Global Coastal Typology.

Calkoen et al. (2025), "A global 100 m coastal typology derived from satellite data using deep
learning", Zenodo 15607678, CC-BY-4.0. For each transect, a deep-learning model predicts the
sediment type (``shore_type``: sand/gravel, mud, rock platform, none), the geomorphology
(``coastal_type``: cliffed or steep, moderately sloped, bedrock plain, sediment plain, dune,
wetland, inlet, engineered structures), and whether it is built up or defended, each with its
probability. CEVA processes only the coasts whose shoreline moves (``batch.shore_types``) and the
dashboard shows the type. The same files carry the Deltares Shoreline Monitor's satellite-derived
change rate of each transect (``sds_rate``, m/yr), kept to compare with CEVA's.

The continent files are reduced to these columns and sorted by transect id, in
``<data dir>/transects/typology/<record>/<continent>.parquet``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ceva.config import Config
from ceva.transects.gcts import _check_continent, zenodo_file

log = logging.getLogger(__name__)

COLUMNS = {
    "transect_id": "transect_id",
    "class:shore_type": "shore_type",
    "class:coastal_type": "coastal_type",
    "class:is_built_environment": "is_built_environment",
    "class:has_defense": "has_defence",
    "class:prob_sandy_gravel_or_small_boulder_sediments": "p_sandy_gravel",
    "class:prob_muddy_sediments": "p_muddy",
    "class:prob_rocky_shore_platform_or_large_boulders": "p_rocky",
    "class:prob_no_sediment_or_shore_platform": "p_no_sediment",
    "sds:change_rate": "sds_rate",
    "sds:tr_is_qa": "sds_qa",
}
SHORE_TYPES = ("sandy_gravel_or_small_boulder_sediments", "muddy_sediments",
               "rocky_shore_platform_or_large_boulders", "no_sediment_or_shore_platform")


def raw_path(cfg: Config, continent: str) -> Path:
    return cfg.data_dir / "raw" / "typology" / str(cfg.transects.typology.zenodo_record) / f"{continent}.parquet"


def index_path(cfg: Config, continent: str) -> Path:
    return cfg.data_dir / "transects" / "typology" / str(cfg.transects.typology.zenodo_record) / f"{continent}.parquet"


def prepare(cfg: Config, continent: str, force: bool = False) -> Path:
    """Download one continent's typology (checksum verified) and keep the columns CEVA uses."""
    _check_continent(continent)
    dest = index_path(cfg, continent)
    if dest.exists() and not force:
        return dest
    src = raw_path(cfg, continent)
    if force or not src.exists():
        zenodo_file(cfg.transects.typology.zenodo_record, f"{continent}.parquet", src)
    table = pd.read_parquet(src, columns=list(COLUMNS)).rename(columns=COLUMNS)
    table = table.sort_values("transect_id", kind="stable").reset_index(drop=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    table.to_parquet(part, index=False, compression="zstd")
    part.replace(dest)
    log.info("typology of %d transects in %s", len(table), dest)
    return dest


def load(cfg: Config, transect_ids=None) -> pd.DataFrame:
    """The typology of these transects (all prepared continents when ``transect_ids`` is None)."""
    folder = index_path(cfg, "Europe").parent
    paths = sorted(folder.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No coastal typology in {folder}. Run `ceva transects prepare --continent <name>`.")
    filters = [("transect_id", "in", list(map(str, transect_ids)))] if transect_ids is not None else None
    return pd.concat([pd.read_parquet(p, filters=filters) for p in paths], ignore_index=True)
