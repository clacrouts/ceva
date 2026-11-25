"""Configuration: packaged worldwide defaults, a region file, then local overrides.

Precedence (last wins): ``default.yaml`` < region file < ``CEVA_DATA_DIR`` <
``overrides`` argument. Values are validated with pydantic, so a typo in a YAML
key fails loudly instead of being silently ignored.
"""
from __future__ import annotations

import copy
import os
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DATA_DIR_ENV = "CEVA_DATA_DIR"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GridConfig(_Strict):
    processing_zoom: int = Field(ge=4, le=12)
    tile_buffer_m: float = Field(ge=0)


class GCTSConfig(_Strict):
    release: str
    zenodo_record: int
    licence: str
    attribution: str


class TypologyConfig(_Strict):
    zenodo_record: int
    licence: str
    attribution: str


class TransectsConfig(_Strict):
    source: Literal["gcts"]
    gcts: GCTSConfig
    typology: TypologyConfig | None = None


class ProviderConfig(_Strict):
    stac_url: str
    s3_endpoint: str | None = None
    m2m_url: str | None = None               # USGS machine-to-machine API
    max_connections: int | None = Field(default=None, ge=1)   # the provider's limit on parallel reads
    gdal: dict[str, str] = Field(default_factory=dict)        # extra GDAL options for its files
    # How units on one tile share the files they all read (see ceva.chips.iter_chips_batch):
    # "cache" reads each unit's window in turn, the blocks they share kept in memory (fewest bytes);
    # "download" copies the file once, whole, and cuts every unit's chips from it (fewest requests, fastest).
    share_files: Literal["cache", "download"] = "cache"


class SensorConfig(_Strict):
    providers: list[str] = Field(min_length=1)
    toa_providers: list[str] = Field(default_factory=list)   # Level-1 (top-of-atmosphere) sources
    platforms: list[str] = Field(min_length=1)
    resolution_m: float = Field(gt=0)
    max_scene_cloud_cover: float = Field(ge=0, le=100)
    collection_categories: list[str] | None = None


class MaskingConfig(_Strict):
    landsat_qa_drop_bits: list[int]
    s2_scl_drop_classes: list[int]


class CubeConfig(_Strict):
    spacing_m: float = Field(gt=0)
    bands: list[str] = Field(min_length=2)
    workers: int = Field(ge=1)


class TidesConfig(_Strict):
    model: str
    reference_elevation_m: float


class EmpiricalWaterlineConfig(_Strict):
    """Extra height of the satellite waterline when waves are above their usual height, fitted to
    surveys: hs_coef * (Hs - mean Hs of the transect) (see ceva.validation.waterline_level). The
    coefficient depends on how the waterline is detected, so there is one per engine."""

    hs_coef: dict[Literal["ceva", "coastsat"], float]
    fitted_on: str = ""


class CdsWavesConfig(_Strict):
    """ERA5 waves from the Copernicus Climate Data Store, stored locally for a box (``ceva.era5``)."""

    area: tuple[float, float, float, float]          # north, west, south, east (degrees)
    variables: list[Literal["hs", "tp", "direction"]] = Field(default_factory=lambda: ["hs"])
    coastal_reach_cells: int = Field(default=3, ge=0)  # sea cells kept: within this many 0.5° cells of land


class WavesConfig(_Strict):
    source: Literal["open_meteo_era5", "cds_era5", "none"]
    component: Literal["none", "empirical"]
    offshore_distance_m: float = Field(ge=0)
    empirical: EmpiricalWaterlineConfig
    cds: CdsWavesConfig | None = None


class AnnualConfig(_Strict):
    threshold: Literal["fixed", "otsu"]
    index_threshold: float
    otsu_window_m: float = Field(gt=0)
    tide_central_fraction: float = Field(gt=0, le=1)
    min_observations: int = Field(ge=1)
    gapfill_years: int = Field(ge=0)
    min_land_run_m: float = Field(ge=0)


class ChipsConfig(_Strict):
    """Small 2D images around the transects, for engines that need image context."""

    margin_m: float = Field(ge=0)
    pixel_m: dict[Literal["landsat", "sentinel2"], float]
    reflectance: dict[Literal["landsat", "sentinel2"], Literal["toa", "surface"]] = Field(
        default_factory=lambda: {"landsat": "surface", "sentinel2": "surface"})
    bands: list[str] = Field(min_length=5)
    workers: int = Field(ge=1)
    segment_km: float = Field(gt=0)
    detect_workers: int = Field(ge=1)
    skip_cloudy: bool = True


class CoastSatConfig(_Strict):
    """Settings of the CoastSat detection (Vos et al. 2019), named as in CoastSat."""

    sand_color: Literal["default", "latest", "dark", "bright"]
    max_dist_ref_m: float = Field(gt=0)
    reference_extend_m: float = Field(ge=0)
    min_beach_area_m2: float = Field(ge=0)
    min_length_sl_m: float = Field(ge=0)
    cloud_thresh: float = Field(ge=0, le=1)
    landsat_cloud_bits: list[int]
    s2_cloud_classes: list[int]
    min_cloud_pixels: int = Field(ge=0)
    cloud_mask_issue: dict[Literal["landsat", "sentinel2"], bool]
    dist_clouds_m: float = Field(ge=0)
    along_dist_m: float = Field(gt=0)
    min_points: int = Field(ge=1)
    max_std_m: float = Field(gt=0)
    max_range_m: float = Field(gt=0)
    min_chainage_m: float
    multiple_inter: Literal["auto", "nan", "max"]
    prc_multiple: float = Field(ge=0, le=1)


class PerImageConfig(_Strict):
    engine: Literal["ceva", "coastsat"]
    threshold: Literal["fixed", "otsu"]
    index_threshold: float
    max_dist_ref_m: float = Field(gt=0)
    tide_stage_bins: int = Field(default=8, ge=2)
    tide_stage_min_obs: int = Field(default=10, ge=1)
    min_valid_fraction: float = Field(ge=0, le=1)
    slope_method: Literal["spectral", "regression"]
    default_slope: float = Field(gt=0)
    slope_bounds: tuple[float, float]
    min_total_water_level_m: float | None = None
    outlier_mad_k: float = Field(gt=0)


class TimeConfig(_Strict):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> TimeConfig:
        if self.end < self.start:
            raise ValueError(f"time.end {self.end} is before time.start {self.start}")
        return self


class Sentinel1Config(_Strict):
    provider: str
    collection: str
    pixel_m: float = Field(gt=0)
    start: date
    nodata: float | None = None
    incidence_deg: float = Field(gt=0, lt=90)   # nominal, for gamma nought -> sigma nought


class SarCalibrationConfig(_Strict):
    """How radar waterlines are aligned on the imagery (see ceva.sar_waterlines)."""

    window_days: float = Field(gt=0)        # imagery positions this close in time give the reference
    min_imagery: int = Field(ge=1)          # fewer imagery positions there: no reference
    min_pairs: int = Field(ge=1)            # per transect and orbit direction; fewer: the unit's value
    terms: list[Literal["hs", "tide"]]      # dependences fitted with the offsets
    per_year: bool = False                  # also remove a unit-wide offset per year
    outlier_mad_k: float = Field(gt=0)


class SarTemplateConfig(_Strict):
    """CEVA solution: radar profile templates learned at the imagery's waterlines."""

    half_width_m: float = Field(gt=0)       # template extent either side of the waterline
    search_m: float = Field(gt=0)           # shifts tried either side of the expected waterline
    step_m: float = Field(gt=0)
    wave_classes: int = Field(ge=1)         # one template per orbit direction and wave class
    min_learn: int = Field(ge=1)            # radar images with an imagery reference, per template
    min_valid: float = Field(gt=0, le=1)    # of the template's samples, for a match
    min_score: float = Field(ge=-1, le=1)   # correlation below this: no waterline


class SarConfig(_Strict):
    enabled: bool
    missions: list[Literal["sentinel1"]]
    workers: int = Field(ge=1)
    block_bytes: int = Field(gt=0)
    detection: Literal["model", "otsu"]     # CoastSat solution: CoastSat's SAR U-Net or its Otsu fallback
    water_threshold: float = Field(gt=0, lt=1)
    smoothing_px: int = Field(ge=1)         # CEVA solution: median filter before sampling the profiles
    template: SarTemplateConfig
    calibration: SarCalibrationConfig
    effective_per_year: int = Field(ge=1)   # independent observations a year, when weighting families
    sentinel1: Sentinel1Config


class RegionConfig(_Strict):
    name: str
    description: str = ""
    bbox_wgs84: tuple[float, float, float, float]
    continent: str

    @field_validator("bbox_wgs84")
    @classmethod
    def _valid_bbox(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        w, s, e, n = v
        # west > east is allowed: it means the box crosses the antimeridian.
        if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s < n <= 90):
            raise ValueError(f"bbox_wgs84 {v} is not lon_min, lat_min, lon_max, lat_max in degrees")
        return v


class StorageConfig(_Strict):
    """Where finished units go online (the bucket and its keys are in the token file ``storage.env``)."""

    prefix: str = "ceva"              # objects under <bucket>/<prefix>/, laid out like the data directory
    region: str = "auto"              # "auto" for Cloudflare R2
    keep_local: bool = False          # keep the local copy once uploaded and checked
    local_budget_gb: float = Field(default=40, ge=0)   # with online storage, most waiting for upload locally
    min_free_gb: float = Field(default=150, ge=0)      # the run pauses while the data disk has less free (shared)
    keep_cubes_above_gb: float = Field(default=300, ge=0)   # local cubes kept only while more than this is free


class BatchConfig(_Strict):
    """Many units in turn (``ceva batch``)."""

    max_transects: int = Field(default=300, ge=1)   # a tile with more is cut along the coast (cube in memory)
    max_consecutive_failures: int = Field(default=5, ge=1)   # then pause (10, 20, 40, 80 min), then stop
    shore_types: list[str] = Field(default_factory=list)     # Global Coastal Typology classes processed (all if empty)
    memory_budget_gb: float = Field(default=30.0, gt=0)      # all workers together (--workers)
    min_free_memory_gb: float = Field(default=16.0, ge=0)    # a worker starts a unit only with this much available
    worker_reads: int = Field(default=8, ge=1)               # parallel image reads of each worker (--workers)


class VulnerabilityConfig(_Strict):
    """Coastal vulnerability index (``ceva.vulnerability``): its sources, and how slopes and tides are measured."""

    # GEBCO 2024 as one cloud-optimised GeoTIFF, cut once over ``area`` (west, south, east, north)
    bathymetry_url: str = ("https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/alexgleith/gebco-2024/"
                           "GEBCO_2024.tif")
    area: tuple[float, float, float, float] = (-34.0, 25.0, 48.0, 69.0)
    slope_half_width_m: float = Field(default=5000.0, gt=0)   # terrain and bathymetry this far either side of the shore
    slope_step_m: float = Field(default=250.0, gt=0)
    gia_record: int = 5560862                     # Zenodo: Steffen (2021), glacial isostatic adjustment grids
    gia_archive: str = "ICEAGE_1Dcomp_grids.zip"
    gia_model: str = "ice6g_vm5a"                 # ICE-6G_C ice history, VM5a earth model: <model>_vup.grd, ...
    sea_level_pattern: str = "*regional_trends*.nc"   # Copernicus Marine OMI_CLIMATE_SL_GLOBAL_regional_trends
    sea_search_deg: float = Field(default=1.0, gt=0)  # nearest sea cell with a trend, within this distance
    tide_year: int = 2020                         # hourly EOT20 tide over this year gives the mean range


class Config(_Strict):
    data_dir: Path
    grid: GridConfig
    transects: TransectsConfig
    providers: dict[str, ProviderConfig]
    sensors: dict[str, SensorConfig]
    masking: MaskingConfig
    cube: CubeConfig
    chips: ChipsConfig
    tides: TidesConfig
    waves: WavesConfig
    annual: AnnualConfig
    perimage: PerImageConfig
    coastsat: CoastSatConfig
    sar: SarConfig
    time: TimeConfig
    storage: StorageConfig = Field(default_factory=StorageConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    vulnerability: VulnerabilityConfig = Field(default_factory=VulnerabilityConfig)
    region: RegionConfig | None = None

    @model_validator(mode="after")
    def _empirical_fitted_for_engine(self) -> Config:
        if self.waves.component == "empirical" and self.perimage.engine not in self.waves.empirical.hs_coef:
            raise ValueError(f"waves.empirical.hs_coef has no coefficient for the {self.perimage.engine!r} engine; "
                             "fit one with `ceva calibrate-waterline`")
        return self

    @model_validator(mode="after")
    def _providers_exist(self) -> Config:
        for family, sensor in self.sensors.items():
            unknown = (set(sensor.providers) | set(sensor.toa_providers)) - set(self.providers)
            if unknown:
                raise ValueError(f"sensors.{family}.providers lists unknown providers {sorted(unknown)}")
        return self


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into a copy of ``base``; lists and scalars are replaced."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def packaged_regions() -> list[str]:
    """Names of the region files shipped with the package."""
    root = resources.files("ceva.config") / "regions"
    return sorted(p.name.removesuffix(".yaml") for p in root.iterdir() if p.name.endswith(".yaml"))


def _region_path(region: str | Path) -> Path:
    candidate = Path(region)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    packaged = resources.files("ceva.config") / "regions" / f"{region}.yaml"
    if packaged.is_file():
        return Path(str(packaged))
    raise FileNotFoundError(
        f"Unknown region {region!r}. Packaged regions: {packaged_regions()}, "
        "or pass a path to a region YAML file."
    )


def load_config(region: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Build the validated configuration, optionally for one region."""
    raw = _read_yaml(Path(str(resources.files("ceva.config") / "default.yaml")))
    if region is not None:
        raw = deep_merge(raw, _read_yaml(_region_path(region)))
    env_dir = os.environ.get(DATA_DIR_ENV)
    if env_dir:
        raw["data_dir"] = env_dir
    if overrides:
        raw = deep_merge(raw, overrides)
    raw["data_dir"] = Path(raw.get("data_dir") or "ceva-data").expanduser().resolve()
    return Config.model_validate(raw)
