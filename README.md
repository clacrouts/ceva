# CEVA: Coastal Erosion and Vulnerability Application

CEVA measures how shorelines move from free satellite imagery (Landsat since 1984, Sentinel-2), corrected
for the tide and the waves, on the transects of the Global Coastal Transect System. The same code runs on
one beach or a continent: the coast is processed tile by tile, each tile in its own projection.

* **Dataset:** satellite-derived shorelines of the sand and gravel coasts of Europe and the Mediterranean,
  1984–2025, on Zenodo ([doi:10.5281/zenodo.24493619](https://doi.org/10.5281/zenodo.24493619)), and the
  satellite images behind them ([doi:10.5281/zenodo.24500960](https://doi.org/10.5281/zenodo.24500960)).
* **Map and figures:** [clacrouts.github.io/ceva-dashboard](https://clacrouts.github.io/ceva-dashboard/)
  ([code](https://github.com/clacrouts/ceva-dashboard)).

## Design in one paragraph

The coast is cut into tiles of about 30 km (quadkeys, zoom 10). Each tile is processed on its own, in
its own metric projection, so the same code runs on one beach or on every coast. Shoreline positions
are measured along the transects of the Global Coastal Transect System, which gives every place on
Earth the same reference lines and stable identifiers. Imagery is discovered through STAC catalogues,
with one small adapter per provider. Several extraction methods ("engines") write the same output
table, so they can be compared against each other and against field surveys.

## How CEVA computes the metrics

Each step in short; the notebook [`notebooks/ceva_step_by_step.ipynb`](notebooks/ceva_step_by_step.ipynb)
runs them one by one on a real beach, with the details of each algorithm, and checks that the result is the
pipeline's.

1. **Transects.** The Global Coastal Transect System gives cross-shore lines 100 m apart; the Global Coastal
   Typology says which are sand or gravel, the only coasts processed by default. A position is a distance
   along a transect from its landward end. *(Notebook step 2.)*
2. **Images along the transects.** Every Landsat 4–9 (30 m) and Sentinel-2 (10 m) surface-reflectance image is
   found through STAC catalogues and read on its own grid at points every 10 m along each transect: green,
   short-wave infrared and the quality flags. The result is a versioned cube (time, transect, distance). *(3)*
3. **Water index and cloud mask.** MNDWI = (green − SWIR) / (green + SWIR), positive over water; samples the
   quality flags reject (cloud, cirrus, and for Landsat shadow and snow) are dropped. *(4)*
4. **Tides.** The EOT20 model gives the tide at each transect when each image was taken. *(5)*
5. **Land and sea.** A transect is measured only if its long-term median index shows land at one end and water
   at the other. *(6)*
6. **Yearly composites.** For each year, the images taken near mean sea level give a median index profile; an
   Otsu threshold near a first guess separates sand from water, and the shoreline is where the profile crosses
   it, to a fraction of a sample (after DEA Coastlines). *(7)*
7. **Tide-stage curves.** All the images of a transect, grouped by tide height, show where the waterline sits
   at each tide; the search on single images follows that curve. *(8)*
8. **One waterline per image.** On each image, the crossing of an Otsu threshold within ±150 m of the expected
   position. *(9)*
9. **Mean sea level.** Each waterline is moved by its water level over the beach slope,
   x + (tide + wave level) / tan β. The wave level is 0.48 m per metre of ERA5 significant wave height above
   the transect's usual height, fitted on six surveyed beaches; the slope is estimated per transect by the
   spectral method of Vos et al. (2020). Outliers beyond 3 median absolute deviations are left out. *(10)*
10. **Yearly positions.** The median of each year's corrected waterlines, with a standard error from their
    scatter and number (a precision, not an accuracy). *(11)*
11. **Rates.** A least-squares line through the yearly positions (at least five years) gives the rate in m/yr
    and its 95 % interval; a trend is significant when the interval excludes zero; classes after Luijendijk
    et al. (2018). *(12)*
12. **Validation.** The same steps on the survey lines of six beaches, and a comparison of every published
    transect with the Shoreline Monitor and EMODnet; see [Validation](#validation). *(14)*
13. **Coastal vulnerability index.** Six variables ranked from 1 (low) to 5 (very high) on the scales of
    Thieler and Hammar-Klose (1999, USGS): the landform (Global Coastal Typology), the rate of step 11, the
    regional slope (GEBCO 2024, from 5 km offshore to the highest ground within 5 km inland), the relative
    sea-level rise (the sea-surface trend from Copernicus Marine altimetry minus the land uplift of glacial
    isostatic adjustment, ICE-6G_C VM5a), the mean ERA5 wave height and the mean EOT20 tidal range. CVI = √(product of the ranks / 6); its
    classes are the quartiles over the transects published together. *(15)*

## Install

```bash
mamba env create -p /path/to/envs/ceva -f environment.yml
conda activate /path/to/envs/ceva
pip install --no-deps -e .
pip install --no-deps "coastsat-package==0.5.1"   # the CoastSat engine (optional)
pytest
```

Python 3.12 is required (Icechunk needs 3.12 or later, CoastSeg needs earlier than 3.13).

## Configure

**Where data goes.** Copy `.env.example` to `.env` and set `CEVA_DATA_DIR` to a directory with room
for downloads and results. `.env` is not committed.

**Secrets** live in token files, one per service, outside the repository in `~/.config/ceva/tokens/`
(or `$CEVA_TOKENS_DIR`), readable by you only. Each file holds only its service's keys. CEVA refuses a
token file that is group- or world-readable, or that sits in a git working tree without being ignored;
`.gitignore` also ignores `*.env`, `tokens/` and `*.token` in case a copy lands in the repository.

```bash
mkdir -p ~/.config/ceva/tokens && chmod 700 ~/.config/ceva/tokens
cd ~/.config/ceva/tokens
printf 'CDSE_S3_ACCESS_KEY=\nCDSE_S3_SECRET_KEY=\n' > cdse.env
printf 'USGS_ERS_USERNAME=\nUSGS_M2M_TOKEN=\n' > usgs.env
printf 'EARTHDATA_TOKEN=\n' > earthdata.env              # NISAR (ASF)
printf 'CDSAPI_URL=https://cds.climate.copernicus.eu/api\nCDSAPI_KEY=\n' > cds.env   # ERA5 waves
chmod 600 *.env                                           # then fill in the values
```

| Credential | Needed for | Where to get it |
|---|---|---|
| `CDSE_S3_ACCESS_KEY`, `CDSE_S3_SECRET_KEY` (`cdse.env`) | Sentinel-2 L1C (top-of-atmosphere) from the Copernicus Data Space, for the CoastSat engine's chips; also a fallback source of Sentinel-2 L2A. Without them the chips fall back to surface reflectance. | Free account, then the [S3 keys manager](https://eodata-s3keysmanager.dataspace.copernicus.eu/). The secret is shown once, when the key is created. |
| `USGS_ERS_USERNAME`, `USGS_M2M_TOKEN` (`usgs.env`) | Landsat Collection 2 Level-1 (top-of-atmosphere) from the USGS. The account also needs the M2M download role. Without it Landsat chips stay on surface reflectance. | Free [ERS account](https://ers.cr.usgs.gov/); an application token, and "Machine to Machine" access requested under Access Request. |
| `CDSAPI_URL`, `CDSAPI_KEY` (`cds.env`) | ERA5 waves from the Copernicus Climate Data Store (`ceva waves download`). Not the same service as the Copernicus Data Space (`cdse.env`). | Free [CDS account](https://cds.climate.copernicus.eu/), the key on its profile page; accept the ERA5 licence once on the [dataset page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences). |
| `EARTHDATA_TOKEN` (`earthdata.env`) | NISAR radar from the Alaska Satellite Facility (not integrated yet). | Free [NASA Earthdata login](https://urs.earthdata.nasa.gov/), then "Generate Token". |
| `PC_SDK_SUBSCRIPTION_KEY` (`planetary_computer.env`) | Optional: a higher rate of Planetary Computer access tokens. Sentinel-1 RTC is read without it. | Not issued to new users (November 2026). |

Surface reflectance (Planetary Computer) and the transects need no account. CDSE allows 4 parallel S3
connections per user, which CEVA respects (`providers.cdse.max_connections`).

`ceva info` shows what is configured, without printing any secret.

## Use

```bash
ceva info                                   # data directory, credentials, providers
ceva transects prepare --region gironde     # download and index the transects of that continent
ceva tiles gironde                          # processing tiles and transect counts
ceva scenes gironde landsat --start 1990-06-01 --end 1990-09-30
ceva process truc_vert                      # read the imagery, write the yearly shoreline product (CEVA solution)
ceva process truc_vert --method coastsat    # the same with CoastSat's detection (comparison only)
ceva process truc_vert lacanau biscarrosse  # several regions: the Level-1C files they share are read once
ceva validate --download                    # run the validation sites and write the report
ceva calibrate-waterline                    # fit the waterline height above the tide to the surveys
```

A whole country or continent runs as a batch of units (a processing tile, cut along the coast into groups of
at most 300 transects), resumable. By default only sand and gravel coasts are processed (`batch.shore_types`,
from the Global Coastal Typology on the same transects; `--shore-types all` for every coast): the six
validation sites are all sandy or gravel beaches, mud is not validated, and rock barely moves.

```bash
ceva transects prepare --continent Europe   # the transects and their coastal typology (also Africa, Asia)
ceva batch plan eu27-sand --countries EU27  # the units and what the run will store
ceva batch plan europe-sand --countries all --area europe-mediterranean --after eu27-sand
                                            # extend a run: the rest of an area, never a transect or unit name twice
ceva batch run eu27-sand                    # in turn (CEVA solution); touch <data dir>/batch/<run>/STOP to stop
taskset -c 11-15 nice -n 19 ceva batch run eu27-sand --workers 5   # tiles side by side on a shared machine (below)
ceva batch status eu27-sand                 # done, failed; days left; bytes stored
ceva keepalive                              # refresh file dates (for disks that delete old files): weekly when idle
ceva references prepare                     # the products CEVA is compared with (Shoreline Monitor, EMODnet)
ceva references sites                       # score the survey beaches from their stored results
ceva vulnerability prepare                  # its sources: GEBCO cut over the area, glacial-rebound grid, sea-level trends
ceva vulnerability compute --run eu27-sand,europe-sand,mediterranean-sand   # the index of every published transect
ceva publish ../ceva-dashboard/data --run eu27-sand   # the dashboard's data (map, series, comparisons, validation)
ceva publish ../ceva-dashboard/data --run eu27-sand,europe-sand,mediterranean-sand --area europe-mediterranean
                                            # several runs as one; and the coast of the area that is not measured
ceva publish ../ceva-dashboard/data --run eu27-sand --only @area.txt   # a preview: only the finished units listed
ceva export <folder> --run eu27-sand,europe-sand,mediterranean-sand --version 1.1     # the full dataset for Zenodo
ceva frames build --run eu27-sand,europe-sand,mediterranean-sand --jobs 10   # the images behind the shorelines
ceva frames pack ../ceva-dashboard/frames        # their archives (one per country) and indexes for the viewer
```

A unit where no transect has land at one end and water at the other (a spit, a lagoon, an island narrower than
a transect) has nothing to measure: it is recorded as empty, not as failed.

`ceva frames` cuts every image CEVA used, 2 km of coast at a time (segments follow the coast, so a curved
coast is measured along its length), into frames aligned with the coast: the sea at the top, 10 m pixels,
±300 m across the shore around the long-term shoreline, wider where the waterlines range further (up to
half a transect). Images come from the scenes the cube was built from, in natural colour with one fixed
stretch. Each segment gets a short H.264 video (image i at i/10 s, keyframes every 10 images; 0.75 KB per
image on the survey beaches, a third less than WebP for the same look) and a JSON file with, for every
image and transect, the waterline detected and the shoreline after the tide and wave correction (drawn by
the viewer, not burned into the images). `frames build --jobs N` builds N units side by side, each in a
fresh process (about one core and up to 2 GB), started only while `batch.min_free_memory_gb` stays free and
never while another process builds that unit; it records each unit built, so a stopped build resumes; a unit
whose cube was deleted finds its scenes in the catalogue, grouped as the cube builder groups them.
`frames pack` puts the segments into one uncompressed ZIP per country (Zenodo allows 100 files per record)
and writes `index.json` (the units with images) and `units/<unit>.json` (where each segment's files start
in its ZIP): the dashboard reads a unit's index when one of its transects is opened, then one segment with
HTTP range requests, from the local server or from Zenodo (`/api/records/<id>/files/<name>/content` allows
ranges from other sites). Needs ffmpeg.

`ceva export` writes the full dataset for download and citation, one Zenodo record: the rates (GeoParquet, the
transects, with the vulnerability index when it has been computed) and the yearly shorelines (GeoParquet,
points) in one file each for the whole area, every waterline (Parquet) in one file per country, a README
(files, columns, method, validation, limits, licence, citation), the deposit's `metadata.json` and
`SHA256SUMS`. Parts are streamed into each file, so memory stays low however large the area. The waterlines
make most of its size: about 7 GB for the sand and gravel coasts of the EU.

`ceva vulnerability compute` ranks every transect of the runs (`ceva.vulnerability`) and stores the index,
its six ranks, the values behind them and the class edges in `<data dir>/vulnerability/index.parquet`;
`publish` and `export` add them when the file is there. The sources sit in `<data dir>/raw/vulnerability`:
`prepare` cuts GEBCO 2024 over `vulnerability.area` from its cloud-optimised GeoTIFF (156 MB for Europe and
the Mediterranean) and takes the rates of glacial isostatic adjustment (land and geoid, `vulnerability.gia_model`)
from the zip of Steffen (2021); the Copernicus Marine product `OMI_CLIMATE_SL_GLOBAL_regional_trends` needs a
free account, so its NetCDF file is downloaded by hand into that folder. Its trends come corrected for glacial
isostatic adjustment: the geoid rate puts that back, as the sea surface at the coast does follow it, before
the land uplift is taken off. Tides and waves are taken once per unit, slopes and sea level per transect;
the file names of the sources and the period of the trends are kept with the index and in the manifest.

`ceva publish` writes what the static site [ceva-dashboard](https://github.com/clacrouts/ceva-dashboard)
reads: one PMTiles file of the transects (latest shoreline position, rate and its 95 % range, change class,
precision, coastal type, the Shoreline Monitor's rate and CEVA's difference with it; made with tippecanoe,
zoom 2 to 11), one JSON file of yearly positions per unit, a summary by country and change class, a manifest
with the attributions, and `stats.json`: the transects grouped by country, change class, coastal type,
material, built-up shore, defences, precision and significance, with their rates, from which the site
computes its filtered figures, its summary by country and its country profiles. With the reference products
prepared, it also writes, per unit, the other products' values at each transect (shown on its card) and
`validation.json`, the data of the dashboard's validation page (see Comparison with other products). With
`--area`, it also draws the coast of that area the dataset does not measure (`coverage.pmtiles`: other coast
types, and sand or gravel without a result; and the same transects as points in `coverage.parquet`, for the
assistant). With `--run`, the site says whether the run is finished: until
it is, the map shows a preview banner with the countries published, and the validation page is marked
provisional. `--only` publishes some of the finished units (names separated by commas, or `@file` with one
per line), for a preview of areas that are complete before the whole run is.

Results stay in the data directory. The disk is shared: the run pauses while it has less than
`storage.min_free_gb` free, and keeps each unit's cube (to reprocess without downloading again) only while
more than `storage.keep_cubes_above_gb` is free; results are always kept. After
`batch.max_consecutive_failures` failures in a row (a service refusing requests) the run pauses 10, 20, 40
and 80 minutes, then stops.

Online storage is optional: any S3-compatible bucket (for example Cloudflare R2: EU jurisdiction, no
download fees). Its endpoint, name and keys go in `~/.config/ceva/tokens/storage.env`:
`STORAGE_ENDPOINT_URL` (`https://<account id>.eu.r2.cloudflarestorage.com`), `STORAGE_BUCKET`,
`STORAGE_ACCESS_KEY_ID`, `STORAGE_SECRET_ACCESS_KEY` (mode 600). A run then uploads every unit, with the
data directory's layout under `storage.prefix` (`cubes/<unit>.icechunk/`, `results/<unit>/<method>/`,
`status/<unit>.json`), opens each cube in the bucket to compare its snapshot, and only then deletes it locally.

A region is a name and a bounding box in `src/ceva/config/regions/`, and an area (for `--area`) a GeoJSON file
in `src/ceva/config/areas/`; any file of the same shape works too. Everything else comes from
`src/ceva/config/default.yaml`, which holds no site-specific value.

`ceva process` writes to `<data dir>/results/<region>/<method>/` the **yearly shoreline product** (imagery
only; with `sar.enabled: true`, also imagery + radar), a GeoParquet table (one row per transect and year:
distance `x` from the landward end, the shoreline point, the number of images behind it, and an estimated
standard error `se_m`) with a rates table beside it:

| Product | Files | Made from |
|---|---|---|
| Imagery only | `shorelines_imagery.parquet`, `rates_imagery.parquet` | the yearly median of the corrected Landsat and Sentinel-2 waterlines |
| Imagery + radar (with `sar.enabled: true`) | `shorelines_imagery_radar.parquet`, `rates_imagery_radar.parquet` | that median and the yearly median of the Sentinel-1 waterlines aligned on the imagery, weighted by their precision; years with too few clear images but enough radar appear here only |

Shorelines are at mean sea level (`tides.reference_elevation_m`), corrected for tide and waves. `se_m` comes
from each source's scatter, the number of observations and the radar's drift between years: an estimate of
precision, not a validated accuracy (see Validation). Alongside are the working tables: `transects`,
`waterlines` (imagery, one row per image and transect), `radar_waterlines`, `annual` (one row per year,
method and sensor set), `rates`, and `run.json`, which records the snapshots, configuration and version
they came from.

Tides need the EOT20 ocean tide files (CC-BY, no account): download them from
[doi:10.17882/79489](https://doi.org/10.17882/79489) and unpack `ocean_tides` into
`<data dir>/tide_models/EOT20/`.

**Run time.** One region of 237 transects (Truc Vert, about 24 km of coast, 1984–2025) on a shared 16-core
machine, radar off: the CEVA solution takes about 4 minutes from an empty data directory (2.5 reading the
imagery along the transects, 1.3 processing). The CoastSat solution also needs image chips (about 5 minutes
for Landsat, then Sentinel-2 Level-1C) and 21 minutes of processing. The Level-1C chips come from the
Copernicus Data Space, whose limits bind: 12 TB per 30 days, 2000 requests a minute, 4 connections of
20 MB/s. Its JPEG 2000 files have no tile index, so reading a window walks the file's tile headers: about
31 MB and 320 requests per acquisition with GDAL's 16 kB requests (set for the whole process in
`ceva.reader`), which the request limit holds to about 6 acquisitions a minute, so 1.5 hours or more for
one region. Regions processed together (`ceva process r1 r2 ...`) share the files they read, as
`providers.<name>.share_files` says: through GDAL's cache (fewest bytes) or downloaded once, whole (fewest
requests, fastest). Every step after the downloads is cached. A batch run of the EU's sand and gravel coasts
(1,852 units, 248,000 transects) takes about 250 worker-hours, 85 % of them reading imagery.

## Questions to the data: MCP server and chat

`ceva.assistant` serves the published dataset to language models. Ten read-only tools answer from the
rates and yearly shorelines of the Zenodo record and the dashboard's files: `about_dataset`,
`transects_near` (a place's transects), `transect_detail`, `change_between` (any period), `area_summary`
(a country, a box, a coastal type, a kind of shore), `rank_hotspots` (fastest erosion or accretion, by
stretch of about 1 km or by transect), `compare_products`, `not_measured` (why there is no data),
`images_for` and `vulnerability`. Every answer is small and rounded, flags trends that are not significant
and imprecise positions, and links to the map and the dataset's DOI.

```bash
pip install -e ".[assistant]"
python -m ceva.assistant --rates <rates.parquet or address> --shorelines <shorelines.parquet or address> \
    --site ../ceva-dashboard                    # or the dashboard's address (the default)
```

It serves the MCP server at `/mcp` (streamable HTTP, for any MCP client) and the dashboard's chat at
`/chat`. The chat sends each question, with the site's language and what the map shows, to a language model
on a free tier (Groq, then Mistral when Groq is busy; keys `GROQ_API_KEY`, `MISTRAL_API_KEY` from the
environment), which calls the tools in the same process: a few questions a minute per visitor, a few rounds
of tools per question, nothing stored. `deploy/huggingface` runs it as a Hugging Face Space; the dashboard
shows its "Ask the data" panel when `index.html` names the service (`<meta name="ceva-chat">`).
`ceva export --products rates,shorelines` writes the two files it reads without the waterlines.

## Comparison with other products

`ceva references prepare` downloads three published products once (`<data dir>/raw/references`) and keeps
what the comparison needs (`<data dir>/references/<key>.parquet`):

| Key | Product | Period | Coverage | Licence |
|---|---|---|---|---|
| `sm` | Shoreline Monitor (Deltares), from the Global Coastal Typology; yearly composite images | 1984–2023 | every GCTS transect | CC-BY-4.0 |
| `emodnet_sat` | EMODnet Geology, coastline migration from satellite data (Deltares) | 2007–2017 | European seas, every 500 m | CC BY-SA 4.0 |
| `emodnet_field` | EMODnet Geology, coastline migration from field data (national surveys, classes only) | varies by source | European seas, coast segments | CC BY-SA 4.0 |

The Shoreline Monitor covers every transect and the EMODnet products the European seas, so a comparison
means the same in every country it covers. National indicators that cover a single country are not used.

`ceva publish` matches each product to each transect (`ceva.validation.references`): the Shoreline Monitor
by transect id; the others by the nearest feature to the transect within 300 m of CEVA's latest shoreline
(a product's coastline can sit tens of metres from the waterline), within 150 m (EMODnet satellite, each
of its transects counted once) or 100 m (EMODnet field). CEVA's rate is refitted on each product's years,
from 1984 at the earliest, with at least five yearly positions, half the years of the period and positions
near both ends. Classes: each product's own (stable within ±0.5 m/yr); CEVA calls a coast eroding or
accreting only where its trend is significant and beyond that band. Shoreline Monitor transects failing its
own quality flag are left out. `ceva.validation.agreement` turns the matches into the numbers and charts of
the validation page: median and typical (median absolute) difference, share within ±0.5 m/yr, correlation,
class agreement and kappa, by country, coastal type, built-up or defended coast and CEVA's precision, and the
accuracy at the survey beaches (`ceva references sites`, from stored results) next to the SDS benchmark's
algorithms.

## Validation

`ceva validate` runs the pipeline on survey lines at six sites with both shoreline detection methods, and
writes one interactive page (`<data dir>/reports/validation.html`). It also scores the published algorithms of
the SDS benchmark (Vos et al. 2023) against the same surveys, transects and time matching, so CEVA can be
compared with the state of the art like for like. `--skip-run` rewrites the page from stored results;
`--detail coastsat|ceva` chooses the method shown in the detailed charts.

| Site | Surveys | Tests |
|---|---|---|
| Truc Vert, France | Topography every 2 to 4 weeks, 2003 to 2019 (Castelle et al. 2020) | Single-image and yearly positions; benchmark |
| Narrabeen–Collaroy, Australia | Monthly profiles on 5 lines since 1976 (Turner et al. 2016) | Single-image and yearly positions; benchmark |
| Noord-Holland, the Netherlands | One dated profile per year, 101 lines (JARKUS, Rijkswaterstaat) | Yearly and survey-date positions, 40-year rates |
| Walcheren, the Netherlands | One dated profile per year, 56 lines (JARKUS); spring tide range about 4 m | Yearly and survey-date positions, 40-year rates |
| Perranporth, England | Profiles about monthly, 2006–2019 (Konstantinou et al. 2023, CC-BY); macrotidal (6.3 m), flat | Single-image and yearly positions |
| Slapton Sands, England | Profiles about monthly, 2007–2020 (same source); steep gravel barrier, macrotidal (4.3 m) | Single-image and yearly positions |

Two shoreline detection methods are kept and compared (`perimage.engine`, and the variant of the same name):

* **CEVA solution** (`ceva`, the default and the production method): MNDWI along each transect with an Otsu
  threshold near the expected shoreline.
* **CoastSat solution** (`coastsat`, kept for comparison at the validation sites): CoastSat's pixel classifier,
  sand/water threshold and 2D contour on image chips, Sentinel-2 read as top-of-atmosphere reflectance.

Both share everything else: the search window that follows the tide, the tide + wave correction fitted to
surveys (one coefficient per method), the spectral beach slope, and the yearly composites.

Single images (all sensors) and yearly positions against surveys, RMSE in metres (satellite minus survey):

| | CoastSat solution | CEVA solution |
|---|---|---|
| Truc Vert, single images / yearly | 25.8 / 14.5 | 23.6 / 13.5 |
| Narrabeen, single images / yearly | 9.2 / 8.0 | 9.7 / 8.5 |
| Slapton Sands, single images / yearly | 12.2 / 6.8 | 11.6 / 8.9 |
| Perranporth, single images / yearly | 49 / 43 | 45 / 40 |
| Noord-Holland, yearly / rates (m/yr) | 17.8 / 0.35 | 17.1 / 0.38 |
| Walcheren, yearly / rates (m/yr) | 24.3 / 0.53 | 23.4 / 0.44 |

The CoastSat solution maps the most images (Slapton Sands: 1099 matched to surveys, against 262), and has the
lower yearly error at Narrabeen and Slapton Sands; the CEVA solution has the lower yearly error at the other four
sites.

**Search window that follows the tide.** On Perranporth's 0.014 slope the tide moves the waterline about
±260 m, beyond a ±150 m window fixed on the annual shoreline, so a fixed window does not follow the tide.
The images of the whole record are grouped by tide height (eight groups); each group's composite gives where
the waterline sits at that tide, and each image's window is moved along that curve from the year's shoreline.
At Perranporth this brings the error from 153–161 m (fixed window) to 45–49 m, with a slope estimated at 0.014
(the fixed window gave 0.15–0.25); what is left is mostly a mean offset (satellite 30–37 m landward). Elsewhere
it changes the error by less than 1 m.

On the four SDS benchmark transects at Truc Vert and five at Narrabeen, published CoastSat (Landsat) scores
18.4 m (bias −0.6) and 8.5 m (bias −3.4).

**CoastSat's published positions.** They sit half a pixel north-west of where their pixels put them: CoastSat
converts contour coordinates, counted from pixel centres, with the GeoTIFF transform, counted from the pixel
corner. That is 8.4 m seaward at the west-facing Truc Vert and 7.6 m landward at the east-facing Narrabeen,
for Landsat. With it removed, CoastSat's bias is −8.9 m and +4.2 m: the same site dependence as CEVA's.

**Sentinel-2 cloud mask.** Only high-probability cloud and thin cirrus are masked, as CoastSat does; Sen2Cor
gives shadow, medium-probability cloud and snow to dry sand and white water. Compared with masking those
classes too, usable Sentinel-2 image–survey pairs go from 295 to 916 at Truc Vert and from 176 to 296 at
Narrabeen, with equal or better accuracy. The same relaxation for Landsat made Truc Vert worse, so Landsat
keeps its mask.

**Where the waterline sits** (`ceva calibrate-waterline`). For each waterline matched to a survey, the height
above the tide that puts it on the surveyed shoreline with CEVA's slope. Within each beach it rises by a similar
amount with the wave height, so the CoastSat solution adds 0.37 m (the CEVA solution: 0.48 m) × (wave height − the
transect's usual wave height), fitted on all six beaches. Fitted on five, it improves the sixth in 12 of 12 cases
(six beaches, two detection methods). Without Perranporth CoastSat's value would be 0.50, which helps Truc Vert
(about 1.5 m less single-image error than 0.37) but makes Slapton Sands and Noord-Holland worse than no wave
correction. Each beach keeps a mean offset: Truc Vert −12 m, Narrabeen +6, Noord-Holland −7, Walcheren +4,
Slapton Sands +2, Perranporth −28 (CoastSat detection). Beach slope and the Iribarren number predict a left-out
beach's offset at 4 of 6 beaches (mean error 9.3 → 7.2 m), mostly through Perranporth; six beaches are too few to
adopt such a model. Duck and Torrey Pines, the other benchmark sites, are not included: their surveys could not
be retrieved (the Duck survey server did not answer; the Torrey Pines data on Dryad need an API token).

**Top-of-atmosphere reflectance.** CoastSat's classifier was trained on top-of-atmosphere images, so the
CoastSat engine reads Sentinel-2 Level-1C (from the Copernicus Data Space), with the cloud mask of the Level-2A
image of the same pass and tile. On surface reflectance Sentinel-2 sat seaward of Landsat at every site; on top
of atmosphere the two agree (gap between their biases: Narrabeen 3.9 → 1.9 m, Slapton Sands 9.5 → 1.4, Truc Vert
6.5 → 3.2, Perranporth 9.4 → 0.2). Sentinel-2 single images: Narrabeen 13.3 → 7.9 m, Slapton Sands 17.6 → 10.7,
Truc Vert 27.2 → 26.5; Perranporth 43.3 → 47.8, where Landsat sits landward of the surveys and Sentinel-2 then
follows it. Landsat stays on surface reflectance: reading Level-1 needs a USGS account with the M2M download
role. The CEVA solution gains little (Narrabeen 9.4 → 8.8 m on 100 test images) and stays on surface reflectance.

Where top-of-atmosphere (Level-1) data can be read, checked in November 2026:

| Mission | Source | Account | Cost | Notes |
|---|---|---|---|---|
| Sentinel-2 L1C, 2015– | **Copernicus Data Space** (`sentinel-2-l1c`, STAC + S3): **used by CEVA** | free (S3 keys) | free | reprocessed to baseline 05.xx; JPEG 2000; 4 connections per user, 12 TB per 30 days; no L1C for about 5 % of the 2015–2016 images |
| Sentinel-2 L1C, 2015– | Google Cloud public bucket `gcp-public-data-sentinel-2` | none | free (Google pays the hosting) | JPEG 2000 in SAFE folders, current to the day before; no published limits, about 118 MB/s on one connection; no STAC, listed through the Cloud Storage API; mixed processing baselines (the original 02.xx–04.xx products, part of the 05.xx reprocessing) |
| Landsat 4–9 Collection 2 Level-1, 1982– | **USGS machine-to-machine API**, searched with the USGS STAC server: **chosen for CEVA** | free USGS ERS account with the M2M download role | free | downloads need the M2M download role (HTTP 403 without it), requested on the ERS site |
| Landsat 4–9 Collection 2 Level-1 | USGS LandsatLook (HTTPS) | free USGS ERS account | free | browser login only: an M2M token is redirected to the login page |
| Landsat 4–9 Collection 2 Level-1 | USGS, `s3://usgs-landsat` (us-west-2) | AWS account | requester pays: transfer out of AWS, small per site | cloud-optimised GeoTIFF with the 15 m panchromatic band CoastSat sharpens with |

Not usable for it: Planetary Computer (Level-1 only for Landsat 1–5 MSS), the Copernicus Data Space's Landsat
collection (one image over Truc Vert in total), Earth Engine (commercial use needs a paid licence). Licences
allow the use: Sentinel data are free, full and open under the Copernicus legal notice (derived products must
state "Contains modified Copernicus Sentinel data [year]"); Landsat data are in the US public domain.

**Radar (Sentinel-1).** Off by default (`sar.enabled: false`); the method and its results follow.
Radar sees through clouds, so it adds waterlines in cloudy seasons and places.
CEVA reads Sentinel-1 terrain-corrected backscatter (VV, VH, 10 m, 2014 onwards) from the Planetary
Computer (`sentinel-1-rtc`; no account needed), as chips like the imagery (`<unit>-sar.icechunk`). Radar
waterlines are detected separately, then aligned on the imagery, which stays the reference:

* **Detection, CEVA solution:** what a waterline looks like in radar is learned from the imagery. For each
  transect, orbit direction and wave class, the median radar profile across the imagery's waterline is a
  template (typically a dark band of wet sand and swash just landward, bright breaking waves just seaward);
  every radar profile is matched to it within 100 m of the expected waterline.
* **Detection, CoastSat solution:** CoastSat's radar land/water model (U-Net trained on Sentinel-1 over US
  coasts), on the chips brought to sigma nought at a nominal incidence angle.
* **Alignment (inter-mission calibration):** each radar waterline is compared with the median of the
  imagery positions within 30 days; the difference is modelled per transect and orbit direction, plus
  terms in wave height and tide, fitted on imagery only (never the surveys), and removed from every radar
  waterline, including those far from any image.
* **Yearly positions "imagery + radar":** the imagery and radar yearly medians weighted by their
  precision (each family's spread of single positions, with at most 12 independent observations a year).

Radar against the surveys, 2015 onwards (CEVA solution; RMSE in metres, rates in m/yr):

| | Radar single images, as detected → aligned (imagery) | Yearly: imagery → imagery + radar | Rates: imagery → imagery + radar | Images a year: imagery / radar |
|---|---|---|---|---|
| Truc Vert | 41.7 → 30.1 (24.5) | 14.4 → 14.2 | 2.29 → 2.27 | 46 / 177 |
| Narrabeen | 19.0 → 14.7 (9.4) | 7.7 → 7.5 | 2.04 → 2.12 | 75 / 80 |
| Slapton Sands | 21.5 → 13.2 (11.9) | 10.1 → 10.1 | 0.87 → 0.80 | 30 / 191 |
| Perranporth | 61.0 → 51.2 (46.3) | 38.8 → 37.9 | 5.72 → 6.06 | 47 / 127 |
| Walcheren | – | 20.1 → 19.8 | 2.12 → 2.14 | 52 / 118 |
| Noord-Holland | – | 15.9 → 15.7 | 1.57 → 1.55 | 32 / 176 |

A single radar waterline is less precise than an optical one, but there are 3 to 6 times more, in any
weather, and the yearly positions improve or hold at every site. A threshold on the backscatter does not
work on these beaches (78 m at Truc Vert): just landward of the waterline wet sand and swash are dark, just
seaward breaking waves are bright, and a threshold latches onto either edge, where the learned template does not.
CoastSat's radar model, trained on US beaches with Earth Engine's processing, does not transfer (78 m at
Truc Vert, 45 m at Narrabeen once aligned) and gets almost no weight. Radar also drifts from the imagery
from year to year (4–14 m landward in 2022–2023, when Sentinel-1A flew alone; at Walcheren, surveyed to
2025, the imagery was right), which the weighting accounts for; removing an offset per year
(`sar.calibration.per_year`) made three sites worse and is off.

Sources of radar, checked in November 2026:

| Mission | Source | Account | Notes |
|---|---|---|---|
| Sentinel-1, 2014– | **Planetary Computer `sentinel-1-rtc`: used by CEVA** | none (documented as needing one; keys not issued any more) | gamma nought, terrain corrected (Catalyst), 10 m UTM, CC BY 4.0 |
| Sentinel-1, 2014– | Planetary Computer `sentinel-1-grd`; Copernicus Data Space GRD | none; free account | the fallback: calibration and map projection to do (e.g. `sarsen`, Apache-2.0) |
| Sentinel-1 | ASF OPERA RTC-S1 | free NASA Earthdata login | global, but 30 m |
| NISAR (L band), June 2026– | ASF, calibrated "provisional" products | free NASA Earthdata login (`earthdata.env`) | not integrated: 10–20 m, public domain; L band is less disturbed by wind on the water |
| Sentinel-1 | Google Earth Engine | commercial use needs a paid licence | not used |

## Features

* Worldwide tile grid, each tile in its own metric projection; transects from the Global Coastal Transect
  System; coastal type from the Global Coastal Typology (sand and gravel processed by default).
* Scene search over Landsat 4–9 and Sentinel-2 across STAC providers (Planetary Computer, Earth Search,
  Copernicus Data Space); pixels read along the transects on each image's own grid; a versioned data cube
  (Icechunk).
* Tides from EOT20 (through eo-tides); waves from ERA5 (Climate Data Store for large runs, Open-Meteo for
  single sites); the height of the waterline above the tide fitted to surveys.
* Two detection methods on the same cube, CEVA (production) and CoastSat (comparison), both with the search
  window that follows the tide, the spectral beach slope, yearly composites and per-image waterlines, and
  rates with their 95 % interval.
* The CoastSat engine without Google Earth Engine: chips of coastal segments of 3 km, detection in parallel
  processes, reproducible (seeded) and resumable per segment; acquisitions CoastSat would skip as too cloudy
  are not downloaded (their quality band is read first); CoastSat's three slowest steps replaced by faster
  versions that give the same results.
* Radar (Sentinel-1) waterlines aligned on the imagery, off by default.
* Coastal vulnerability index (USGS method) of every measured transect, with its six ranks and inputs.
* Validation at six survey sites with the SDS benchmark; comparison with the Shoreline Monitor and EMODnet.
* Batch runs over countries or areas: resumable, several workers within a memory budget on a shared
  machine, runs that extend earlier ones (`--after`), optional online storage.
* Outputs: the Zenodo dataset (`ceva export`), the dashboard's data (`ceva publish`) and the image archives
  of its viewer (`ceva frames`).

## Limits

* Only sand and gravel coasts are measured: the method is validated on six sandy or gravel beaches, all in
  temperate climates. Mud (marshes, tidal flats) and rock are not measured.
* Transects with water at both ends or land at both ends (spits, lagoons, islands narrower than a transect)
  get no result.
* The CoastSat engine reads Landsat as surface reflectance unless the USGS account has the download role.
* Landsat 5, 7, 8, 9 and Sentinel-2 are not cross-calibrated against each other beyond the shared detection.
* Each beach's mean offset of the waterline is not modelled: six survey sites are too few to fit it.
* Global tide models are least reliable inside estuaries and lagoons.
* The vulnerability index ranks coasts against each other, as its classes are quartiles. Its inputs are
  regional: GEBCO's grid is about 460 m, the sea-level trends 0.25°, tides and waves one value per unit;
  the vertical land motion is glacial rebound only (not the subsidence of deltas or from pumping).

## Layout

```
src/ceva/
  config/              defaults, region files and areas
  catalog/             STAC providers and the common scene description
  transects/           Global Coastal Transect System, coastal typology, sampling along transects
  grid.py              tiles and projections
  reader.py            satellite bands at points, on each image's own grid
  cube.py              the transect cube and its versioned storage
  chips.py             small 2D images around the transects, for the CoastSat engine
  sar.py               Sentinel-1 radar chips
  tides.py             tide height per transect and image
  waves.py, era5.py    wave level at the shore; the local ERA5 wave store
  shoreline.py         water index and sub-pixel shoreline detection
  engines.py           yearly composites, per-image waterlines, slopes, rates
  engines_coastsat.py  per-image waterlines with CoastSat's detection
  sar_waterlines.py    radar waterlines, aligned on the imagery
  pipeline.py          one unit: cube, tides, waves, engines, result tables
  products.py          the yearly shoreline products
  batch.py             many units: plan, run, resume, several workers
  storage.py           optional online storage (S3-compatible)
  publish.py           the dashboard's data
  export.py            the Zenodo dataset
  frames.py            the images behind the shorelines, for the dashboard's viewer
  validation/          survey sites, benchmark, report; reference products and their agreement
  vulnerability.py     coastal vulnerability index
  assistant/           questions to the published dataset: MCP server, the dashboard's chat
  change_metrics.py    change metrics (DSAS-style)
  cli.py               the `ceva` command
notebooks/             ceva_step_by_step.ipynb: the processing of one beach, step by step
deploy/huggingface/    the chat and MCP server as a Hugging Face Space
tests/                 offline unit tests
```

## Data sources and attribution

| Data | Source | Licence |
|---|---|---|
| Transects | Global Coastal Transect System, Calkoen et al. (2025), Deltares / TU Delft, [doi:10.5281/zenodo.14056925](https://doi.org/10.5281/zenodo.14056925); derived from OpenStreetMap coastlines | CC-BY-4.0 |
| Coastal type of each transect (sediment, geomorphology, built-up, defences); Shoreline Monitor change rates | Global Coastal Typology, Calkoen et al. (2025), Deltares / TU Delft, [doi:10.5281/zenodo.15607678](https://doi.org/10.5281/zenodo.15607678) | CC-BY-4.0 |
| Landsat Collection 2 Level-2 | USGS, served by Microsoft Planetary Computer | Public domain |
| Tides | EOT20, Hart-Davis et al. (2021), DGFI-TUM, [doi:10.17882/79489](https://doi.org/10.17882/79489) | CC-BY-4.0 |
| Waves | ERA5 ocean waves, Hersbach et al. (2020), Copernicus Climate Change Service: from the Climate Data Store (continental runs) or through the Open-Meteo marine API (single sites) | ERA5: Copernicus licence; Open-Meteo: CC-BY-4.0, free API for non-commercial use |
| Sentinel-2 L2A and L1C | Copernicus (ESA / EU), served by Planetary Computer (L2A), the Copernicus Data Space (L1C, and L2A as a fallback) or Earth Search | Copernicus Sentinel data terms (free, full and open); products state "Contains modified Copernicus Sentinel data [year]" |
| Landsat Collection 2 Level-1 | USGS EROS, through the machine-to-machine API (ERS account) | Public domain |
| Shoreline detection (CoastSat engine) | CoastSat, Vos et al. (2019), [doi:10.1016/j.envsoft.2019.104528](https://doi.org/10.1016/j.envsoft.2019.104528), through coastsat-package (UNSW Water Research Laboratory, USGS CoastSeg team); its trained classifiers | GPL-3.0-only |
| Sentinel-1 radar (gamma nought, terrain corrected) | Copernicus (ESA / EU) Sentinel-1 GRD, terrain corrected by Catalyst, served by Microsoft Planetary Computer (`sentinel-1-rtc`) | CC-BY-4.0 (the RTC product); Copernicus Sentinel data terms; products state "Contains modified Copernicus Sentinel data [year]" |
| Reference products for the comparison (not used to make CEVA's results): Shoreline Monitor rates (Global Coastal Typology, above); EMODnet Geology coastline migration (satellite and field data), [EMODnet](https://emodnet.ec.europa.eu/geonetwork/srv/api/records/8fbb18deb12b8ea73a91e8a1141e4ea395142ac5) | CC-BY-4.0; CC BY-SA 4.0 (their values on the dashboard's cards stay under it) |
| Terrain and bathymetry (regional slope of the vulnerability index) | GEBCO Compilation Group (2024), GEBCO 2024 Grid, [doi:10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f](https://doi.org/10.5285/1c44ce99-0a0d-5f4f-e063-7086abc0ea0f), read from its cloud-optimised copy on Source Cooperative | Public domain, attribution requested |
| Sea-level trends (vulnerability index) | Copernicus Marine Service, `OMI_CLIMATE_SL_GLOBAL_regional_trends` (satellite altimetry, DUACS DT2024; 1999–2026 in the version used) | Copernicus Marine licence (free, attribution: "Generated using E.U. Copernicus Marine Service Information") |
| Glacial isostatic adjustment (vulnerability index) | ICE-6G_C (VM5a), Peltier et al. (2015), [doi:10.1002/2014JB011176](https://doi.org/10.1002/2014JB011176); grids of Steffen (2021), [doi:10.5281/zenodo.5560862](https://doi.org/10.5281/zenodo.5560862) | CC-BY-4.0 (grids) |
| Radar land/water model (CoastSat solution) | coastsat-package's SAR model (U-Net, ResNet-50), [huggingface.co/2320sharon/SAR_3_band_model](https://huggingface.co/2320sharon/SAR_3_band_model), downloaded once and checked against its hash | CC0-1.0 (weights) |

Only data that may be redistributed in derived products, including commercially, is used by the pipeline.
The validation surveys (Truc Vert, Narrabeen, JARKUS) and the benchmark submissions are used for comparison
only; JARKUS is free for research with acknowledgement of Rijkswaterstaat.

## Licence

GPL-3.0-only, see [LICENSE](LICENSE). The CoastSat engine calls CoastSat (coastsat-package), which is
under the same licence.
