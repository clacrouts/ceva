"""Landsat Collection 2 Level-1 (top-of-atmosphere) from the USGS, the only source of the full archive.

Scenes are found with the USGS STAC server (no account). The files themselves are behind
the USGS EROS Registration System (ERS): LandsatLook sends anonymous and token requests to
a browser login, so scripts must go through the machine-to-machine (M2M) API, logging in
with an ERS user name and application token (``USGS_ERS_USERNAME``, ``USGS_M2M_TOKEN``). The
M2M API only hands out files to accounts that USGS has given the "download" role, which is
requested separately on https://ers.cr.usgs.gov/ (Access Request > Machine to Machine).

Login, the role check and search are implemented; reading the band files through M2M download
links is not, as it cannot be tested without the download role. ``download_status`` reports
why Level-1 cannot be read, and CEVA then uses Level-2 surface reflectance for Landsat.
Landsat data are in the US public domain.

Top-of-atmosphere reflectance from a Level-1 band (USGS Landsat 8-9 Data Users Handbook,
section 5.1): rho = (M * DN + A) / sin(sun elevation), with M and A the band's
REFLECTANCE_MULT_BAND_n and REFLECTANCE_ADD_BAND_n from the scene's MTL file.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

from ceva.config import Config
from ceva.credentials import Credentials

log = logging.getLogger(__name__)

USER_AGENT = "ceva (+https://github.com/clacrouts/ceva)"
ACCESS_URL = "https://ers.cr.usgs.gov/"
DATASETS = {"landsat-4": "landsat_tm_c2_l1", "landsat-5": "landsat_tm_c2_l1",
            "landsat-7": "landsat_etm_c2_l1", "landsat-8": "landsat_ot_c2_l1", "landsat-9": "landsat_ot_c2_l1"}


class M2MError(RuntimeError):
    pass


def _call(api: str, endpoint: str, payload: dict, key: str | None = None, timeout: float = 120) -> Any:
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    if key:
        headers["X-Auth-Token"] = key
    request = urllib.request.Request(api.rstrip("/") + "/" + endpoint, data=json.dumps(payload).encode(),
                                     headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        raise M2MError(f"{endpoint}: HTTP {exc.code}") from None
    if body.get("errorCode"):
        raise M2MError(f"{endpoint}: {body['errorCode']}: {body.get('errorMessage')}")
    return body.get("data")


class M2MSession:
    """A logged-in M2M session; use as a context manager so it is logged out again."""

    def __init__(self, api: str, creds: Credentials) -> None:
        self.api = api
        self.creds = creds
        self.key: str | None = None

    def __enter__(self) -> M2MSession:
        self.key = _call(self.api, "login-token", {"username": self.creds.get("USGS_ERS_USERNAME"),
                                                   "token": self.creds.get("USGS_M2M_TOKEN")})
        return self

    def __exit__(self, *exc) -> None:
        if self.key:
            try:
                _call(self.api, "logout", {}, self.key, timeout=30)
            except (M2MError, OSError):
                pass
            self.key = None

    def call(self, endpoint: str, payload: dict) -> Any:
        return _call(self.api, endpoint, payload, self.key)

    def permissions(self) -> list[str]:
        return list(self.call("permissions", {}) or [])


@lru_cache(maxsize=4)
def _download_status(api: str, username: str, token: str) -> str | None:
    creds = Credentials({"USGS_ERS_USERNAME": username, "USGS_M2M_TOKEN": token})
    try:
        with M2MSession(api, creds) as session:
            roles = session.permissions()
    except (M2MError, OSError) as exc:
        return f"USGS M2M login failed ({exc})"
    if "download" not in roles:
        return (f"the USGS account has no M2M download role (roles: {', '.join(roles) or 'none'}); "
                f"request 'Machine to Machine' access on {ACCESS_URL}")
    return "reading Level-1 files through M2M download links is not implemented yet (see ceva.catalog.usgs)"


def download_status(cfg: Config, creds: Credentials) -> str | None:
    """None when Landsat Level-1 pixels can be read, otherwise why not. Asks USGS once per process."""
    if not creds.usgs_ready:
        missing = [k for k in ("USGS_ERS_USERNAME", "USGS_M2M_TOKEN") if not creds.get(k)]
        return f"missing {' and '.join(missing)}"
    return _download_status(cfg.providers["usgs"].m2m_url, creds.get("USGS_ERS_USERNAME"), creds.get("USGS_M2M_TOKEN"))


def band_number(href: str) -> int:
    """The band number in a Landsat Level-1 file name (``..._B3.TIF`` -> 3)."""
    match = re.search(r"_B(\d+)\.TIF$", href, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"not a Landsat band file: {href}")
    return int(match.group(1))


def toa_scaling(mtl: dict, band: int, sun_elevation: float | None = None) -> tuple[float, float]:
    """Scale and offset turning a Level-1 band's DN into top-of-atmosphere reflectance.

    ``mtl`` is the scene's MTL.json. reflectance = DN * scale + offset, with the sun
    elevation correction folded in (from the MTL unless ``sun_elevation`` is given).
    """
    meta = mtl.get("LANDSAT_METADATA_FILE", mtl)
    rescaling = meta["LEVEL1_RADIOMETRIC_RESCALING"]
    mult = float(rescaling[f"REFLECTANCE_MULT_BAND_{band}"])
    add = float(rescaling[f"REFLECTANCE_ADD_BAND_{band}"])
    elevation = float(sun_elevation if sun_elevation is not None else meta["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"])
    if not 0 < elevation <= 90:
        raise ValueError(f"sun elevation {elevation} is not above the horizon")
    sine = math.sin(math.radians(elevation))
    return mult / sine, add / sine
