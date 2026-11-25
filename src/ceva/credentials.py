"""Local settings and secrets. Neither is committed.

* ``.env`` (working directory or a parent): non-secret machine settings such as
  ``CEVA_DATA_DIR``.
* Token files, one per service, in ``~/.config/ceva/tokens/`` (or ``$CEVA_TOKENS_DIR``):
  ``cdse.env``, ``cds.env``, ``usgs.env``, ``earthdata.env``, ``planetary_computer.env``, ``storage.env``. Each holds
  only the keys of its service (``TOKEN_FILES``), and must be readable by its owner only,
  like an SSH key. A token file inside a git working tree is refused unless git ignores
  it, so that a secret cannot be committed by mistake.

Real environment variables win over the files. Secret values are never logged or
included in ``repr``.
"""
from __future__ import annotations

import logging
import os
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, find_dotenv

log = logging.getLogger(__name__)

TOKENS_DIR_ENV = "CEVA_TOKENS_DIR"
DEFAULT_TOKENS_DIR = Path("~/.config/ceva/tokens")
# One file per service, and the keys each may hold.
TOKEN_FILES: dict[str, tuple[str, ...]] = {
    "cdse": ("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY"),               # Copernicus Data Space S3
    "usgs": ("USGS_ERS_USERNAME", "USGS_M2M_TOKEN"),                    # USGS machine-to-machine API
    "earthdata": ("EARTHDATA_TOKEN",),                                  # NASA Earthdata (ASF: NISAR)
    "cds": ("CDSAPI_URL", "CDSAPI_KEY"),                                # Copernicus Climate Data Store (ERA5)
    "planetary_computer": ("PC_SDK_SUBSCRIPTION_KEY",),                 # Planetary Computer account
    # online store of the results (any S3-compatible bucket: Cloudflare R2, AWS S3, Scaleway...)
    "storage": ("STORAGE_ENDPOINT_URL", "STORAGE_BUCKET", "STORAGE_ACCESS_KEY_ID", "STORAGE_SECRET_ACCESS_KEY"),
}
SECRET_KEYS = tuple(k for keys in TOKEN_FILES.values() for k in keys)
LEGACY_CREDENTIALS = Path("~/.config/ceva/credentials.env")


@dataclass(frozen=True, repr=False)
class Credentials:
    values: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str | None:
        return self.values.get(key) or None

    def has(self, *keys: str) -> bool:
        return all(self.values.get(k) for k in keys)

    @property
    def cdse_s3_ready(self) -> bool:
        """Both halves of the Copernicus Data Space S3 key pair are present."""
        return self.has("CDSE_S3_ACCESS_KEY", "CDSE_S3_SECRET_KEY")

    @property
    def usgs_ready(self) -> bool:
        """A USGS EROS (ERS) user name and machine-to-machine application token are present."""
        return self.has("USGS_ERS_USERNAME", "USGS_M2M_TOKEN")

    def status(self) -> dict[str, bool]:
        """Which secrets are set, without their values."""
        return {k: bool(self.values.get(k)) for k in SECRET_KEYS}

    def __repr__(self) -> str:
        return f"Credentials({self.status()})"


def tokens_dir() -> Path:
    return Path(os.environ.get(TOKENS_DIR_ENV) or DEFAULT_TOKENS_DIR).expanduser()


def token_file(service: str, directory: Path | None = None) -> Path:
    if service not in TOKEN_FILES:
        raise KeyError(f"unknown service {service!r}; choose from {list(TOKEN_FILES)}")
    return (directory or tokens_dir()) / f"{service}.env"


def _git_tracked_risk(path: Path) -> bool:
    """True when ``path`` lies in a git working tree and git does not ignore it."""
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            try:
                ignored = subprocess.run(["git", "-C", str(parent), "check-ignore", "-q", str(path.resolve())],
                                         capture_output=True, timeout=10).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ignored = False
            return not ignored
    return False


def _read_token_file(path: Path, allowed: tuple[str, ...]) -> dict[str, str]:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(f"{path} is readable by group or others (mode {mode:o}). Run: chmod 600 {path}")
    if _git_tracked_risk(path):
        raise PermissionError(f"{path} is inside a git working tree and not ignored by git; move it to "
                              f"{DEFAULT_TOKENS_DIR} (or set {TOKENS_DIR_ENV}) so it cannot be committed")
    values = {}
    for key, value in dotenv_values(path).items():
        if key not in allowed:
            log.warning("%s: %s does not belong in this file (it holds %s); ignored", path, key, ", ".join(allowed))
        elif value:
            values[key] = value
    return values


def load_credentials(directory: Path | None = None) -> Credentials:
    """Read secrets from the token files, then from the environment."""
    directory = directory or tokens_dir()
    values: dict[str, str] = {}
    for service, keys in TOKEN_FILES.items():
        path = token_file(service, directory)
        if path.is_file():
            values.update(_read_token_file(path, keys))
    legacy = LEGACY_CREDENTIALS.expanduser()
    if legacy.is_file():
        log.warning("%s is no longer read; move its keys to the token files in %s", legacy, directory)
    for key in SECRET_KEYS:
        if os.environ.get(key):
            values[key] = os.environ[key]
    return Credentials(values)


def load_local_settings(start: Path | None = None) -> Path | None:
    """Export non-secret settings from the nearest ``.env`` into the environment.

    Variables that are already set are left alone. Returns the file used, if any.
    """
    if start is not None:
        candidate = Path(start) / ".env"
        found = str(candidate) if candidate.is_file() else ""
    else:
        found = find_dotenv(usecwd=True)
    if not found:
        return None
    for key, value in dotenv_values(found).items():
        if key in SECRET_KEYS:
            log.warning("%s holds the secret %s; move it to %s", found, key, tokens_dir())
            continue
        if value and key not in os.environ:
            os.environ[key] = value
    return Path(found)
