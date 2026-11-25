"""Online storage of finished units: any S3-compatible object store (Cloudflare R2, AWS S3, Scaleway...).

The bucket, its endpoint and its keys are in the token file ``storage.env`` (``ceva.credentials``):
``STORAGE_ENDPOINT_URL`` (for a Cloudflare R2 bucket kept in the EU,
``https://<account id>.eu.r2.cloudflarestorage.com``), ``STORAGE_BUCKET``, ``STORAGE_ACCESS_KEY_ID``
and ``STORAGE_SECRET_ACCESS_KEY``. Objects go under ``storage.prefix`` with the layout of the data
directory: ``cubes/<unit>.icechunk/...``, ``results/<unit>/<method>/...``, plus ``status/<unit>.json``.

An Icechunk repository on a local disk is a set of files whose relative paths are its object keys,
so a repository copied file by file is the same repository in the bucket. Every upload is checked by
opening the copy with Icechunk and comparing its snapshot before the local copy may be deleted.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ceva.config import Config
from ceva.credentials import TOKEN_FILES, Credentials, load_credentials

log = logging.getLogger(__name__)

KEYS = TOKEN_FILES["storage"]
LAST = ("repo",)    # Icechunk's entry point, uploaded after everything it points to


class RemoteStore:
    """One bucket and prefix; ``client`` is a boto3 S3 client (or a stand-in with the same methods)."""

    def __init__(self, cfg: Config, creds: Credentials, client=None):
        self.bucket = creds.get("STORAGE_BUCKET")
        self.endpoint = creds.get("STORAGE_ENDPOINT_URL")
        self.prefix = cfg.storage.prefix.strip("/")
        self.region = cfg.storage.region
        self._key_id = creds.get("STORAGE_ACCESS_KEY_ID")
        self._secret = creds.get("STORAGE_SECRET_ACCESS_KEY")
        if client is None:
            import boto3
            from botocore.config import Config as BotoConfig

            client = boto3.client("s3", endpoint_url=self.endpoint, aws_access_key_id=self._key_id,
                                  aws_secret_access_key=self._secret, region_name=self.region,
                                  config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}))
        self.client = client

    def __repr__(self) -> str:          # never the keys
        return f"RemoteStore({self.endpoint}/{self.bucket}/{self.prefix})"

    def key(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in (self.prefix, *parts) if p)

    def upload_tree(self, local: Path, *parts: str) -> tuple[int, int]:
        """Upload every file under ``local`` to ``<prefix>/<parts>/<relative path>``. Returns files and bytes."""
        files = sorted(p for p in Path(local).rglob("*") if p.is_file())
        files.sort(key=lambda p: p.relative_to(local).parts[:1] in [LAST])     # the entry point last
        size = 0
        for path in files:
            self.client.upload_file(str(path), self.bucket, self.key(*parts, path.relative_to(local).as_posix()))
            size += path.stat().st_size
        return len(files), size

    def put_json(self, obj: dict, *parts: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.key(*parts), Body=json.dumps(obj, default=str).encode(),
                               ContentType="application/json")

    def get_json(self, *parts: str) -> dict | None:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self.key(*parts))["Body"].read()
        except Exception as exc:                       # missing (404) or unreadable: treated as absent
            if _is_missing(exc):
                return None
            raise
        return json.loads(body)

    def icechunk_storage(self, *parts: str):
        """Icechunk storage for the repository at ``<prefix>/<parts>``."""
        import icechunk

        return icechunk.s3_storage(bucket=self.bucket, prefix=self.key(*parts), endpoint_url=self.endpoint,
                                   region=self.region, access_key_id=self._key_id,
                                   secret_access_key=self._secret, force_path_style=True)

    def check_icechunk(self, snapshot_id: str, *parts: str) -> bool:
        """Whether the uploaded repository opens and its main branch is at ``snapshot_id``."""
        import icechunk

        repo = icechunk.Repository.open(self.icechunk_storage(*parts))
        return repo.readonly_session("main").snapshot_id == snapshot_id


def _is_missing(exc: Exception) -> bool:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
    return code in ("404", "NoSuchKey", "NotFound") or isinstance(exc, FileNotFoundError | KeyError)


def remote_store(cfg: Config, creds: Credentials | None = None) -> RemoteStore | None:
    """The configured online store, or None while ``storage.env`` is missing or incomplete."""
    creds = creds if creds is not None else load_credentials()
    if not creds.has(*KEYS):
        return None
    return RemoteStore(cfg, creds)
