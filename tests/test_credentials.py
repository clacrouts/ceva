import os
import subprocess

import pytest

from ceva.credentials import SECRET_KEYS, TOKEN_FILES, load_credentials, load_local_settings, token_file


def _write(path, text, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


@pytest.fixture(autouse=True)
def _no_secrets_in_environment(monkeypatch):
    for key in SECRET_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_half_a_key_pair_is_not_ready_and_never_printed(tmp_path):
    _write(tmp_path / "cdse.env", "CDSE_S3_ACCESS_KEY=AKIDEXAMPLE\nCDSE_S3_SECRET_KEY=\n")
    creds = load_credentials(tmp_path)
    assert creds.get("CDSE_S3_ACCESS_KEY") == "AKIDEXAMPLE"
    assert creds.get("CDSE_S3_SECRET_KEY") is None
    assert not creds.cdse_s3_ready
    assert "AKIDEXAMPLE" not in repr(creds)
    assert creds.status()["CDSE_S3_ACCESS_KEY"] is True


def test_one_file_per_service(tmp_path):
    _write(tmp_path / "cdse.env", "CDSE_S3_ACCESS_KEY=a\nCDSE_S3_SECRET_KEY=b\n")
    _write(tmp_path / "usgs.env", "USGS_ERS_USERNAME=u\nUSGS_M2M_TOKEN=t\n")
    creds = load_credentials(tmp_path)
    assert creds.cdse_s3_ready and creds.usgs_ready
    assert token_file("earthdata", tmp_path).name == "earthdata.env"
    assert set(SECRET_KEYS) == {k for keys in TOKEN_FILES.values() for k in keys}


def test_a_key_in_the_wrong_file_is_ignored(tmp_path):
    _write(tmp_path / "usgs.env", "USGS_M2M_TOKEN=t\nCDSE_S3_SECRET_KEY=misplaced\n")
    creds = load_credentials(tmp_path)
    assert creds.get("USGS_M2M_TOKEN") == "t" and creds.get("CDSE_S3_SECRET_KEY") is None


def test_environment_wins_over_file(tmp_path, monkeypatch):
    _write(tmp_path / "cdse.env", "CDSE_S3_ACCESS_KEY=from_file\n")
    monkeypatch.setenv("CDSE_S3_ACCESS_KEY", "from_env")
    assert load_credentials(tmp_path).get("CDSE_S3_ACCESS_KEY") == "from_env"


def test_group_readable_file_is_refused(tmp_path):
    _write(tmp_path / "cdse.env", "CDSE_S3_ACCESS_KEY=a\n", mode=0o640)
    with pytest.raises(PermissionError, match="chmod 600"):
        load_credentials(tmp_path)


def test_token_file_in_a_git_tree_must_be_ignored(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _write(repo / "tokens" / "cdse.env", "CDSE_S3_ACCESS_KEY=a\n")
    with pytest.raises(PermissionError, match="git"):
        load_credentials(repo / "tokens")
    (repo / ".gitignore").write_text("tokens/\n")
    assert load_credentials(repo / "tokens").get("CDSE_S3_ACCESS_KEY") == "a"


def test_missing_files_give_empty_credentials(tmp_path):
    creds = load_credentials(tmp_path / "absent")
    assert not any(creds.status().values())


def test_local_settings_do_not_override_or_leak_secrets(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("CEVA_DATA_DIR=/data/a\nOTHER=1\nCDSE_S3_SECRET_KEY=oops\n")
    monkeypatch.setenv("OTHER", "kept")
    monkeypatch.delenv("CEVA_DATA_DIR", raising=False)
    assert load_local_settings(tmp_path) == tmp_path / ".env"
    assert os.environ["CEVA_DATA_DIR"] == "/data/a"
    assert os.environ["OTHER"] == "kept"
    assert "CDSE_S3_SECRET_KEY" not in os.environ
    monkeypatch.delenv("CEVA_DATA_DIR")
