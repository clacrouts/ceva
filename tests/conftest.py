import pytest

from ceva.config import DATA_DIR_ENV, load_config
from ceva.credentials import SECRET_KEYS


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Tests never see the developer's data directory or secrets."""
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    for key in SECRET_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def cfg(tmp_path):
    return load_config("gironde", overrides={"data_dir": str(tmp_path)})
