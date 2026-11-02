import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app

TEST_DB = "test_usage.db"


@pytest.fixture
def test_settings(tmp_path):
    db_file = tmp_path / "test_usage.db"
    return Settings(
        app_name="examate-rag",
        environment="test",
        database_url=f"sqlite:///{db_file}",
        primary_model="gemini-3.8-flash",
        secondary_model="gemini-3.1-flash-lite",
        llm_timeout_seconds=10.0,
        llm_max_retries=1,
        usd_to_inr_rate=84.0,
    )


@pytest.fixture
def client(test_settings):
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
