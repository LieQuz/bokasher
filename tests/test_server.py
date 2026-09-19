from __future__ import annotations

from fastapi.testclient import TestClient

from bokasher.server import APP_DISPLAY_NAME, app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["name"] == APP_DISPLAY_NAME


def test_probe_missing_file() -> None:
    client = TestClient(app)
    response = client.post("/probe", json={"path": "/tmp/bokasher-missing.mov"})
    assert response.status_code == 404
