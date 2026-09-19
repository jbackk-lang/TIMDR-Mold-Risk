"""
test_api.py — testy integracyjne api.py (Flask test client, bez sieci).
"""

import time

import pytest

import api as api_module


@pytest.fixture
def client():
    api_module.app.config["TESTING"] = True
    api_module.ha_buffer.clear()
    with api_module.app.test_client() as c:
        yield c
    api_module.ha_buffer.clear()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_scenarios_list(client):
    r = client.get("/api/scenarios")
    assert r.status_code == 200
    ids = {s["id"] for s in r.get_json()}
    assert ids == {
        "normal_ventilated_room", "poor_ventilation_bathroom",
        "monsoon_humidity_onset", "water_leak_sudden",
    }


def test_demo_then_analyze_roundtrip(client):
    demo = client.get("/api/demo?scenario=water_leak_sudden").get_json()
    assert len(demo["t_hours"]) > 0

    r = client.post("/api/analyze", json={
        "t_hours": demo["t_hours"],
        "temperature": demo["temperature"],
        "humidity": demo["humidity"],
        "threshold": demo["default_threshold"],
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["current_risk_level"] >= 3  # wyciek wody - powinien przekroczyc widocznosc
    assert body["time_to_visible_growth_hours"] == 0.0


def test_analyze_missing_fields(client):
    r = client.post("/api/analyze", json={"t_hours": [1, 2, 3]})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_dashboard_served(client):
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert b"TIMDR-Mold-Risk" in r.data


def test_chartjs_served_locally(client):
    r = client.get("/static/chart.umd.js")
    assert r.status_code == 200
    assert b"Chart.js" in r.data[:200]


def test_ha_status_empty_before_any_ingest(client):
    r = client.get("/api/ha/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["readings_count"] == 0
    assert body["current_risk_level"] == 0
    assert body["M"] is None


def test_ha_ingest_missing_fields(client):
    r = client.post("/api/ha/ingest", json={"temperature": 23.0})
    assert r.status_code == 400


def test_ha_ingest_and_status_flow(client):
    base = time.time() - 3600 * 24 * 30
    for i in range(60):
        t = base + i * 3600
        hum = 55.0 if i < 30 else 96.0
        r = client.post("/api/ha/ingest", json={
            "temperature": 23.0, "humidity": hum, "timestamp": t,
        })
        assert r.status_code == 200

    status = client.get("/api/ha/status").get_json()
    assert status["readings_count"] == 60
    assert status["last_humidity"] == 96.0
    assert status["last_temperature"] == 23.0
    assert status["M"] is not None

    reset = client.post("/api/ha/reset")
    assert reset.status_code == 200
    status_after = client.get("/api/ha/status").get_json()
    assert status_after["readings_count"] == 0


def test_ha_ingest_accepts_iso_timestamp(client):
    r = client.post("/api/ha/ingest", json={
        "temperature": 22.5, "humidity": 60.0, "timestamp": "2026-09-19T10:00:00+00:00",
    })
    assert r.status_code == 200
    status = client.get("/api/ha/status").get_json()
    assert status["readings_count"] == 1


def test_ha_ingest_rejects_bad_timestamp(client):
    r = client.post("/api/ha/ingest", json={
        "temperature": 22.5, "humidity": 60.0, "timestamp": "nie-jest-to-data",
    })
    assert r.status_code == 400
