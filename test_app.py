"""
Tests for the research sandbox API. Run from the deployment/ folder:
  pytest -q
"""

from fastapi.testclient import TestClient

from app import DISCLAIMER, app

client = TestClient(app)

VALID_INPUT = {
    "age": 60,
    "bmi": 28,
    "systolic_bp": 135,
    "biomarker": 1.2,
    "comorbidities": 2,
    "treatment": 1,
}


def test_health_reports_research_status():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["disclaimer"] == DISCLAIMER
    assert "synthetic" in body["metadata"]["trained_on"]


def test_predict_returns_probability_and_disclaimer():
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["predicted_probability"] <= 1.0
    assert body["predicted_label"] in (0, 1)
    assert body["research_only"] is True
    assert body["disclaimer"] == DISCLAIMER


def test_predict_rejects_out_of_range_values():
    bad = {**VALID_INPUT, "age": 250}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_rejects_invalid_treatment_code():
    bad = {**VALID_INPUT, "treatment": 2}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_rejects_missing_fields():
    partial = {"age": 60}
    response = client.post("/predict", json=partial)
    assert response.status_code == 422
