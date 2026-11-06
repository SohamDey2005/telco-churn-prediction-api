"""Unit and integration tests for the churn API (>=15 tests)."""

import os
import sys
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["API_KEY"] = "test-key"

from main import app, load_model  # noqa: E402

client = TestClient(app)

VALID_RECORD = {
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.0,
}

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    load_model()
    yield


# ---------- Health ----------
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["data_hash"] is not None


def test_health_contains_trained_at():
    r = client.get("/health")
    assert "trained_at" in r.json()


# ---------- Auth ----------
def test_predict_missing_api_key():
    r = client.post("/predict", json={"records": [VALID_RECORD]})
    assert r.status_code == 422  # missing header


def test_predict_wrong_api_key():
    r = client.post(
        "/predict",
        json={"records": [VALID_RECORD]},
        headers={"X-API-Key": "wrong"},
    )
    assert r.status_code == 401


# ---------- Validation ----------
def test_predict_empty_records():
    r = client.post("/predict", json={"records": []}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_invalid_tenure_negative():
    bad = VALID_RECORD.copy()
    bad["tenure"] = -1
    r = client.post("/predict", json={"records": [bad]}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_missing_required_field():
    bad = VALID_RECORD.copy()
    del bad["gender"]
    r = client.post("/predict", json={"records": [bad]}, headers=HEADERS)
    assert r.status_code == 422


def test_predict_invalid_type_monthly_charges():
    bad = VALID_RECORD.copy()
    bad["MonthlyCharges"] = "not-a-float"
    r = client.post("/predict", json={"records": [bad]}, headers=HEADERS)
    assert r.status_code == 422


# ---------- Prediction ----------
def test_predict_single_success():
    r = client.post("/predict/single", json=VALID_RECORD, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert "threshold_used" in body


def test_predict_batch_success():
    r = client.post(
        "/predict",
        json={"records": [VALID_RECORD, VALID_RECORD]},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2
    assert "model_data_hash" in body
    assert "model_trained_at" in body


def test_predict_unseen_category_graceful():
    """OneHotEncoder handle_unknown='ignore' should not crash."""
    rec = VALID_RECORD.copy()
    rec["Contract"] = "Three year"  # unseen
    r = client.post("/predict/single", json=rec, headers=HEADERS)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0


def test_predict_high_churn_risk_profile():
    """Month-to-month + high charges + short tenure should tend toward higher proba."""
    risky = VALID_RECORD.copy()
    risky["tenure"] = 1
    risky["Contract"] = "Month-to-month"
    risky["MonthlyCharges"] = 110.0
    risky["TotalCharges"] = 110.0
    r = client.post("/predict/single", json=risky, headers=HEADERS)
    assert r.status_code == 200
    # Not a hard assert on value, but probability should be defined
    assert isinstance(r.json()["churn_probability"], float)


def test_latency_header_present():
    r = client.post("/predict/single", json=VALID_RECORD, headers=HEADERS)
    assert "X-Latency-Ms" in r.headers
    latency = float(r.headers["X-Latency-Ms"])
    assert latency < 500  # generous; production target <100 ms median


# ---------- Drift ----------
def test_drift_endpoint():
    r = client.post(
        "/drift",
        json={"records": [VALID_RECORD] * 5},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert "overall_flag" in body
    assert "features" in body
    assert len(body["features"]) >= 1


def test_drift_psi_non_negative():
    r = client.post(
        "/drift",
        json={"records": [VALID_RECORD] * 10},
        headers=HEADERS,
    )
    for f in r.json()["features"]:
        assert f["psi"] >= 0.0


# ---------- Model artefact integrity ----------
def test_model_artefact_exists():
    path = ROOT / "models" / "churn_pipeline.joblib"
    assert path.exists()
    art = joblib.load(path)
    assert "model" in art
    assert "threshold" in art
    assert "data_hash" in art
    assert "metrics" in art
    assert "versions" in art


def test_model_metrics_reasonable():
    art = joblib.load(ROOT / "models" / "churn_pipeline.joblib")
    m = art["metrics"]
    assert 0.7 < m["roc_auc"] < 1.0
    assert 0.05 < m["threshold"] < 0.95
    assert m["recall"] > 0.5  # cost-sensitive high recall
