"""
AURA-Term Phase 4 — End-to-End API Integration & Verification Test Suite
========================================================================
Validates full request/response contracts, input validation boundaries,
batch size boundaries, CSV upload edge cases, authentication enforcement across
protected vs public endpoints, and safe error masking.
"""
import io
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app import app
from settings import settings

client = TestClient(app)

VALID_CLIENT_PAYLOAD = {
    "age": 42,
    "balance": 2500.0,
    "pdays": -1,
    "previous": 0,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "month": "jun",
    "poutcome": "unknown"
}


# ---------------------------------------------------------------------------
# 1. Single Prediction API Contract Tests
# ---------------------------------------------------------------------------

def test_predict_single_valid_contract():
    """Verify that a valid single prediction returns the complete expected schema."""
    response = client.post("/api/predict", json=VALID_CLIENT_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    # Validate essential schema fields and types per model_engine contract
    assert "request_id" in data
    assert data["request_id"].startswith("REQ-")
    assert isinstance(data["propensity_probability"], (float, int))
    assert 0.0 <= data["propensity_probability"] <= 1.0
    assert isinstance(data["propensity_percentage"], (float, int))
    assert isinstance(data["predicted_subscribed"], bool)
    assert data["confidence_rating"] in ("High", "Moderate")
    assert isinstance(data["inference_time_ms"], (float, int))

    assert "derived_features" in data
    assert "age_group" in data["derived_features"]
    assert "balance_status" in data["derived_features"]

    assert "decision" in data
    assert "tier_name" in data["decision"]
    assert "tier_badge" in data["decision"]
    assert "estimated_deposit_volume" in data["decision"]

    assert "feature_attributions" in data
    assert isinstance(data["feature_attributions"], list)
    assert len(data["feature_attributions"]) > 0
    for factor in data["feature_attributions"]:
        assert "feature" in factor
        assert "display_name" in factor
        assert "impact_percentage" in factor
        assert "direction" in factor


@pytest.mark.parametrize("missing_field", ["age", "balance", "job", "marital", "education", "housing"])
def test_predict_single_missing_required_field_returns_422(missing_field):
    """Verify that omitting any required field returns HTTP 422."""
    payload = dict(VALID_CLIENT_PAYLOAD)
    del payload[missing_field]
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("invalid_age", [10, 17, 106, 150, -5])
def test_predict_single_out_of_range_age_returns_422(invalid_age):
    """Verify that age values outside [18, 105] return HTTP 422."""
    payload = dict(VALID_CLIENT_PAYLOAD, age=invalid_age)
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_single_non_numeric_balance_returns_422():
    """Verify that non-numeric balance returns HTTP 422."""
    payload = dict(VALID_CLIENT_PAYLOAD, balance="not-a-number")
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 2. Batch Request & Boundary Behavior Tests
# ---------------------------------------------------------------------------

def test_batch_single_item_contract():
    """Verify that a batch of length 1 returns batch telemetry and scored account matching schema."""
    response = client.post("/api/batch", json={"accounts": [VALID_CLIENT_PAYLOAD]})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert data["total_accounts_scored"] == 1
    assert "average_propensity" in data
    assert "accounts" in data
    assert len(data["accounts"]) == 1
    acc = data["accounts"][0]
    assert acc["account_id"] == "ACC-10000"
    assert "propensity_score" in acc
    assert "tier" in acc


def test_batch_multi_item_order_and_indexing():
    """Verify that multiple accounts in a batch preserve indexing and ordering."""
    payloads = [
        dict(VALID_CLIENT_PAYLOAD, age=25, balance=500.0),
        dict(VALID_CLIENT_PAYLOAD, age=45, balance=15000.0),
        dict(VALID_CLIENT_PAYLOAD, age=65, balance=80000.0),
    ]
    response = client.post("/api/batch", json={"accounts": payloads})
    assert response.status_code == 200
    data = response.json()
    assert data["total_accounts_scored"] == 3
    assert len(data["accounts"]) == 3
    for idx, item in enumerate(data["accounts"]):
        assert item["account_id"] == f"ACC-{10000 + idx}"
        assert 0.0 <= item["propensity_score"] <= 100.0


def test_batch_exact_max_boundary_accepted():
    """Verify that exactly MAX_BATCH_ROWS items is accepted."""
    orig_limit = settings.max_batch_rows
    try:
        settings.max_batch_rows = 4
        payloads = [VALID_CLIENT_PAYLOAD] * 4
        response = client.post("/api/batch", json={"accounts": payloads})
        assert response.status_code == 200
        data = response.json()
        assert data["total_accounts_scored"] == 4
        assert len(data["accounts"]) == 4
    finally:
        settings.max_batch_rows = orig_limit


# ---------------------------------------------------------------------------
# 3. CSV Upload Integration & Edge Cases
# ---------------------------------------------------------------------------

def test_csv_upload_extra_columns_accepted():
    """Verify that CSV with additional unknown columns succeeds as long as required columns are present."""
    csv_text = (
        "age,balance,pdays,previous,job,marital,education,default,housing,loan,month,poutcome,extra_id,notes\n"
        "42,2500,-1,0,technician,married,secondary,no,yes,no,jun,unknown,EXT-998,test client\n"
    )
    files = {"file": ("extra_cols.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}
    response = client.post("/api/batch-upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_accounts_scored"] == 1
    assert len(data["accounts"]) == 1
    assert "propensity_score" in data["accounts"][0]


def test_csv_upload_row_empty_string_error_detail():
    """Verify that an empty string in a required field points to the exact row and field."""
    csv_text = (
        "age,balance,pdays,previous,job,marital,education,default,housing,loan,month,poutcome\n"
        "42,2500,-1,0,technician,married,secondary,no,yes,no,jun,unknown\n"
        "35,1200,-1,0,,single,tertiary,no,no,no,may,unknown\n"
    )
    files = {"file": ("bad_row.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}
    response = client.post("/api/batch-upload", files=files)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "row 3" in detail
    assert "job" in detail


def test_csv_upload_malformed_csv_error():
    """Verify that malformed CSV with unescaped null bytes returns HTTP 400."""
    corrupted_data = b"age,balance,pdays\x00corrupted\x00data\xff\xfe"
    files = {"file": ("corrupt.csv", io.BytesIO(corrupted_data), "text/csv")}
    response = client.post("/api/batch-upload", files=files)
    assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 4. Authentication Behavior Across Endpoints
# ---------------------------------------------------------------------------

def test_public_endpoints_accessible_when_api_key_configured():
    """Verify that public endpoints remain open even when an institutional API key is enforced."""
    orig_key = settings.api_key
    try:
        settings.api_key = "secure-institutional-key"

        # Public endpoints should return 200 without auth header
        res_home = client.get("/")
        assert res_home.status_code == 200

        res_profiles = client.get("/api/profiles")
        assert res_profiles.status_code == 200
        assert isinstance(res_profiles.json(), list)

        # Protected endpoints must return 401 without key
        res_predict = client.post("/api/predict", json=VALID_CLIENT_PAYLOAD)
        assert res_predict.status_code == 401

        res_gov = client.get("/api/governance")
        assert res_gov.status_code == 401

        res_cohort = client.get("/api/cohorts/young_professionals")
        assert res_cohort.status_code == 401

        # Protected endpoints succeed with valid key
        res_gov_auth = client.get(
            "/api/governance",
            headers={"Authorization": "Bearer secure-institutional-key"}
        )
        assert res_gov_auth.status_code == 200
    finally:
        settings.api_key = orig_key


# ---------------------------------------------------------------------------
# 5. Safe Client-Facing Errors for Analytical Endpoints
# ---------------------------------------------------------------------------

def test_safe_error_masking_sensitivity():
    """Verify that unexpected exceptions on /api/sensitivity return masked 500 error."""
    with patch("model_engine.engine.generate_sensitivity_curves", side_effect=RuntimeError("GPU kernel failure in math library")):
        response = client.post("/api/sensitivity", json=VALID_CLIENT_PAYLOAD)
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "Internal processing error. Reference ID:" in detail
        assert "GPU kernel failure" not in detail


def test_safe_error_masking_cohorts():
    """Verify that unexpected exceptions on /api/cohorts return masked 500 error."""
    with patch("model_engine.engine.get_sample_cohort", side_effect=KeyError("Unrecognized internal cohort key")):
        response = client.get("/api/cohorts/test_cohort")
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "Internal processing error. Reference ID:" in detail
        assert "Unrecognized internal cohort key" not in detail
