"""
Tests for Phase 3 — Backend Security Hardening
===============================================
Covers:
1. CORS configuration (allowed origins, credentials, wildcard protection).
2. Request & Resource Limits (streaming upload size limit, JSON batch limit, CSV row limit, empty batch rejection).
3. Safe Error Handling & Traceback Masking (reference IDs, no raw str(e) leaks).
4. Authentication Boundary (demo mode pass-through, API key enforcement via X-API-Key and Bearer tokens).
5. Audit Log Honesty (governance notice confirming ephemeral in-memory session logging).
"""
import io
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app import app
from settings import settings, Settings

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. CORS Configuration Tests
# ---------------------------------------------------------------------------

def test_cors_allowed_origin():
    """Verify that requests from configured origins receive CORS headers."""
    response = client.options(
        "/api/profiles",
        headers={
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "GET",
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_cors_disallowed_origin():
    """Verify that requests from unlisted origins do NOT receive CORS allow origin headers."""
    response = client.options(
        "/api/profiles",
        headers={
            "Origin": "http://malicious-external-site.com",
            "Access-Control-Request-Method": "GET",
        }
    )
    assert response.headers.get("access-control-allow-origin") is None


def test_cors_wildcard_credentials_safety():
    """Verify that if wildcard origin is configured, credentials must be disabled per W3C spec."""
    with patch.dict("os.environ", {"CORS_ALLOWED_ORIGINS": "*"}):
        s = Settings()
        assert s.cors_allow_credentials is False

    with patch.dict("os.environ", {"CORS_ALLOWED_ORIGINS": "https://trusted-bank.internal"}):
        s = Settings()
        assert s.cors_allow_credentials is True


# ---------------------------------------------------------------------------
# 2. Request and Resource Limits Tests
# ---------------------------------------------------------------------------

def test_streaming_upload_size_limit():
    """Verify that uploads exceeding MAX_UPLOAD_BYTES are rejected with HTTP 413."""
    # Temporarily set max_upload_bytes to 500 bytes for test
    orig_limit = settings.max_upload_bytes
    try:
        settings.max_upload_bytes = 500
        large_content = b"age,balance,pdays,previous,job,marital,education,default,housing,loan,month,poutcome\n" + (b"40,1000,-1,0,admin,married,secondary,no,no,no,may,unknown\n" * 20)
        assert len(large_content) > 500

        files = {"file": ("large_test.csv", io.BytesIO(large_content), "text/csv")}
        response = client.post("/api/batch-upload", files=files)
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]
    finally:
        settings.max_upload_bytes = orig_limit


def test_batch_json_empty_rejection():
    """Verify that empty JSON batch requests are rejected with HTTP 400."""
    response = client.post("/api/batch", json={"accounts": []})
    assert response.status_code == 400
    assert "contains no client records" in response.json()["detail"]


def test_batch_json_row_count_limit():
    """Verify that JSON batches exceeding MAX_BATCH_ROWS are rejected with HTTP 413."""
    orig_limit = settings.max_batch_rows
    try:
        settings.max_batch_rows = 5
        sample_client = {
            "age": 40,
            "balance": 1000,
            "pdays": -1,
            "previous": 0,
            "job": "management",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "month": "may",
            "poutcome": "unknown"
        }
        batch = [sample_client] * 6
        response = client.post("/api/batch", json={"accounts": batch})
        assert response.status_code == 413
        assert "Batch size exceeds maximum limit" in response.json()["detail"]
    finally:
        settings.max_batch_rows = orig_limit


def test_csv_row_count_limit():
    """Verify that CSV files exceeding MAX_CSV_ROWS are rejected with HTTP 413."""
    orig_limit = settings.max_csv_rows
    try:
        settings.max_csv_rows = 3
        csv_data = (
            "age,balance,pdays,previous,job,marital,education,default,housing,loan,month,poutcome\n"
            "40,1000,-1,0,admin,married,secondary,no,no,no,may,unknown\n"
            "41,1000,-1,0,admin,married,secondary,no,no,no,may,unknown\n"
            "42,1000,-1,0,admin,married,secondary,no,no,no,may,unknown\n"
            "43,1000,-1,0,admin,married,secondary,no,no,no,may,unknown\n"
        )
        files = {"file": ("rows_test.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
        response = client.post("/api/batch-upload", files=files)
        assert response.status_code == 413
        assert "Batch size exceeds maximum limit of 3 rows" in response.json()["detail"]
    finally:
        settings.max_csv_rows = orig_limit


# ---------------------------------------------------------------------------
# 3. Safe Error Responses & Diagnostic Logging Tests
# ---------------------------------------------------------------------------

def test_safe_error_masking_predict():
    """Verify that internal unexpected exceptions do not expose tracebacks or internal detail."""
    sample_client = {
        "age": 40,
        "balance": 1000,
        "pdays": -1,
        "previous": 0,
        "job": "management",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "housing": "no",
        "loan": "no",
        "month": "may",
        "poutcome": "unknown"
    }
    with patch("model_engine.engine.predict_client", side_effect=RuntimeError("Secret database path /var/secrets/key")):
        response = client.post("/api/predict", json=sample_client)
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "Internal processing error. Reference ID:" in detail
        assert "Secret database path" not in detail
        assert "RuntimeError" not in detail


def test_safe_error_masking_batch():
    """Verify that unexpected exceptions in batch inference return a safe masked error response."""
    sample_client = {
        "age": 40,
        "balance": 1000,
        "pdays": -1,
        "previous": 0,
        "job": "management",
        "marital": "married",
        "education": "secondary",
        "default": "no",
        "housing": "no",
        "loan": "no",
        "month": "may",
        "poutcome": "unknown"
    }
    with patch("model_engine.engine.predict_batch", side_effect=ValueError("Internal tensor layout failure")):
        response = client.post("/api/batch", json={"accounts": [sample_client]})
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "Internal processing error. Reference ID:" in detail
        assert "Internal tensor layout failure" not in detail


# ---------------------------------------------------------------------------
# 4. Authentication & Authorization Boundary Tests
# ---------------------------------------------------------------------------

def test_auth_demo_mode_allows_requests():
    """In local demo mode (api_key=None), requests are permitted without authentication."""
    orig_key = settings.api_key
    try:
        settings.api_key = None
        sample_client = {
            "age": 40,
            "balance": 1000,
            "pdays": -1,
            "previous": 0,
            "job": "management",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "month": "may",
            "poutcome": "unknown"
        }
        response = client.post("/api/predict", json=sample_client)
        assert response.status_code == 200
    finally:
        settings.api_key = orig_key


def test_auth_enforced_when_api_key_configured():
    """When AURA_API_KEY is configured, unauthenticated or bad key requests must return HTTP 401."""
    orig_key = settings.api_key
    try:
        settings.api_key = "test-secret-institutional-key-12345"
        sample_client = {
            "age": 40,
            "balance": 1000,
            "pdays": -1,
            "previous": 0,
            "job": "management",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "month": "may",
            "poutcome": "unknown"
        }

        # 1. Missing header
        res_missing = client.post("/api/predict", json=sample_client)
        assert res_missing.status_code == 401
        assert "Unauthorized" in res_missing.json()["detail"]

        # 2. Invalid X-API-Key
        res_bad = client.post(
            "/api/predict",
            json=sample_client,
            headers={"X-API-Key": "wrong-key"}
        )
        assert res_bad.status_code == 401

        # 3. Valid X-API-Key
        res_ok_api_key = client.post(
            "/api/predict",
            json=sample_client,
            headers={"X-API-Key": "test-secret-institutional-key-12345"}
        )
        assert res_ok_api_key.status_code == 200

        # 4. Valid Authorization Bearer header
        res_ok_bearer = client.post(
            "/api/predict",
            json=sample_client,
            headers={"Authorization": "Bearer test-secret-institutional-key-12345"}
        )
        assert res_ok_bearer.status_code == 200
    finally:
        settings.api_key = orig_key


# ---------------------------------------------------------------------------
# 5. Audit Log Honesty Tests
# ---------------------------------------------------------------------------

def test_governance_audit_log_honesty():
    """Verify that the governance endpoint explicitly communicates the in-memory nature of audit logs."""
    response = client.get("/api/governance")
    assert response.status_code == 200
    data = response.json()
    assert "audit_log_notice" in data
    assert "in-memory" in data["audit_log_notice"].lower()
    assert "not immutable" in data["audit_log_notice"].lower()
