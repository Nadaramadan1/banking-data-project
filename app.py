import os
import io
import csv
import math
import uuid
import secrets
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_engine import engine
from settings import settings

logger = logging.getLogger(__name__)

# Validate security configuration at application startup
settings.validate_on_startup()

# Maximum accepted CSV upload size — exposes settings.max_upload_bytes for backward compatibility
MAX_UPLOAD_BYTES: int = settings.max_upload_bytes

# All 12 columns that must be present in every uploaded CSV header.
REQUIRED_CSV_COLUMNS = frozenset({
    "age", "balance", "pdays", "previous",
    "job", "marital", "education", "default",
    "housing", "loan", "month", "poutcome",
})

app = FastAPI(
    title="AURA-Term™ Deposit Campaign Propensity & Prioritization Portal",
    description="Neural propensity scoring engine for retail term deposit marketing campaign prioritization.",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


async def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Minimal, maintainable authentication boundary.
    - Local demo mode: if AURA_API_KEY is not configured in settings, requests proceed unauthenticated.
    - Authenticated deployment: if AURA_API_KEY is configured, valid X-API-Key or Bearer token is enforced.
    """
    if not settings.api_key:
        return None

    provided_key = None
    if x_api_key:
        provided_key = x_api_key
    elif authorization and authorization.lower().startswith("bearer "):
        provided_key = authorization[7:].strip()

    if not provided_key or not secrets.compare_digest(provided_key, settings.api_key):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided_key


def mask_internal_error(context: str, exc: Exception) -> HTTPException:
    """Logs the full exception server-side and returns an opaque reference ID to the client."""
    ref_id = str(uuid.uuid4())
    logger.error(f"[Error Ref: {ref_id}] {context}: {exc}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail=f"Internal processing error. Reference ID: {ref_id}"
    )


# Pydantic Schemas
class ClientFeatures(BaseModel):
    age: float = Field(..., ge=18, le=105, description="Client age")
    balance: float = Field(..., description="Average annual balance in EUR/USD")
    pdays: float = Field(-1, description="Days passed after previous campaign contact (-1 = not contacted)")
    previous: float = Field(0, ge=0, description="Number of contacts performed before this campaign")
    job: str = Field(..., description="Job classification")
    marital: str = Field(..., description="Marital status (married, single, divorced)")
    education: str = Field(..., description="Education level (primary, secondary, tertiary, unknown)")
    default: str = Field("no", description="Credit in default (no, yes)")
    housing: str = Field(..., description="Housing loan (no, yes)")
    loan: str = Field("no", description="Personal loan (no, yes)")
    month: str = Field("may", description="Last contact month (jan-dec)")
    poutcome: str = Field("unknown", description="Outcome of previous marketing campaign (unknown, failure, other, success)")
    age_group: Optional[str] = None
    balance_status: Optional[str] = None

class BatchRequest(BaseModel):
    accounts: List[ClientFeatures]

@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/predict", dependencies=[Depends(require_api_key)])
def predict_single(client: ClientFeatures):
    try:
        result = engine.predict_client(client.model_dump())
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise mask_internal_error("Failed to score single client", e)

@app.post("/api/batch", dependencies=[Depends(require_api_key)])
def predict_batch_json(req: BatchRequest):
    if not req.accounts:
        raise HTTPException(status_code=400, detail="Batch request contains no client records.")
    if len(req.accounts) > settings.max_batch_rows:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size exceeds maximum limit of {settings.max_batch_rows} records."
        )
    try:
        records = [acc.model_dump() for acc in req.accounts]
        return engine.predict_batch(records)
    except HTTPException:
        raise
    except Exception as e:
        raise mask_internal_error("Failed to process JSON batch", e)

@app.post("/api/batch-upload", dependencies=[Depends(require_api_key)])
async def predict_batch_csv(file: UploadFile = File(...)):
    # --- 1. Enforce configurable upload size limit via streaming chunks ---
    chunks = []
    total_bytes = 0
    effective_max_upload_bytes = min(settings.max_upload_bytes, MAX_UPLOAD_BYTES)
    while True:
        chunk = await file.read(settings.upload_chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > effective_max_upload_bytes:
            msg = f"{effective_max_upload_bytes // (1024 * 1024)} MB" if effective_max_upload_bytes >= 1024 * 1024 else f"{effective_max_upload_bytes} bytes"
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed: {msg}."
            )
        chunks.append(chunk)

    contents = b"".join(chunks)

    try:
        decoded = contents.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(decoded))

        # --- 2. Validate required columns before scoring a single row ---
        if not reader.fieldnames:
            raise HTTPException(status_code=400, detail="CSV file is empty or has no header row.")

        actual_cols = {c.strip().lower() for c in reader.fieldnames if c}
        missing_cols = {c for c in REQUIRED_CSV_COLUMNS if c.lower() not in actual_cols}
        if missing_cols:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"CSV is missing required columns: {sorted(missing_cols)}. "
                    f"All required columns: {sorted(REQUIRED_CSV_COLUMNS)}."
                )
            )

        records = []
        for row_idx, row in enumerate(reader, start=2):
            if not row or not any(v and str(v).strip() for v in row.values() if v is not None):
                continue

            row_l = {k.strip().lower(): v for k, v in row.items() if k is not None}

            # Validate all required columns are present and not empty/whitespace
            for col in sorted(REQUIRED_CSV_COLUMNS):
                val = row_l.get(col)
                if val is None or not str(val).strip():
                    raise HTTPException(
                        status_code=422,
                        detail=f"Validation error at row {row_idx}: required field '{col}' is missing or empty."
                    )

            # Field-specific numeric validation (must be finite float)
            numeric_fields = {}
            for num_col in ("age", "balance", "pdays", "previous"):
                raw_val = str(row_l[num_col]).strip()
                try:
                    num_val = float(raw_val)
                    if not math.isfinite(num_val):
                        raise ValueError()
                    numeric_fields[num_col] = num_val
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Validation error at row {row_idx}: field '{num_col}' must be a valid numeric value."
                    )

            # Categorical string fields: non-empty, stripped
            str_cols = ("job", "marital", "education", "default", "housing", "loan", "month", "poutcome")
            str_fields = {}
            for s_col in str_cols:
                clean_str = str(row_l[s_col]).strip()
                if not clean_str:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Validation error at row {row_idx}: required field '{s_col}' cannot be empty or whitespace."
                    )
                str_fields[s_col] = clean_str

            records.append({
                **numeric_fields,
                **str_fields,
            })

            # Check CSV row count during iteration
            if len(records) > settings.max_csv_rows:
                raise HTTPException(
                    status_code=413,
                    detail=f"Batch size exceeds maximum limit of {settings.max_csv_rows} rows."
                )

        if not records:
            raise HTTPException(status_code=400, detail="CSV file contains no valid client rows.")

        return engine.predict_batch(records)

    except HTTPException:
        raise
    except csv.Error:
        raise HTTPException(status_code=400, detail="Malformed CSV file structure.")
    except Exception as e:
        raise mask_internal_error("Unexpected error processing CSV upload", e)

@app.post("/api/sensitivity", dependencies=[Depends(require_api_key)])
def compute_sensitivity(client: ClientFeatures):
    try:
        return engine.generate_sensitivity_curves(client.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        raise mask_internal_error("Failed to compute sensitivity curves", e)

@app.get("/api/profiles")
def get_sample_profiles():
    return engine.get_sample_profiles()

@app.get("/api/cohorts/{cohort_id}", dependencies=[Depends(require_api_key)])
def get_sample_cohort(cohort_id: str):
    try:
        records = engine.get_sample_cohort(cohort_id)
        scored = engine.predict_batch(records)
        return scored
    except HTTPException:
        raise
    except Exception as e:
        raise mask_internal_error(f"Failed to fetch sample cohort {cohort_id}", e)

@app.get("/api/governance", dependencies=[Depends(require_api_key)])
def get_governance():
    data = engine.get_governance_metadata()
    data["audit_log_notice"] = (
        "Notice: The audit log entries reflect the current in-memory process session. "
        "They are not immutable, tamper-proof, or stored in a persistent external ledger."
    )
    return data

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

