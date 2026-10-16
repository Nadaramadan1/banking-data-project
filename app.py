import os
import io
import csv
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_engine import engine

app = FastAPI(
    title="AURA-Term™ Institutional Banking & Sovereign Deposit Intelligence Portal",
    description="Enterprise-grade neural deposit underwriting engine for central banks and global financial institutions.",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

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

@app.post("/api/predict")
def predict_single(client: ClientFeatures):
    try:
        result = engine.predict_client(client.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch")
def predict_batch_json(req: BatchRequest):
    try:
        records = [acc.dict() for acc in req.accounts]
        return engine.predict_batch(records)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-upload")
async def predict_batch_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        decoded = contents.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(decoded))
        records = []
        for row in reader:
            if not row:
                continue
            records.append({
                "age": float(row.get("age", 40)),
                "balance": float(row.get("balance", 1000)),
                "pdays": float(row.get("pdays", -1)),
                "previous": float(row.get("previous", 0)),
                "job": row.get("job", "management").strip(),
                "marital": row.get("marital", "married").strip(),
                "education": row.get("education", "secondary").strip(),
                "default": row.get("default", "no").strip(),
                "housing": row.get("housing", "no").strip(),
                "loan": row.get("loan", "no").strip(),
                "month": row.get("month", "may").strip(),
                "poutcome": row.get("poutcome", "unknown").strip()
            })
        if not records:
            raise HTTPException(status_code=400, detail="CSV file contains no valid client rows.")
        return engine.predict_batch(records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

@app.post("/api/sensitivity")
def compute_sensitivity(client: ClientFeatures):
    try:
        return engine.generate_sensitivity_curves(client.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profiles")
def get_sample_profiles():
    return engine.get_sample_profiles()

@app.get("/api/cohorts/{cohort_id}")
def get_sample_cohort(cohort_id: str):
    try:
        records = engine.get_sample_cohort(cohort_id)
        scored = engine.predict_batch(records)
        return scored
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/governance")
def get_governance():
    return engine.get_governance_metadata()

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
