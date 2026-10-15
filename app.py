"""
Research sandbox API for the treatment outcome model.

RESEARCH SANDBOX ONLY.
- The model is trained on a SYNTHETIC cohort. It has not been validated on real patients.
- Predictions must not be used for diagnosis, treatment, or any clinical decision.
- Run locally for testing. Do not expose this service to the public internet.

Run locally:
  uvicorn app:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"
DISCLAIMER = (
    "RESEARCH SANDBOX ONLY. Model trained on synthetic data. "
    "Not medical advice. Not for clinical use."
)


class PatientFeatures(BaseModel):
    """Synthetic-style feature inputs. Do not submit real patient data."""
    age: float = Field(..., ge=18, le=90, description="Age in years")
    bmi: float = Field(..., ge=15, le=50, description="Body mass index")
    systolic_bp: float = Field(..., ge=90, le=200, description="Systolic blood pressure, mmHg")
    biomarker: float = Field(..., gt=0, description="Biomarker level (positive value)")
    comorbidities: int = Field(..., ge=0, le=20, description="Number of comorbidities")
    treatment: Literal[0, 1] = Field(..., description="0 = standard care, 1 = new treatment")


class PredictionResponse(BaseModel):
    predicted_probability: float
    predicted_label: int
    disclaimer: str = DISCLAIMER
    research_only: bool = True


def load_bundle():
    if not ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {ARTIFACT_PATH}. "
            "Run train_and_export.py first."
        )
    return joblib.load(ARTIFACT_PATH)


bundle = load_bundle()
app = FastAPI(
    title="Treatment outcome model (research sandbox)",
    version=bundle["metadata"]["version"],
    description=DISCLAIMER,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "metadata": bundle["metadata"],
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    row = pd.DataFrame([features.model_dump()])[bundle["features"]]
    try:
        proba = float(bundle["model"].predict_proba(row)[0, 1])
    except Exception as exc:  # defensive: unexpected model failure
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return PredictionResponse(
        predicted_probability=round(proba, 4),
        predicted_label=int(proba >= 0.5),
    )
