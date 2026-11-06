"""
FastAPI application serving the churn prediction model.
- Single & batch prediction endpoints
- Health check
- API-key authentication
- Structured logging + latency
- Graceful handling of unseen categories (via OneHotEncoder handle_unknown)
- Basic PSI drift endpoint (stretch)
"""

import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from schemas import (
    BatchPredictionResponse,
    CustomerFeatures,
    DriftFeature,
    DriftResponse,
    HealthResponse,
    PredictionRequest,
    SinglePrediction,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
API_KEY = os.getenv("API_KEY", "churn-demo-key-change-me")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("churn-api")

# Global artefact
artefact: Optional[Dict[str, Any]] = None


def load_model():
    global artefact
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    artefact = joblib.load(MODEL_PATH)
    logger.info(
        "Model loaded | data_hash=%s | trained_at=%s | threshold=%.3f",
        artefact["data_hash"],
        artefact["trained_at"],
        artefact["threshold"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Secure real-time and batch churn scoring service. "
        "Pipeline bundles cleaning, encoding, scaling and a tuned classifier. "
        "Decision threshold selected by expected cost of FN vs FP."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Middleware: latency + structured request log
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status=%s latency_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    response.headers["X-Latency-Ms"] = f"{latency_ms:.1f}"
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def records_to_dataframe(records: List[CustomerFeatures]) -> pd.DataFrame:
    data = [r.model_dump() for r in records]
    df = pd.DataFrame(data)
    # Ensure correct dtypes expected by the pipeline
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    return df


def predict_df(df: pd.DataFrame) -> List[SinglePrediction]:
    model = artefact["model"]
    thr = artefact["threshold"]
    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= thr).astype(int)
    return [
        SinglePrediction(
            churn_probability=float(p),
            churn_prediction=int(y),
            threshold_used=float(thr),
        )
        for p, y in zip(proba, preds)
    ]


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Simple PSI for a numeric or binary feature (after binning)."""
    # Use quantiles of expected distribution
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) < 2:
        return 0.0
    expected_perc = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_perc = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    # Avoid div by zero
    expected_perc = np.clip(expected_perc, 1e-6, None)
    actual_perc = np.clip(actual_perc, 1e-6, None)
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(psi)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    loaded = artefact is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        data_hash=artefact["data_hash"] if loaded else None,
        trained_at=artefact["trained_at"] if loaded else None,
    )


@app.post(
    "/predict",
    response_model=BatchPredictionResponse,
    tags=["prediction"],
    dependencies=[Depends(verify_api_key)],
)
async def predict(request: PredictionRequest):
    """
    Score one or more customers.
    Unseen categories are handled gracefully by the OneHotEncoder (handle_unknown='ignore').
    """
    if artefact is None:
        raise HTTPException(503, "Model not loaded")
    try:
        df = records_to_dataframe(request.records)
        preds = predict_df(df)
        return BatchPredictionResponse(
            predictions=preds,
            model_data_hash=artefact["data_hash"],
            model_trained_at=artefact["trained_at"],
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}") from exc


@app.post(
    "/predict/single",
    response_model=SinglePrediction,
    tags=["prediction"],
    dependencies=[Depends(verify_api_key)],
)
async def predict_single(features: CustomerFeatures):
    """Convenience endpoint for a single record."""
    if artefact is None:
        raise HTTPException(503, "Model not loaded")
    df = records_to_dataframe([features])
    return predict_df(df)[0]


@app.post(
    "/drift",
    response_model=DriftResponse,
    tags=["ops"],
    dependencies=[Depends(verify_api_key)],
)
async def drift_check(request: PredictionRequest):
    """
    Basic population stability index (PSI) drift check.
    Compares recent batch of inputs against a sample of the training numeric distributions
    stored at training time. Flags features with PSI > 0.2 (common rule of thumb).
    Note: for production you would persist full training histograms; here we recompute
    a lightweight approximation from the loaded pipeline metadata and the incoming batch.
    """
    if artefact is None:
        raise HTTPException(503, "Model not loaded")

    # For demo: load a small reference sample from the original data
    data_path = ROOT / "data" / "telco_churn.csv"
    if not data_path.exists():
        return DriftResponse(
            overall_flag=False,
            features=[],
            message="Reference data not available inside container; drift check skipped.",
        )

    ref = pd.read_csv(data_path)
    ref["TotalCharges"] = pd.to_numeric(ref["TotalCharges"], errors="coerce")
    ref = ref.dropna(subset=["TotalCharges"])

    incoming = records_to_dataframe(request.records)
    numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    features_out: List[DriftFeature] = []
    overall = False
    for col in numeric:
        psi = population_stability_index(ref[col].values, incoming[col].values)
        flagged = psi > 0.2
        if flagged:
            overall = True
        features_out.append(DriftFeature(feature=col, psi=round(psi, 4), flagged=flagged))

    msg = "Drift detected on one or more features" if overall else "No significant drift"
    return DriftResponse(overall_flag=overall, features=features_out, message=msg)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
