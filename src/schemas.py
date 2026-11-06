"""Pydantic schemas for request/response validation."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: str = Field(..., examples=["0"])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, le=100, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])
    TotalCharges: float = Field(..., ge=0, examples=[845.0])

    @field_validator("SeniorCitizen")
    @classmethod
    def senior_as_str(cls, v):
        return str(v)


class PredictionRequest(BaseModel):
    records: List[CustomerFeatures] = Field(..., min_length=1, max_length=100)


class SinglePrediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    threshold_used: float


class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    model_data_hash: str
    model_trained_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_hash: Optional[str] = None
    trained_at: Optional[str] = None


class DriftFeature(BaseModel):
    feature: str
    psi: float
    flagged: bool


class DriftResponse(BaseModel):
    overall_flag: bool
    features: List[DriftFeature]
    message: str
