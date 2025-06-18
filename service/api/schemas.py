"""Pydantic schemas for MLOps Titanic Classification API."""

from pydantic import BaseModel, Field


class PassengerData(BaseModel):
    """Schema for passenger data input."""

    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Sex: int = Field(..., ge=0, le=1, description="Sex (0 = female, 1 = male)")
    Age: float | None = Field(None, ge=0, le=120, description="Age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float | None = Field(None, ge=0, description="Passenger fare")
    Embarked_C: int = Field(
        0, ge=0, le=1, description="Embarked at Cherbourg (1 if yes, 0 if no)"
    )
    Embarked_Q: int = Field(
        0, ge=0, le=1, description="Embarked at Queenstown (1 if yes, 0 if no)"
    )

    class Config:
        """Pydantic configuration for PassengerData."""

        schema_extra = {
            "example": {
                "Pclass": 3,
                "Sex": 1,
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked_C": 0,
                "Embarked_Q": 0,
            }
        }


class BatchPassengerData(BaseModel):
    """Schema for batch prediction input."""

    passengers: list[PassengerData] = Field(..., description="List of passenger data")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    prediction: int = Field(
        ..., description="Survival prediction (0 = did not survive, 1 = survived)"
    )
    probability: float = Field(..., description="Probability of survival")
    model_version: str = Field(..., description="Model version used for prediction")

    class Config:
        """Pydantic configuration for PredictionResponse."""

        schema_extra = {
            "example": {
                "prediction": 0,
                "probability": 0.23,
                "model_version": "model_extended.joblib",
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""

    predictions: list[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    total_count: int = Field(..., description="Total number of predictions")


class ModelInfo(BaseModel):
    """Schema for model information."""

    model_name: str = Field(..., description="Name of the loaded model")
    model_type: str = Field(..., description="Type of the model")
    feature_columns: list[str] = Field(..., description="List of feature columns")
    is_loaded: bool = Field(..., description="Whether model is loaded")


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str = Field(..., description="Service status")
    model_status: str = Field(..., description="Model status")
    version: str = Field(..., description="API version")
