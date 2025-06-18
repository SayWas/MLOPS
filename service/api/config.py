"""Configuration for MLOps Titanic Classification API."""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

# Маппинг алгоритмов и признаков к Git ревизиям
MODEL_REVISIONS: dict[str, dict[str, dict[str, str]]] = {
    "baseline": {
        "logistic_regression": {"rev": "6675a3a7fb5b5519f13eaf98d382fa752e6f44fb"},
        "random_forest": {"rev": "1abcf93f0d2ad0277e2e4f0e0748f1f318afc956"},
    },
    "extended": {
        "logistic_regression": {"rev": "f72213bdc4f194f42d3e7e19b8f915e6a42e6c46"},
        "random_forest": {"rev": "51b200543cd9debff28b0c21950924c6b37a9395"},
    },
}


class APISettings(BaseSettings):
    """API configuration settings."""

    # API settings
    api_title: str = "MLOps Titanic Classification API"
    api_version: str = "1.0.0"
    api_description: str = (
        "API service for Titanic survival prediction with data preprocessing "
        "and model inference"
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    reload: bool = Field(default=False, description="Enable reload")
    log_level: str = Field(default="info", description="Log level")

    # Directory paths
    data_dir: str = Field(default="data", description="Data directory")
    models_dir: str = Field(default="models", description="Models directory")

    # Model settings
    default_model: str | None = Field(default=None, description="Default model")
    auto_load_models: bool = Field(default=True, description="Auto load models")
    available_models: list[str] = Field(
        default_factory=lambda: ["random_forest", "baseline"],
        description="Available model types",
    )

    # Prediction settings
    max_batch_size: int = Field(default=1000, description="Max batch size")
    default_probability_threshold: float = Field(
        default=0.5, description="Default probability threshold"
    )
    cache_predictions: bool = Field(default=False, description="Cache predictions")

    # Data preprocessing settings
    age_median: float = Field(default=28.0, description="Median age for missing values")
    fare_median: float = Field(
        default=14.4542, description="Median fare for missing values"
    )

    # Performance settings
    request_timeout: int = Field(default=300, description="Request timeout")

    # CORS settings
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS origins"
    )
    cors_credentials: bool = Field(default=True, description="CORS credentials")
    cors_methods: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS methods"
    )
    cors_headers: list[str] = Field(
        default_factory=lambda: ["*"], description="CORS headers"
    )

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"


class ModelConfig(BaseSettings):
    """Model-specific configuration."""

    model_name: str = Field(description="Model name")
    model_version: str = Field(default="1.0", description="Model version")
    model_type: str = Field(description="Model type")
    model_path: str = Field(description="Model directory path")
    default_model: str | None = Field(default=None, description="Default model to load")
    feature_columns: list[str] = Field(description="List of feature columns")

    # Preprocessing parameters
    preprocessing_params: dict[str, Any] = Field(
        default_factory=dict, description="Preprocessing parameters"
    )

    # Model performance metrics (can be loaded from files)
    accuracy: float | None = Field(default=None, description="Model accuracy")
    precision: float | None = Field(default=None, description="Model precision")
    recall: float | None = Field(default=None, description="Model recall")
    f1_score: float | None = Field(default=None, description="Model F1 score")

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = APISettings()

# Feature columns for Titanic dataset
TITANIC_FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_C",
    "Embarked_Q",
]

# Model configurations for different model types
MODEL_CONFIGS = {
    "random_forest": ModelConfig(
        model_name="random_forest",
        model_type="random_forest",
        model_path=settings.models_dir,
        default_model="model_extended.joblib",
        feature_columns=TITANIC_FEATURES,
        preprocessing_params={
            "fill_age_median": True,
            "fill_fare_median": True,
            "encode_sex": True,
            "encode_embarked": True,
        },
        accuracy=0.82,  # Example values - should be loaded from evaluation files
        precision=0.79,
        recall=0.76,
        f1_score=0.77,
    ),
    "baseline": ModelConfig(
        model_name="baseline",
        model_type="baseline",
        model_path=settings.models_dir,
        default_model="model_baseline.joblib",
        feature_columns=TITANIC_FEATURES,
        preprocessing_params={
            "fill_age_median": True,
            "fill_fare_median": True,
            "encode_sex": True,
            "encode_embarked": True,
        },
        accuracy=0.78,
        precision=0.75,
        recall=0.72,
        f1_score=0.73,
    ),
}


def get_model_config(model_name: str) -> ModelConfig | None:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name)


def get_data_path() -> Path:
    """Get the data directory path."""
    return Path(settings.data_dir)


def get_models_path() -> Path:
    """Get the models directory path."""
    return Path(settings.models_dir)


def get_available_models() -> dict[str, ModelConfig]:
    """Get all available model configurations."""
    return MODEL_CONFIGS
