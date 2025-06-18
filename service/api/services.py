"""Services for MLOps Titanic Classification API."""

import logging
import os
from typing import Any

import dvc.api
import joblib
import numpy as np
import polars as pl

from .config import MODEL_REVISIONS, ModelConfig
from .schemas import PassengerData, PredictionResponse

logger = logging.getLogger(__name__)


class TitanicClassificationService:
    """Service for Titanic survival classification."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the classification service with configuration."""
        self.config = config
        self.model: Any | None = None
        self.model_name: str | None = None
        self.is_loaded = False

        # Median values for missing data (calculated from training data)
        self.age_median = 28.0
        self.fare_median = 14.4542

    def load_model(self, model_name: str) -> bool:
        """Load a model from the specified path."""
        try:
            model_path = os.path.join(self.config.model_path, model_name)

            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False

            self.model = joblib.load(model_path)
            self.model_name = model_name
            self.is_loaded = True

            logger.info(f"Successfully loaded model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self.model = None
            self.model_name = None
            self.is_loaded = False
            return False

    def load_model_from_revision(self, algorithm: str, features: str) -> bool:
        """Load model from the correct Git revision based on algorithm and features."""
        try:
            if features not in MODEL_REVISIONS:
                logger.error(
                    f"Unknown features: {features}. "
                    f"Available: {list(MODEL_REVISIONS.keys())}"
                )
                return False

            if algorithm not in MODEL_REVISIONS[features]:
                available = list(MODEL_REVISIONS[features].keys())
                logger.error(f"Unknown algorithm: {algorithm}. Available: {available}")
                return False

            revision = MODEL_REVISIONS[features][algorithm]["rev"]

            # Определяем путь к модели на основе features
            model_paths = {
                "baseline": "models/model_baseline.joblib",
                "extended": "models/model_extended.joblib",
            }
            model_path = model_paths[features]

            logger.info(
                f"Loading {algorithm} model with {features} features "
                f"from revision {revision[:8]}..."
            )

            # Загружаем модель из DVC с указанной ревизией
            with dvc.api.open(model_path, repo=".", rev=revision, mode="rb") as f:
                self.model = joblib.load(f)

            self.model_name = f"{algorithm}_{features}@{revision[:8]}"
            self.is_loaded = True

            logger.info(f"Successfully loaded model: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading model from revision: {e}")
            self.model = None
            self.model_name = None
            self.is_loaded = False
            return False

    def preprocess_data(self, data: PassengerData) -> np.ndarray:
        """Preprocess passenger data for prediction."""
        # Convert to dictionary first
        data_dict = data.dict()

        # Handle missing values
        if data_dict["Age"] is None:
            data_dict["Age"] = self.age_median

        if data_dict["Fare"] is None:
            data_dict["Fare"] = self.fare_median

        # Create Polars DataFrame
        df = pl.DataFrame([data_dict])

        # Generate extended features if model expects them
        # Определяем тип модели по имени
        is_extended_model = (
            hasattr(self, "model_name")
            and self.model_name
            and ("extended" in self.model_name.lower() or self._is_extended_model())
        )

        if is_extended_model:
            # Добавляем дополнительные признаки для extended модели
            df = df.with_columns(
                [
                    # FamilySize = SibSp + Parch + 1
                    (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize"),
                    # IsAlone = 1 if FamilySize == 1, else 0
                    ((pl.col("SibSp") + pl.col("Parch") + 1) == 1)
                    .cast(pl.Int8)
                    .alias("IsAlone"),
                    # NameLength - используем среднюю длину имени (приблизительно)
                    pl.lit(25).alias("NameLength"),  # Средняя длина имени
                ]
            )

            # Extended feature columns
            feature_columns = [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked_C",
                "Embarked_Q",
                "FamilySize",
                "IsAlone",
                "NameLength",
            ]
        else:
            # Baseline feature columns
            feature_columns = [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked_C",
                "Embarked_Q",
            ]

        # Select only the required columns and fill missing ones with 0
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            # Add missing columns with default values
            for col in missing_cols:
                df = df.with_columns(pl.lit(0).alias(col))

        # Select features in the correct order and convert to numpy
        features_array = df.select(feature_columns).to_numpy()

        return features_array

    def _is_extended_model(self) -> bool:
        """Determine if model is extended based on expected feature count."""
        if self.model and hasattr(self.model, "n_features_in_"):
            return bool(self.model.n_features_in_ > 8)
        return False

    def predict_single(self, passenger_data: PassengerData) -> PredictionResponse:
        """Make prediction for a single passenger."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Preprocess data
        processed_data = self.preprocess_data(passenger_data)

        # Make prediction
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0][
            1
        ]  # Probability of survival

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_version=self.model_name or "unknown",
        )

    def predict_batch(
        self, passengers_data: list[PassengerData]
    ) -> list[PredictionResponse]:
        """Make predictions for multiple passengers."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        results = []

        for passenger_data in passengers_data:
            try:
                prediction_result = self.predict_single(passenger_data)
                results.append(prediction_result)
            except Exception as e:
                logger.error(f"Error predicting for passenger: {e}")
                # Add a default response for failed predictions
                results.append(
                    PredictionResponse(
                        prediction=0,
                        probability=0.0,
                        model_version=self.model_name or "unknown",
                    )
                )

        return results

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name or "None",
            "model_type": self.config.model_type,
            "feature_columns": self.config.feature_columns,
            "is_loaded": self.is_loaded,
        }

    def load_clearml_model(self, clearml_id: str) -> bool:
        """Load ClearML model by ID from current Git revision."""
        try:
            # Construct model filename
            model_filename = (
                f"clearml_{clearml_id}_Random_Forest_Extended_Features.joblib"
            )
            model_path = f"models/{model_filename}"

            logger.info(f"Loading ClearML model: {model_filename}")

            # Load model from current revision
            # (since all ClearML models are in current commit)
            with dvc.api.open(model_path, repo=".", mode="rb") as f:
                self.model = joblib.load(f)

            self.model_name = f"clearml_{clearml_id}"
            self.is_loaded = True

            logger.info(f"Successfully loaded ClearML model: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading ClearML model {clearml_id}: {e}")
            self.model = None
            self.model_name = None
            self.is_loaded = False
            return False

    def get_available_models(self) -> list[str]:
        """Get list of available model files."""
        try:
            model_files = []
            for file in os.listdir(self.config.model_path):
                if file.endswith(".joblib"):
                    model_files.append(file)
            return model_files
        except Exception as e:
            logger.error(f"Error listing model files: {e}")
            return []

    def get_available_clearml_models(self) -> list[str]:
        """Get list of available ClearML model IDs."""
        try:
            clearml_ids = []
            for file in os.listdir(self.config.model_path):
                if file.startswith("clearml_") and file.endswith(".joblib"):
                    # Extract ID from filename:
                    # clearml_2b1d16b9_Random_Forest_Extended_Features.joblib
                    parts = file.split("_")
                    if len(parts) >= 2:
                        clearml_id = parts[1]
                        clearml_ids.append(clearml_id)
            return clearml_ids
        except Exception as e:
            logger.error(f"Error listing ClearML model files: {e}")
            return []
