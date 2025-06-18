"""FastAPI application for MLOps Titanic Classification API."""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_model_config
from .services import TitanicClassificationService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    # Startup
    logging.info("Starting up MLOps Titanic Classification API")

    # Get model configuration
    model_config = get_model_config(app.state.model_type)
    if model_config is None:
        logging.error(f"Invalid model type: {app.state.model_type}")
        raise ValueError(f"Invalid model type: {app.state.model_type}")

    # Initialize classification service
    app.state.classification_service = TitanicClassificationService(model_config)

    # Load default model based on environment variables
    try:
        # Получаем алгоритм и признаки из переменных окружения
        default_algorithm = os.getenv("DEFAULT_ALGORITHM")
        default_features = os.getenv("DEFAULT_FEATURES")

        if default_algorithm and default_features:
            # Загружаем модель из Git ревизии на основе env vars
            success = app.state.classification_service.load_model_from_revision(
                default_algorithm, default_features
            )
            logging.info(
                f"Loading model from env vars: {default_algorithm} + {default_features}"
            )
        elif app.state.model_type == "random_forest":
            # По умолчанию загружаем Random Forest с extended признаками
            success = app.state.classification_service.load_model_from_revision(
                "random_forest", "extended"
            )
        elif app.state.model_type == "baseline":
            # Для baseline загружаем Logistic Regression с baseline признаками
            success = app.state.classification_service.load_model_from_revision(
                "logistic_regression", "baseline"
            )
        else:
            # Fallback: пытаемся загрузить обычную модель
            default_model = model_config.default_model
            success = (
                app.state.classification_service.load_model(default_model)
                if default_model
                else False
            )

        if success:
            logging.info(
                f"Successfully loaded default model for type: {app.state.model_type}"
            )
        else:
            logging.warning(
                f"Failed to load default model for type: {app.state.model_type}"
            )

    except Exception as e:
        logging.error(f"Error loading default model: {e}")
        print(f"Error loading default model: {e}")

    yield

    # Shutdown
    logging.info("Shutting down MLOps Titanic Classification API")


def create_app(model_type: str = "random_forest") -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="MLOps Titanic Classification API",
        description="Titanic Survival Prediction API for MLOps course",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Store model type in app state
    app.state.model_type = model_type

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from .views import router

    app.include_router(router)

    return app


# Get model type from environment
model_type = os.getenv("MODEL_TYPE", "random_forest")
app = create_app(model_type)
