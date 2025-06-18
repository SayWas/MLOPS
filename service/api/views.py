"""API endpoints for MLOps Titanic Classification API."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .schemas import (
    BatchPassengerData,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo,
    PassengerData,
    PredictionResponse,
)
from .services import TitanicClassificationService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_classification_service(request: Request) -> TitanicClassificationService:
    """Dependency to get classification service from app state."""
    service: TitanicClassificationService = request.app.state.classification_service
    return service


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: TitanicClassificationService = Depends(get_classification_service),
) -> HealthResponse:
    """Health check endpoint."""
    model_status = "loaded" if service.is_loaded else "not loaded"

    return HealthResponse(status="healthy", model_status=model_status, version="1.0.0")


@router.post("/predict", response_model=PredictionResponse)
async def predict_survival(
    passenger_data: PassengerData,
    service: TitanicClassificationService = Depends(get_classification_service),
) -> PredictionResponse:
    """Predict survival for a single passenger."""
    import time

    from .metrics import (
        ACTIVE_REQUESTS,
        ERROR_COUNTER,
        PREDICTION_COUNTER,
        PREDICTION_LATENCY,
    )

    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        if not service.is_loaded:
            ERROR_COUNTER.labels(error_type="ModelNotLoaded", endpoint="predict").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        prediction = service.predict_single(passenger_data)

        # Track successful prediction
        PREDICTION_COUNTER.labels(
            model_name=prediction.model_version, prediction_result="success"
        ).inc()

        # Track latency
        duration = time.time() - start_time
        PREDICTION_LATENCY.labels(model_name=prediction.model_version).observe(duration)

        return prediction

    except RuntimeError as e:
        ERROR_COUNTER.labels(error_type="RuntimeError", endpoint="predict").inc()
        PREDICTION_COUNTER.labels(
            model_name=service.model_name or "unknown", prediction_result="error"
        ).inc()
        logger.error(f"Runtime error in prediction: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__, endpoint="predict").inc()
        PREDICTION_COUNTER.labels(
            model_name=service.model_name or "unknown", prediction_result="error"
        ).inc()
        logger.error(f"Unexpected error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        ACTIVE_REQUESTS.dec()


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_survival_batch(
    batch_data: BatchPassengerData,
    service: TitanicClassificationService = Depends(get_classification_service),
) -> BatchPredictionResponse:
    """Predict survival for multiple passengers."""
    try:
        if not service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        predictions = service.predict_batch(batch_data.passengers)

        return BatchPredictionResponse(
            predictions=predictions, total_count=len(predictions)
        )

    except RuntimeError as e:
        logger.error(f"Runtime error in batch prediction: {e}")
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info(
    service: TitanicClassificationService = Depends(get_classification_service),
) -> ModelInfo:
    """Get information about the currently loaded model."""
    model_info = service.get_model_info()
    return ModelInfo(**model_info)


@router.get("/models/available")
async def get_available_models(
    service: TitanicClassificationService = Depends(get_classification_service),
) -> dict:
    """Get list of available models."""
    models = service.get_available_models()
    return {"available_models": models}


@router.get("/models/clearml/available")
async def get_available_clearml_models(
    service: TitanicClassificationService = Depends(get_classification_service),
) -> dict:
    """Get list of available ClearML model IDs."""
    clearml_ids = service.get_available_clearml_models()
    return {"available_clearml_models": clearml_ids}


@router.post("/model/load/clearml")
async def load_clearml_model(
    clearml_id: str,
    service: TitanicClassificationService = Depends(get_classification_service),
) -> dict:
    """Load ClearML model by ID."""
    try:
        success = service.load_clearml_model(clearml_id)

        if success:
            return {
                "message": f"ClearML model {clearml_id} loaded successfully",
                "clearml_id": clearml_id,
                "model_info": service.get_model_info(),
                "status": "loaded",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to load ClearML model with ID '{clearml_id}'",
            )

    except Exception as e:
        logger.error(f"Error loading ClearML model: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/model/load")
async def load_model_from_revision(
    algorithm: str,
    features: str,
    service: TitanicClassificationService = Depends(get_classification_service),
) -> dict:
    """Load model from the correct Git revision based on algorithm and features."""
    try:
        success = service.load_model_from_revision(algorithm, features)

        if success:
            return {
                "message": (
                    f"Model {algorithm} with {features} features loaded successfully"
                ),
                "algorithm": algorithm,
                "features": features,
                "model_info": service.get_model_info(),
                "status": "loaded",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Failed to load model with algorithm '{algorithm}' "
                    f"and features '{features}'"
                ),
            )

    except Exception as e:
        logger.error(f"Error loading model from revision: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics")
async def get_metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint."""
    from .metrics import get_metrics

    metrics_data = get_metrics()
    return PlainTextResponse(content=metrics_data, media_type="text/plain")


@router.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "MLOps Titanic Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "endpoints": {
            "model_loading": {
                "load_by_revision": "/model/load",
                "load_clearml": "/model/load/clearml",
            },
            "models": {
                "available": "/models/available",
                "clearml_available": "/models/clearml/available",
            },
            "prediction": {"single": "/predict", "batch": "/predict/batch"},
        },
    }
