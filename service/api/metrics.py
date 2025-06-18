"""Prometheus metrics for MLOps Titanic Classification API."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, generate_latest

PREDICTION_COUNTER = Counter(
    "titanic_predictions_total",
    "Total number of predictions made",
    ["model_name", "prediction_result"],
)

PREDICTION_LATENCY = Histogram(
    "titanic_prediction_duration_seconds", "Time spent on prediction", ["model_name"]
)

MODEL_ACCURACY = Gauge(
    "titanic_model_accuracy", "Current model accuracy", ["model_name"]
)

MODEL_LOAD_TIME = Histogram(
    "titanic_model_load_duration_seconds", "Time spent loading model", ["model_name"]
)

ACTIVE_REQUESTS = Gauge("titanic_active_requests", "Number of active requests")

ERROR_COUNTER = Counter(
    "titanic_errors_total", "Total number of errors", ["error_type", "endpoint"]
)


def track_prediction_metrics(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to track prediction metrics."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        model_name = "unknown"

        try:
            ACTIVE_REQUESTS.inc()
            result = await func(*args, **kwargs)

            if hasattr(result, "model_version"):
                model_name = result.model_version
            elif hasattr(result, "predictions") and result.predictions:
                model_name = result.predictions[0].model_version

            prediction_result = "success"
            PREDICTION_COUNTER.labels(
                model_name=model_name, prediction_result=prediction_result
            ).inc()

            duration = time.time() - start_time
            PREDICTION_LATENCY.labels(model_name=model_name).observe(duration)

            return result

        except Exception as e:
            ERROR_COUNTER.labels(
                error_type=type(e).__name__, endpoint=func.__name__
            ).inc()

            PREDICTION_COUNTER.labels(
                model_name=model_name, prediction_result="error"
            ).inc()

            raise

        finally:
            ACTIVE_REQUESTS.dec()

    return wrapper


def track_model_load_time(model_name: str, duration: float) -> None:
    """Track model loading time."""
    MODEL_LOAD_TIME.labels(model_name=model_name).observe(duration)


def update_model_accuracy(model_name: str, accuracy: float) -> None:
    """Update model accuracy metric."""
    MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)


def get_metrics() -> bytes:
    """Get all metrics in Prometheus format."""
    return generate_latest()
