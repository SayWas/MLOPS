"""Experiment tracking utilities using ClearML."""

import os
import shutil
import subprocess
import time
from typing import Any

from clearml import Task


def init_task(
    project_name: str = "MLOps-Titanic",
    task_name: str = "Default Task",
    task_type: str = Task.TaskTypes.training,
) -> Task:
    """
    Initialize a ClearML task for experiment tracking.

    Args:
        project_name: Name of the project
        task_name: Name of the task/experiment
        task_type: Type of task (training, testing, etc.)

    Returns:
        The initialized ClearML task
    """
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
    )

    # Track DVC information if available
    try:
        dvc_status = subprocess.check_output(["dvc", "status"], text=True)
        task.upload_artifact("dvc_status", artifact_object=dvc_status)

        dvc_dag = subprocess.check_output(["dvc", "dag"], text=True)
        task.upload_artifact("dvc_dag", artifact_object=dvc_dag)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return task


def log_parameters(task: Task, params: dict[str, Any]) -> None:
    """
    Log parameters to a ClearML task.

    Args:
        task: The ClearML task
        params: Dictionary of parameters to log
    """
    task.connect(params)


def log_metrics(
    task: Task, title: str, series: str, value: float, iteration: int | None = None
) -> None:
    """
    Log metrics to a ClearML task.

    Args:
        task: The ClearML task
        title: Title of the metric group
        series: Name of the metric
        value: Value of the metric
        iteration: Optional iteration number
    """
    task.logger.report_scalar(
        title=title, series=series, value=value, iteration=iteration
    )


def log_artifact(task: Task, name: str, artifact_object: str) -> None:
    """
    Log an artifact to a ClearML task.

    Args:
        task: The ClearML task
        name: Name of the artifact
        artifact_object: Artifact object or path
    """
    task.upload_artifact(name=name, artifact_object=artifact_object)


def log_model(task: Task, model_path: str, model_name: str) -> None:
    """
    Log a model to a ClearML task.

    Args:
        task: The ClearML task
        model_path: Path to the model file
        model_name: Name to give the model
    """
    copies_dir = os.path.join(os.path.dirname(model_path), "clearml_copies")
    os.makedirs(copies_dir, exist_ok=True)

    model_copy_path = os.path.join(copies_dir, os.path.basename(model_path))
    shutil.copy2(model_path, model_copy_path)

    task.update_output_model(model_path=model_copy_path, model_name=model_name)


def get_model_from_clearml(
    project_name: str = "MLOps-Titanic",
    model_name: str | None = None,
    task_id: str | None = None,
    target_path: str | None = None,
) -> str:
    """
    Download a model from ClearML by model name or task ID.

    Args:
        project_name: ClearML project name
        model_name: Model name (if known)
        task_id: Task ID (if known)
        target_path: Path to save the model
    Returns:
        Path to the downloaded model
    """
    from clearml import Model, Task

    if task_id:
        task = Task.get_task(task_id=task_id)
        models = task.get_models().get("output", [])
        if not models:
            raise ValueError(f"Task {task_id} contains no models")
        model = Model(models[0].id)

    elif model_name:
        models = Model.query_models(
            project_name=project_name, model_name=model_name, only_published=False
        )
        if not models:
            msg = f"Model with name {model_name} not found in project {project_name}"
            raise ValueError(msg)
        model = models[0]
    else:
        raise ValueError("Must specify either model_name or task_id")

    if not target_path:
        import uuid

        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_filename = (
            f"clearml_{uuid.uuid4().hex[:8]}_{model.name.replace(' ', '_')}.joblib"
        )
        target_path = os.path.join(model_dir, model_filename)

    try:
        local_copy = model.get_local_copy()
        print(f"Model downloaded to temporary file: {local_copy}")

        shutil.copy2(local_copy, target_path)
        print(f"Model copied to: {target_path}")

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"File {target_path} was not created")

        time.sleep(1)

        return target_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise ValueError(f"Failed to download model: {e}") from e
