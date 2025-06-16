"""DAG for training and evaluating Titanic models with DVC integration."""

import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator

from dags.dvc_helpers import push_to_dvc_remote, track_file_with_dvc

PROJECT_PATH = "/app"
MODELS_DIR = os.path.join(PROJECT_PATH, "models")
DATA_PROCESSED_DIR = os.path.join(PROJECT_PATH, "data/processed")

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def prepare_directories(**kwargs: dict[str, Any]) -> bool:
    """Creates necessary directories for models and processed data."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    print(f"Models directory created: {MODELS_DIR}")
    print(f"Data processed directory created: {DATA_PROCESSED_DIR}")
    return True


def track_files_with_dvc(**kwargs: dict[str, Any]) -> bool:
    """Adds files to DVC tracking."""
    files_to_track = kwargs.get(
        "files",
        [
            "data/processed/train_features.csv",
            "data/processed/train_features_extended.csv",
            "models/logreg_baseline.joblib",
            "models/logreg_extended.joblib",
            "models/rf_baseline.joblib",
            "models/rf_extended.joblib",
            "models/evaluation_results.txt",
        ],
    )
    for file_path in files_to_track:
        full_path = os.path.join(PROJECT_PATH, file_path)
        if not os.path.exists(full_path):
            print(f"Warning: File {full_path} does not exist, skipping")
            continue
        success, message = track_file_with_dvc(PROJECT_PATH, file_path)
        print(message)
        if not success:
            print(f"Warning: Failed to track {file_path}: {message}")
    return True


def push_to_remote(**kwargs: dict[str, Any]) -> bool:
    """Pushes all data and models to DVC remote storage."""
    success, message = push_to_dvc_remote(PROJECT_PATH)
    print(message)
    if not success:
        print(f"Warning: Failed to push to remote: {message}")
    return True


with DAG(
    "titanic_training_pipeline",
    default_args=default_args,
    description=(
        "Training and evaluation pipeline for Titanic models " "with DVC integration"
    ),
    schedule=None,  # Run manually or set a schedule
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["mlops", "titanic", "dvc"],
) as dag:
    # Step 0: Pull latest data & models from DVC
    dvc_pull = BashOperator(
        task_id="pull_dvc_data",
        bash_command=f"cd {PROJECT_PATH} && dvc pull",
    )

    # Step 1: Prepare environment
    prepare_env = PythonOperator(
        task_id="prepare_environment",
        python_callable=prepare_directories,
    )

    # Step 2: Data preprocessing
    data_preparation = BashOperator(
        task_id="data_preparation",
        bash_command=f"cd {PROJECT_PATH} && python -m mlops.scripts.data_prep",
    )

    # Step 3: Track processed data with DVC
    track_data = PythonOperator(
        task_id="track_data_with_dvc",
        python_callable=track_files_with_dvc,
        op_kwargs={
            "files": [
                "data/processed/train_features.csv",
                "data/processed/train_features_extended.csv",
            ]
        },
    )

    # Step 4: Train log reg
    train_models = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_PATH} && "
            "python -m mlops.scripts.train_logreg --features baseline"
        ),
    )

    # Step 5: Track models with DVC
    track_models = PythonOperator(
        task_id="track_model_with_dvc",
        python_callable=track_files_with_dvc,
        op_kwargs={
            "files": [
                "models/model_baseline.joblib",
            ]
        },
    )

    # Step 6: Evaluate models
    evaluate_models = BashOperator(
        task_id="evaluate_model",
        bash_command=f"cd {PROJECT_PATH} && python -m mlops.scripts.eval_models",
    )

    # Step 7: Track evaluation results with DVC
    track_evaluation = PythonOperator(
        task_id="track_evaluation_with_dvc",
        python_callable=track_files_with_dvc,
        op_kwargs={"files": ["models/evaluation_results.txt"]},
    )

    # Step 8: Push everything to DVC remote
    push_dvc = PythonOperator(
        task_id="push_to_dvc_remote",
        python_callable=push_to_remote,
    )

    # Set dependencies so dvc_pull runs first
    (
        dvc_pull
        >> prepare_env
        >> data_preparation
        >> track_data
        >> train_models
        >> track_models
        >> evaluate_models
        >> track_evaluation
        >> push_dvc
    )
