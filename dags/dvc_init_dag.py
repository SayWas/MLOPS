"""DAG for initializing and configuring DVC."""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from dags.dvc_helpers import check_dvc_initialized, initialize_dvc, setup_dvc_remote

PROJECT_PATH = "/app"
DVC_REMOTE_NAME = "myremote"
DVC_REMOTE_URL = "/app/dvc-store"

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def init_dvc_task(**kwargs: dict[str, Any]) -> str:
    """Initializes DVC in the project."""
    if check_dvc_initialized(PROJECT_PATH):
        print("DVC already initialized")
        return "DVC already initialized"

    success, message = initialize_dvc(PROJECT_PATH)
    if not success:
        raise Exception(message)

    print(message)
    return str(message)


def setup_dvc_remote_task(**kwargs: dict[str, Any]) -> str:
    """Configures the DVC remote storage."""
    import os

    remote_dir = DVC_REMOTE_URL
    if not os.path.exists(remote_dir):
        os.makedirs(remote_dir)

    success, message = setup_dvc_remote(PROJECT_PATH, DVC_REMOTE_NAME, DVC_REMOTE_URL)
    if not success:
        raise Exception(message)

    print(message)
    return str(message)


with DAG(
    "dvc_init",
    default_args=default_args,
    description="Initialize and configure DVC",
    schedule=None,  # Only triggered manually
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["dvc", "setup"],
) as dag:
    # Task 1: DVC initialization
    init_dvc = PythonOperator(
        task_id="init_dvc",
        python_callable=init_dvc_task,
    )

    # Task 2: Configure DVC remote
    setup_remote = PythonOperator(
        task_id="setup_dvc_remote",
        python_callable=setup_dvc_remote_task,
    )

    init_dvc >> setup_remote
