FROM apache/airflow:3.0.2-python3.12

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-airflow.txt /requirements-airflow.txt

USER airflow

RUN pip install --no-cache-dir -r /requirements-airflow.txt

USER airflow
