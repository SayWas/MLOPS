FROM python:3.12-slim AS base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_DEFAULT_TIMEOUT=300

RUN pip install --upgrade pip && \
    pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock README.md ./

FROM base AS development
COPY . .
RUN poetry install --no-interaction --no-ansi

FROM base AS production
COPY . .
RUN poetry install --without dev --no-interaction --no-ansi
RUN mkdir -p /app/data && chmod 777 /app/data

CMD ["python", "-m", "mlops"]
