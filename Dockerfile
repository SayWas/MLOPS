FROM python:3.12-slim as base

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1

WORKDIR /app

ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_DEFAULT_TIMEOUT=300

RUN pip install --upgrade pip && \
    pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./

FROM base as development
RUN poetry install --no-interaction --no-ansi
COPY . .

FROM base as production
RUN poetry install --without dev --no-interaction --no-ansi
COPY . .
RUN mkdir -p /app/data && chmod 777 /app/data

CMD ["python", "-m", "mlops"]
