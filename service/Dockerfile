FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies including Git for DVC
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file first for better caching
COPY service/requirements.txt ./

# Install Python dependencies including DVC for Git revision loading
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files for API
COPY service/ ./service/
COPY run_api.py ./
COPY mlops/ ./mlops/
COPY data/ ./data/
COPY models/ ./models/
COPY .git/ ./.git/
COPY .dvc/ ./.dvc/

# Create directories and set permissions
RUN mkdir -p /app/logs && \
    useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app && \
    git config --global --add safe.directory /app

USER apiuser

# Set Git safe directory for current user too
RUN git config --global --add safe.directory /app

# Expose port
EXPOSE 8000

# Environment variables (will be overridden by docker-compose)
ENV PYTHONPATH=/app
ENV MODEL_TYPE=random_forest
ENV DEFAULT_ALGORITHM=random_forest
ENV DEFAULT_FEATURES=extended

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "run_api.py"]
