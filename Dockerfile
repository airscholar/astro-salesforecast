FROM astrocrpublic.azurecr.io/runtime:3.0-5

# Install additional system dependencies for ML libraries
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

USER astro

# Set environment variables for MLflow and MinIO
ENV MLFLOW_TRACKING_URI=http://mlflow:5001
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV AWS_DEFAULT_REGION=us-east-1
