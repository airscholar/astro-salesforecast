#!/bin/bash

echo "Restarting Airflow with local MLflow configuration..."

# Stop services
echo "Stopping services..."
astro dev stop

# Clear Python cache
echo "Clearing Python cache..."
find /Users/airscholar/PycharmProjects/Astro-SalesForecast -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /Users/airscholar/PycharmProjects/Astro-SalesForecast -name "*.pyc" -delete 2>/dev/null || true

# Export environment variable
export MLFLOW_TRACKING_URI="file:///tmp/mlruns"

# Start services
echo "Starting services with local MLflow..."
astro dev start

echo "Services restarted. MLflow will use local file tracking at /tmp/mlruns"
echo "You can now run the DAG without MLflow server dependency."