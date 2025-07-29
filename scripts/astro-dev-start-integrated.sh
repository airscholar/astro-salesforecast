#!/bin/bash

# Integrated Astro Dev Start
# This script starts Astro with all additional services seamlessly

set -e

echo "🚀 Starting integrated Astro development environment..."

# Check if astro CLI is available
if ! command -v astro &> /dev/null; then
    echo "❌ Astro CLI not found. Please install it first."
    echo "Visit: https://docs.astronomer.io/astro/cli/install-cli"
    exit 1
fi

# Get the Astro project name dynamically
PROJECT_NAME=$(astro dev info 2>/dev/null | grep "Project Name:" | awk '{print $3}' || echo "")

if [ -z "$PROJECT_NAME" ]; then
    # Fallback: use directory name
    PROJECT_NAME=$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
fi

echo "📦 Project name: $PROJECT_NAME"

# Export COMPOSE_PROJECT_NAME for docker-compose
export COMPOSE_PROJECT_NAME="$PROJECT_NAME"

# Start Astro first
echo "🌟 Starting Astro dev environment..."
astro dev start

# Give Astro a moment to create the network
sleep 5

# Verify network exists
if docker network ls | grep -q "${PROJECT_NAME}_airflow"; then
    echo "✅ Airflow network is ready"
    
    # Start additional services using docker-compose with the override file
    echo "🐳 Starting additional services..."
    docker-compose -p "$PROJECT_NAME" -f docker-compose.override.yml up -d
    
    # Wait for services to be healthy
    echo "⏳ Waiting for services to be healthy..."
    
    # Check MLflow
    attempts=0
    while [ $attempts -lt 30 ]; do
        if curl -s http://localhost:5001/health >/dev/null 2>&1; then
            echo "✅ MLflow is ready"
            break
        fi
        attempts=$((attempts + 1))
        sleep 2
    done
    
    # Check MinIO
    attempts=0
    while [ $attempts -lt 30 ]; do
        if curl -s http://localhost:9000/minio/health/live >/dev/null 2>&1; then
            echo "✅ MinIO is ready"
            break
        fi
        attempts=$((attempts + 1))
        sleep 2
    done
    
    echo ""
    echo "📊 Service Status:"
    docker-compose -p "$PROJECT_NAME" -f docker-compose.override.yml ps
    
    echo ""
    echo "✅ All services started successfully!"
    echo ""
    echo "🌐 Available services:"
    echo "  - Airflow UI: http://localhost:8080 (admin/admin)"
    echo "  - MLflow UI: http://localhost:5001"
    echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
    echo "  - Streamlit UI: http://localhost:8501"
    echo ""
    echo "💡 Tips:"
    echo "  - To view logs: docker-compose -p $PROJECT_NAME -f docker-compose.override.yml logs -f [service]"
    echo "  - To stop all: ./scripts/astro-dev-stop-integrated.sh"
    echo "  - To restart a service: docker-compose -p $PROJECT_NAME -f docker-compose.override.yml restart [service]"
else
    echo "❌ Failed to find Airflow network. Please check Astro startup."
    exit 1
fi