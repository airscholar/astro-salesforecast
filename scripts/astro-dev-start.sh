#!/bin/bash

# Astro Dev Start with Docker Compose Override
# This script ensures all services start correctly with Astro CLI

set -e

echo "🚀 Starting Astro development environment with additional services..."

# Check if astro CLI is available
if ! command -v astro &> /dev/null; then
    echo "❌ Astro CLI not found. Please install it first."
    echo "Visit: https://docs.astronomer.io/astro/cli/install-cli"
    exit 1
fi

# Function to wait for network creation
wait_for_network() {
    local network_name=$1
    local max_attempts=30
    local attempt=0
    
    echo "⏳ Waiting for Airflow network to be created..."
    
    while [ $attempt -lt $max_attempts ]; do
        if docker network ls | grep -q "$network_name"; then
            echo "✅ Network $network_name is ready"
            return 0
        fi
        
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo "❌ Network $network_name was not created in time"
    return 1
}

# Function to get the project name from Astro
get_project_name() {
    # Extract project name from .astro/config.yaml or use directory name
    if [ -f ".astro/config.yaml" ]; then
        project_name=$(grep "project_name:" .astro/config.yaml | cut -d' ' -f2 || echo "")
    fi
    
    if [ -z "$project_name" ]; then
        # Use directory name as fallback
        project_name=$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
    fi
    
    echo "$project_name"
}

# Start Astro in background
echo "🌟 Starting Astro dev environment..."
astro dev start &
ASTRO_PID=$!

# Get the project name
PROJECT_NAME=$(get_project_name)
NETWORK_NAME="${PROJECT_NAME}_airflow"

# Wait for the Airflow network to be created
if wait_for_network "$NETWORK_NAME"; then
    # Update docker-compose.override.yml with the correct network name
    echo "📝 Updating docker-compose.override.yml with network: $NETWORK_NAME"
    
    # Create a temporary file with updated network
    sed "s/astro-sales-forecast_eaa509_airflow/$NETWORK_NAME/g" docker-compose.override.yml > docker-compose.override.yml.tmp
    mv docker-compose.override.yml.tmp docker-compose.override.yml
    
    # Start additional services
    echo "🐳 Starting additional services (MLflow, MinIO, Redis, Streamlit)..."
    docker-compose -f docker-compose.override.yml up -d
    
    # Wait for services to be healthy
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
    
    # Check service status
    echo "📊 Service Status:"
    docker-compose -f docker-compose.override.yml ps
    
    echo ""
    echo "✅ All services started successfully!"
    echo ""
    echo "🌐 Available services:"
    echo "  - Airflow UI: http://localhost:8080"
    echo "  - MLflow UI: http://localhost:5001"
    echo "  - MinIO Console: http://localhost:9001"
    echo "  - Streamlit UI: http://localhost:8501"
    echo ""
    echo "To stop all services, run: ./scripts/astro-dev-stop.sh"
else
    echo "❌ Failed to start Astro services properly"
    kill $ASTRO_PID 2>/dev/null || true
    exit 1
fi

# Wait for Astro process
wait $ASTRO_PID