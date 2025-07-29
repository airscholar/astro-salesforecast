#!/bin/bash

# Integrated Astro Dev Stop
# This script stops all services cleanly

set -e

echo "🛑 Stopping integrated development environment..."

# Get the Astro project name
PROJECT_NAME=$(astro dev info 2>/dev/null | grep "Project Name:" | awk '{print $3}' || echo "")

if [ -z "$PROJECT_NAME" ]; then
    # Fallback: use directory name
    PROJECT_NAME=$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
fi

export COMPOSE_PROJECT_NAME="$PROJECT_NAME"

# Stop additional services first
if [ -f "docker-compose.override.yml" ]; then
    echo "🐳 Stopping additional services..."
    docker-compose -p "$PROJECT_NAME" -f docker-compose.override.yml down || true
fi

# Stop Astro
echo "🌟 Stopping Astro dev environment..."
astro dev stop

echo "✅ All services stopped successfully!"