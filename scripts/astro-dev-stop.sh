#!/bin/bash

# Astro Dev Stop with Docker Compose Override
# This script ensures all services stop correctly

set -e

echo "🛑 Stopping all services..."

# Stop additional services first
if [ -f "docker-compose.override.yml" ]; then
    echo "🐳 Stopping additional services..."
    docker-compose -f docker-compose.override.yml down || true
fi

# Stop Astro
echo "🌟 Stopping Astro dev environment..."
astro dev stop

echo "✅ All services stopped successfully!"