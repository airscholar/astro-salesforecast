#!/usr/bin/env python3
"""
Script to test MLflow connection from within Airflow container
"""

import os
import sys
import requests
from urllib.parse import urlparse


def test_mlflow_connection():
    """Test connection to MLflow server"""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001')
    
    print(f"Testing connection to MLflow at: {mlflow_uri}")
    
    try:
        # Try to reach MLflow health endpoint
        response = requests.get(f"{mlflow_uri}/health", timeout=5)
        if response.status_code == 200:
            print("✓ MLflow server is reachable")
        else:
            print(f"✗ MLflow returned status code: {response.status_code}")
            
        # Try to get experiments
        response = requests.get(f"{mlflow_uri}/api/2.0/mlflow/experiments/list", timeout=5)
        if response.status_code == 200:
            print("✓ MLflow API is accessible")
            experiments = response.json().get('experiments', [])
            print(f"  Found {len(experiments)} experiments")
        else:
            print(f"✗ MLflow API returned status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError as e:
        print(f"✗ Cannot connect to MLflow: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure MLflow service is running: docker-compose ps mlflow")
        print("2. Check if services are on the same network: docker network ls")
        print("3. Verify the network name in docker-compose.override.yml")
        print("4. Restart services: astro dev restart")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def test_minio_connection():
    """Test connection to MinIO"""
    minio_url = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
    
    print(f"\nTesting connection to MinIO at: {minio_url}")
    
    try:
        response = requests.get(f"{minio_url}/minio/health/live", timeout=5)
        if response.status_code == 200:
            print("✓ MinIO server is reachable")
        else:
            print(f"✗ MinIO returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"✗ Cannot connect to MinIO: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


if __name__ == "__main__":
    print("MLflow Connection Test")
    print("=" * 50)
    
    # Show environment variables
    print("Environment variables:")
    print(f"  MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'Not set')}")
    print(f"  MLFLOW_S3_ENDPOINT_URL: {os.getenv('MLFLOW_S3_ENDPOINT_URL', 'Not set')}")
    print(f"  AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID', 'Not set')}")
    print()
    
    test_mlflow_connection()
    test_minio_connection()