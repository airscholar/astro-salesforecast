"""
S3 artifact verification utilities for MLflow
"""

import os
import boto3
from botocore.client import Config
import mlflow
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def verify_s3_artifacts(run_id: str, expected_artifacts: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Verify that MLflow artifacts are stored in MinIO S3
    
    Args:
        run_id: MLflow run ID to check
        expected_artifacts: List of expected artifact paths (optional)
    
    Returns:
        Dictionary with verification results
    """
    results = {
        "success": False,
        "artifact_uri": "",
        "s3_artifacts": [],
        "missing_artifacts": [],
        "errors": []
    }
    
    try:
        # Get artifact URI from MLflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        results["artifact_uri"] = artifact_uri
        
        # Check if artifact URI is S3
        if not artifact_uri.startswith("s3://"):
            results["errors"].append(f"Artifact URI is not S3: {artifact_uri}")
            return results
        
        # Parse S3 URI
        # Format: s3://bucket/path/to/artifacts
        parts = artifact_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        
        # Create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin'),
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        # List objects in S3
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )
        
        if 'Contents' in response:
            # Extract relative paths
            s3_objects = []
            for obj in response['Contents']:
                relative_path = obj['Key'].replace(prefix, "").lstrip("/")
                if relative_path:  # Skip empty paths
                    s3_objects.append(relative_path)
            
            results["s3_artifacts"] = s3_objects
            
            # Check for expected artifacts
            if expected_artifacts:
                for expected in expected_artifacts:
                    found = False
                    for artifact in s3_objects:
                        if expected in artifact:
                            found = True
                            break
                    if not found:
                        results["missing_artifacts"].append(expected)
            
            results["success"] = len(s3_objects) > 0 and len(results["missing_artifacts"]) == 0
            
            logger.info(f"Found {len(s3_objects)} artifacts in S3 for run {run_id}")
            logger.info(f"Artifacts: {', '.join(s3_objects[:5])}...")  # Log first 5
            
        else:
            results["errors"].append("No artifacts found in S3")
            
    except Exception as e:
        results["errors"].append(str(e))
        logger.error(f"Error verifying S3 artifacts: {e}")
    
    return results


def log_s3_verification_results(results: Dict[str, any]):
    """Log S3 verification results"""
    if results["success"]:
        logger.info("✓ S3 artifact verification PASSED")
        logger.info(f"  - Artifact URI: {results['artifact_uri']}")
        logger.info(f"  - Total artifacts: {len(results['s3_artifacts'])}")
    else:
        logger.error("✗ S3 artifact verification FAILED")
        logger.error(f"  - Artifact URI: {results['artifact_uri']}")
        for error in results["errors"]:
            logger.error(f"  - Error: {error}")
        if results["missing_artifacts"]:
            logger.error(f"  - Missing artifacts: {', '.join(results['missing_artifacts'])}")