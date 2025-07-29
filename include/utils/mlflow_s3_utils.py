"""
MLflow S3 utilities to ensure artifacts are stored in MinIO
"""

import os
import mlflow
import boto3
from botocore.client import Config
import shutil
from typing import Optional, Dict, Any
import logging
from .service_discovery import get_minio_endpoint

logger = logging.getLogger(__name__)


class MLflowS3Manager:
    """
    Manager class to ensure MLflow artifacts are stored in S3/MinIO
    """
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=get_minio_endpoint(),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin'),
            config=Config(signature_version='s3v4'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        self.bucket_name = 'mlflow-artifacts'
        
    def upload_artifact_to_s3(self, local_path: str, run_id: str, artifact_path: Optional[str] = None):
        """
        Upload an artifact directly to S3
        
        Args:
            local_path: Local file path
            run_id: MLflow run ID
            artifact_path: Optional subdirectory in artifacts
        """
        try:
            # Construct S3 key
            if artifact_path:
                s3_key = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts/{artifact_path}/{os.path.basename(local_path)}"
            else:
                s3_key = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts/{os.path.basename(local_path)}"
            
            # Upload to S3
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            
            return s3_key
            
        except Exception as e:
            logger.error(f"Failed to upload artifact to S3: {e}")
            raise
    
    def log_artifact_with_s3(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact to MLflow and ensure it's uploaded to S3
        
        Args:
            local_path: Local file path
            artifact_path: Optional subdirectory in artifacts
        """
        # First log to MLflow normally
        if artifact_path:
            mlflow.log_artifact(local_path, artifact_path)
        else:
            mlflow.log_artifact(local_path)
        
        # Then ensure it's in S3
        run = mlflow.active_run()
        if run:
            self.upload_artifact_to_s3(local_path, run.info.run_id, artifact_path)
    
    def sync_mlflow_artifacts_to_s3(self, run_id: str):
        """
        Sync all artifacts from a MLflow run to S3
        
        Args:
            run_id: MLflow run ID
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Download all artifacts locally
            local_dir = f"/tmp/mlflow_sync/{run_id}"
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            
            artifacts_dir = client.download_artifacts(run_id, "", dst_path=local_dir)
            
            # Upload each file to S3
            for root, dirs, files in os.walk(artifacts_dir):
                for file in files:
                    local_file = os.path.join(root, file)
                    # Calculate relative path
                    relative_path = os.path.relpath(local_file, artifacts_dir)
                    
                    # Upload to S3
                    s3_key = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/artifacts/{relative_path}"
                    self.s3_client.upload_file(local_file, self.bucket_name, s3_key)
                    logger.info(f"Synced {relative_path} to S3")
            
            # Clean up temp directory
            shutil.rmtree(local_dir)
            
            logger.info(f"Successfully synced all artifacts for run {run_id} to S3")
            
        except Exception as e:
            logger.error(f"Failed to sync artifacts to S3: {e}")
            raise
    
    def list_s3_artifacts(self, run_id: str) -> list:
        """
        List all S3 artifacts for a run
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            List of S3 object keys
        """
        try:
            prefix = f"{run_id[:2]}/{run_id[2:4]}/{run_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to list S3 artifacts: {e}")
            return []