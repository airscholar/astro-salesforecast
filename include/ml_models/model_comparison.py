"""
Model Comparison and Promotion System
Compares new models against production models and promotes only if performance improves
"""

import mlflow
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class ModelComparison:
    """Compare and promote models based on performance metrics"""
    
    def __init__(self, mlflow_uri: str = "http://localhost:5001"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()
        
    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get current production model details"""
        try:
            # Get the latest version with 'Production' stage
            versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'",
                order_by=["version_number DESC"]
            )
            
            for version in versions:
                if version.current_stage == "Production":
                    # Get run details
                    run = self.client.get_run(version.run_id)
                    metrics = run.data.metrics
                    
                    return {
                        "version": version.version,
                        "run_id": version.run_id,
                        "metrics": metrics,
                        "model_uri": f"models:/{model_name}/{version.version}"
                    }
                    
        except Exception as e:
            logger.warning(f"No production model found for {model_name}: {e}")
            
        return None
    
    def get_candidate_model(self, run_id: str, model_name: str) -> Dict[str, Any]:
        """Get candidate model details from a specific run"""
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        
        return {
            "run_id": run_id,
            "metrics": metrics,
            "model_uri": f"runs:/{run_id}/{model_name}"
        }
    
    def compare_metrics(
        self, 
        prod_metrics: Dict[str, float], 
        candidate_metrics: Dict[str, float],
        comparison_metrics: list = None,
        improvement_threshold: float = 0.01
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compare production and candidate model metrics
        
        Args:
            prod_metrics: Production model metrics
            candidate_metrics: Candidate model metrics
            comparison_metrics: List of metrics to compare (default: RMSE, MAE, R2)
            improvement_threshold: Minimum relative improvement required (default: 1%)
            
        Returns:
            Tuple of (should_promote, comparison_details)
        """
        if comparison_metrics is None:
            # Default metrics - lower is better for RMSE/MAE, higher for R2
            comparison_metrics = [
                ("test_rmse", "lower"),
                ("test_mae", "lower"),
                ("test_r2", "higher")
            ]
        
        comparison_results = {}
        improvements = []
        
        for metric_name, direction in comparison_metrics:
            if metric_name not in prod_metrics or metric_name not in candidate_metrics:
                logger.warning(f"Metric {metric_name} not found in both models")
                continue
                
            prod_value = prod_metrics[metric_name]
            cand_value = candidate_metrics[metric_name]
            
            # Calculate improvement
            if direction == "lower":
                # For metrics where lower is better (RMSE, MAE)
                improvement = (prod_value - cand_value) / prod_value
                is_better = cand_value < prod_value
            else:
                # For metrics where higher is better (R2, accuracy)
                improvement = (cand_value - prod_value) / abs(prod_value)
                is_better = cand_value > prod_value
            
            comparison_results[metric_name] = {
                "production": prod_value,
                "candidate": cand_value,
                "improvement": improvement * 100,  # As percentage
                "is_better": is_better,
                "direction": direction
            }
            
            # Only count as improvement if it exceeds threshold
            if is_better and abs(improvement) >= improvement_threshold:
                improvements.append(metric_name)
        
        # Decide if we should promote (all key metrics should improve)
        key_metrics = ["test_rmse", "test_mae"]  # Primary metrics
        key_improvements = [m for m in key_metrics if m in improvements]
        
        should_promote = len(key_improvements) == len([m for m in key_metrics if m in comparison_results])
        
        return should_promote, {
            "comparison": comparison_results,
            "improvements": improvements,
            "should_promote": should_promote,
            "summary": f"Improved on {len(improvements)} out of {len(comparison_results)} metrics"
        }
    
    def validate_model_performance(
        self, 
        run_id: str,
        test_data: pd.DataFrame,
        target_col: str = "sales",
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """
        Additional validation on holdout test data
        """
        try:
            # Load the candidate model
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Sample test data if too large
            if len(test_data) > sample_size:
                test_sample = test_data.sample(n=sample_size, random_state=42)
            else:
                test_sample = test_data.copy()
            
            # Make predictions
            X_test = test_sample.drop(columns=[target_col])
            y_test = test_sample[target_col]
            
            predictions = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            return {
                "validation_rmse": rmse,
                "validation_mae": mae,
                "validation_r2": r2,
                "validation_samples": len(test_sample)
            }
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return {}
    
    def promote_model(
        self, 
        model_name: str,
        candidate_run_id: str,
        comparison_results: Dict[str, Any]
    ) -> bool:
        """
        Promote candidate model to production
        """
        try:
            # Register model if not already registered
            try:
                self.client.create_registered_model(model_name)
            except:
                pass  # Model already registered
            
            # Create new model version
            model_uri = f"runs:/{candidate_run_id}/model"
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=candidate_run_id,
                description=f"Promoted based on improved metrics: {comparison_results['summary']}"
            )
            
            # Transition to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True
            )
            
            # Log promotion details
            with mlflow.start_run(run_id=candidate_run_id):
                mlflow.log_dict(comparison_results, "promotion_details.json")
                mlflow.set_tag("promoted_to_production", "true")
                mlflow.set_tag("promotion_date", datetime.now().isoformat())
            
            logger.info(f"Successfully promoted model {model_name} version {model_version.version} to production")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False
    
    def evaluate_and_promote(
        self,
        candidate_run_id: str,
        model_names: list = ["xgboost", "lightgbm", "ensemble"],
        test_data: Optional[pd.DataFrame] = None,
        force_first_deployment: bool = False
    ) -> Dict[str, Any]:
        """
        Main method to evaluate and potentially promote models
        
        Args:
            candidate_run_id: Run ID of candidate models
            model_names: List of model names to evaluate
            test_data: Optional additional test data for validation
            force_first_deployment: If True, deploy even if no production model exists
            
        Returns:
            Dictionary with promotion results for each model
        """
        results = {}
        
        for model_name in model_names:
            logger.info(f"Evaluating {model_name} for promotion...")
            
            # Get production model
            prod_model = self.get_production_model(model_name)
            
            if prod_model is None:
                if force_first_deployment:
                    # No production model, deploy this as first version
                    logger.info(f"No production model found for {model_name}, deploying as first version")
                    promoted = self.promote_model(
                        model_name, 
                        candidate_run_id,
                        {"summary": "First deployment"}
                    )
                    results[model_name] = {
                        "promoted": promoted,
                        "reason": "First deployment"
                    }
                else:
                    results[model_name] = {
                        "promoted": False,
                        "reason": "No production model to compare against"
                    }
                continue
            
            # Get candidate model
            candidate_model = self.get_candidate_model(candidate_run_id, model_name)
            
            # Compare metrics
            should_promote, comparison = self.compare_metrics(
                prod_model["metrics"],
                candidate_model["metrics"]
            )
            
            # Additional validation if test data provided
            if test_data is not None and should_promote:
                validation_metrics = self.validate_model_performance(
                    candidate_run_id, 
                    test_data
                )
                comparison["validation"] = validation_metrics
            
            # Promote if better
            if should_promote:
                promoted = self.promote_model(model_name, candidate_run_id, comparison)
                results[model_name] = {
                    "promoted": promoted,
                    "comparison": comparison,
                    "previous_version": prod_model["version"]
                }
            else:
                results[model_name] = {
                    "promoted": False,
                    "comparison": comparison,
                    "reason": "Performance not improved sufficiently"
                }
                
            logger.info(f"{model_name}: {'Promoted' if results[model_name]['promoted'] else 'Not promoted'}")
        
        return results