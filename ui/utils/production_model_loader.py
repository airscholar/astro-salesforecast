"""
Production Model Loader - Loads models from MLflow Model Registry Production stage
"""

import os
import mlflow
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class ProductionModelLoader:
    """Load production models from MLflow Model Registry"""
    
    def __init__(self):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()
        
        self.models = {}
        self.model_versions = {}
        self.loaded = False
        
    def load_production_models(self) -> bool:
        """Load all production models from registry"""
        try:
            logger.info("Loading production models from MLflow Model Registry...")
            
            model_names = ["xgboost", "lightgbm", "ensemble"]
            
            for model_name in model_names:
                try:
                    # Get latest production version
                    versions = self.client.search_model_versions(
                        filter_string=f"name='{model_name}'",
                        order_by=["version_number DESC"]
                    )
                    
                    production_version = None
                    for version in versions:
                        if version.current_stage == "Production":
                            production_version = version
                            break
                    
                    if production_version:
                        # Load model
                        model_uri = f"models:/{model_name}/Production"
                        model = mlflow.pyfunc.load_model(model_uri)
                        
                        self.models[model_name] = model
                        self.model_versions[model_name] = {
                            "version": production_version.version,
                            "run_id": production_version.run_id,
                            "stage": production_version.current_stage,
                            "description": production_version.description
                        }
                        
                        logger.info(f"Loaded {model_name} v{production_version.version} from production")
                    else:
                        logger.warning(f"No production version found for {model_name}")
                        
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
            
            self.loaded = len(self.models) > 0
            
            if self.loaded:
                logger.info(f"Successfully loaded {len(self.models)} production models")
            else:
                logger.warning("No production models found. Using fallback to latest run.")
                # Fallback to latest run
                return self._load_from_latest_run()
                
            return self.loaded
            
        except Exception as e:
            logger.error(f"Error loading production models: {e}")
            return False
    
    def _load_from_latest_run(self) -> bool:
        """Fallback: Load from latest successful run if no production models"""
        try:
            from .simple_model_loader import SimpleModelLoader
            
            logger.info("Falling back to latest run models...")
            
            simple_loader = SimpleModelLoader()
            run_id = simple_loader.get_latest_run()
            
            if run_id and simple_loader.load_models_from_run(run_id):
                self.models = simple_loader.models
                self.loaded = True
                
                # Set version info
                for model_name in self.models.keys():
                    self.model_versions[model_name] = {
                        "version": "latest",
                        "run_id": run_id,
                        "stage": "None",
                        "description": "Loaded from latest run (no production version)"
                    }
                
                return True
                
        except Exception as e:
            logger.error(f"Fallback loading failed: {e}")
            
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "loaded": self.loaded,
            "model_count": len(self.models),
            "models": {}
        }
        
        for model_name, version_info in self.model_versions.items():
            info["models"][model_name] = {
                "version": version_info["version"],
                "stage": version_info["stage"],
                "run_id": version_info["run_id"][:8] if version_info["run_id"] else "N/A"
            }
        
        return info
    
    def predict(self, X: pd.DataFrame, model_type: str = 'ensemble') -> Any:
        """Make predictions with specified model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not loaded. Available: {list(self.models.keys())}")
            
        return self.models[model_type].predict(X)