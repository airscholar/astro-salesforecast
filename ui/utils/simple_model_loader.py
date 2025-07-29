"""
Simplified model loader for Streamlit UI
"""

import os
import pickle
import joblib
import mlflow
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SimpleModelLoader:
    """Simplified model loader that works with pickled models"""
    
    def __init__(self):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        self.models = {}
        self.scalers = None
        self.encoders = None
        self.feature_cols = None
        self.loaded = False
        
    def load_models_from_run(self, run_id: str) -> bool:
        """Load models from a specific MLflow run"""
        try:
            logger.info(f"Loading models from run: {run_id}")
            
            # Download artifacts
            client = mlflow.tracking.MlflowClient()
            local_dir = f"/tmp/mlflow_models/{run_id}"
            os.makedirs(local_dir, exist_ok=True)
            
            # Download all artifacts
            artifacts_path = client.download_artifacts(run_id, "", dst_path=local_dir)
            logger.info(f"Downloaded artifacts to: {artifacts_path}")
            
            # Load scalers
            scalers_path = os.path.join(artifacts_path, "scalers.pkl")
            if os.path.exists(scalers_path):
                try:
                    self.scalers = joblib.load(scalers_path)
                    logger.info("Loaded scalers")
                except Exception as e:
                    logger.warning(f"Could not load scalers with joblib, trying pickle: {e}")
                    with open(scalers_path, 'rb') as f:
                        self.scalers = pickle.load(f)
                    logger.info("Loaded scalers with pickle")
            
            # Load encoders
            encoders_path = os.path.join(artifacts_path, "encoders.pkl")
            if os.path.exists(encoders_path):
                try:
                    self.encoders = joblib.load(encoders_path)
                    logger.info("Loaded encoders")
                except Exception as e:
                    logger.warning(f"Could not load encoders with joblib, trying pickle: {e}")
                    with open(encoders_path, 'rb') as f:
                        self.encoders = pickle.load(f)
                    logger.info("Loaded encoders with pickle")
            
            # Load feature columns
            feature_cols_path = os.path.join(artifacts_path, "feature_cols.pkl")
            if os.path.exists(feature_cols_path):
                try:
                    self.feature_cols = joblib.load(feature_cols_path)
                    logger.info(f"Loaded {len(self.feature_cols)} feature columns")
                except Exception as e:
                    logger.warning(f"Could not load feature_cols with joblib, trying pickle: {e}")
                    with open(feature_cols_path, 'rb') as f:
                        self.feature_cols = pickle.load(f)
                    logger.info(f"Loaded {len(self.feature_cols)} feature columns with pickle")
            
            # Load models
            models_dir = os.path.join(artifacts_path, "models")
            if os.path.exists(models_dir):
                # Load XGBoost
                xgb_path = os.path.join(models_dir, "xgboost", "xgboost_model.pkl")
                if os.path.exists(xgb_path):
                    try:
                        self.models['xgboost'] = joblib.load(xgb_path)
                        logger.info("Loaded XGBoost model")
                    except Exception as e:
                        logger.warning(f"Could not load XGBoost with joblib, trying pickle: {e}")
                        with open(xgb_path, 'rb') as f:
                            self.models['xgboost'] = pickle.load(f)
                        logger.info("Loaded XGBoost model with pickle")
                
                # Load LightGBM
                lgb_path = os.path.join(models_dir, "lightgbm", "lightgbm_model.pkl")
                if os.path.exists(lgb_path):
                    try:
                        self.models['lightgbm'] = joblib.load(lgb_path)
                        logger.info("Loaded LightGBM model")
                    except Exception as e:
                        logger.warning(f"Could not load LightGBM with joblib, trying pickle: {e}")
                        with open(lgb_path, 'rb') as f:
                            self.models['lightgbm'] = pickle.load(f)
                        logger.info("Loaded LightGBM model with pickle")
                
                # Load Ensemble
                ensemble_path = os.path.join(models_dir, "ensemble", "ensemble_model.pkl")
                if os.path.exists(ensemble_path):
                    try:
                        # First try regular loading
                        self.models['ensemble'] = joblib.load(ensemble_path)
                        logger.info("Loaded Ensemble model")
                    except Exception as e:
                        logger.warning(f"Could not load Ensemble model with joblib: {e}")
                        # Try to recreate ensemble from loaded models
                        if 'xgboost' in self.models and 'lightgbm' in self.models:
                            from .ensemble_model_standalone import EnsembleModel
                            ensemble_models = {
                                'xgboost': self.models['xgboost'],
                                'lightgbm': self.models['lightgbm']
                            }
                            ensemble_weights = {'xgboost': 0.5, 'lightgbm': 0.5}
                            self.models['ensemble'] = EnsembleModel(ensemble_models, ensemble_weights)
                            logger.info("Created Ensemble model from loaded models")
                        else:
                            logger.warning("Not enough models to create ensemble")
            
            self.loaded = len(self.models) > 0
            return self.loaded
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_latest_run(self) -> Optional[str]:
        """Get the latest successful run ID"""
        try:
            exp = mlflow.get_experiment_by_name("sales_forecasting")
            if not exp:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                return runs.iloc[0]['run_id']
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest run: {e}")
            return None
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions by averaging available models"""
        predictions = []
        
        if 'xgboost' in self.models:
            predictions.append(self.models['xgboost'].predict(X))
            
        if 'lightgbm' in self.models:
            predictions.append(self.models['lightgbm'].predict(X))
        
        if predictions:
            return np.mean(predictions, axis=0)
        else:
            raise ValueError("No models available for prediction")
    
    def predict(self, X: np.ndarray, model_type: str = 'ensemble') -> np.ndarray:
        """Make predictions with specified model"""
        if model_type == 'ensemble':
            # Check if we have a saved ensemble model
            if 'ensemble' in self.models:
                return self.models['ensemble'].predict(X)
            else:
                # Fall back to creating ensemble on the fly
                return self.predict_ensemble(X)
        elif model_type in self.models:
            return self.models[model_type].predict(X)
        else:
            raise ValueError(f"Model type '{model_type}' not available")