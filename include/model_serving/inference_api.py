from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
import mlflow
import yaml
from datetime import datetime, timedelta
import logging
import asyncio
from contextlib import asynccontextmanager

from utils.mlflow_utils import MLflowManager
from feature_engineering.feature_pipeline import FeatureEngineer
from data_validation.validators import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    store_id: str
    product_id: str
    date: str
    additional_features: Optional[Dict[str, Any]] = {}
    
    @validator('date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]


class PredictionResponse(BaseModel):
    store_id: str
    product_id: str
    date: str
    prediction: float
    confidence_interval_80: List[float]
    confidence_interval_95: List[float]
    model_version: str
    prediction_timestamp: str


class ModelInferenceService:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.data_validator = DataValidator(config_path)
        
        self.models = {}
        self.scalers = None
        self.encoders = None
        self.feature_cols = None
        self.model_version = None
        
    def load_models(self, model_stage: str = "Production"):
        logger.info(f"Loading models from stage: {model_stage}")
        
        try:
            # Load XGBoost model
            xgb_version = self.mlflow_manager.get_latest_model_version("xgboost", stage=model_stage)
            xgb_uri = f"models:/{self.mlflow_manager.registry_name}_xgboost/{xgb_version['version']}"
            self.models['xgboost'] = mlflow.xgboost.load_model(xgb_uri)
            
            # Load LightGBM model
            lgb_version = self.mlflow_manager.get_latest_model_version("lightgbm", stage=model_stage)
            lgb_uri = f"models:/{self.mlflow_manager.registry_name}_lightgbm/{lgb_version['version']}"
            self.models['lightgbm'] = mlflow.lightgbm.load_model(lgb_uri)
            
            # Load Prophet model
            prophet_version = self.mlflow_manager.get_latest_model_version("prophet", stage=model_stage)
            prophet_uri = f"models:/{self.mlflow_manager.registry_name}_prophet/{prophet_version['version']}"
            self.models['prophet'] = mlflow.prophet.load_model(prophet_uri)
            
            # Load preprocessing artifacts
            run_id = xgb_version['run_id']  # Using XGBoost run as reference
            artifact_uri = f"runs:/{run_id}/artifacts"
            
            # Download artifacts
            mlflow.artifacts.download_artifacts(f"{artifact_uri}/scalers.pkl", dst_path="/tmp/")
            mlflow.artifacts.download_artifacts(f"{artifact_uri}/encoders.pkl", dst_path="/tmp/")
            mlflow.artifacts.download_artifacts(f"{artifact_uri}/feature_cols.pkl", dst_path="/tmp/")
            
            self.scalers = joblib.load("/tmp/scalers.pkl")
            self.encoders = joblib.load("/tmp/encoders.pkl")
            self.feature_cols = joblib.load("/tmp/feature_cols.pkl")
            
            self.model_version = xgb_version['version']
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        # Create features
        df_features = self.feature_engineer.create_all_features(
            df, 
            target_col='sales',  # Dummy column for feature creation
            date_col='date',
            group_cols=['store_id', 'product_id']
        )
        
        # Select only the features used during training
        X = df_features[self.feature_cols]
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            if col in self.encoders:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Scale features
        X_scaled = self.scalers['standard'].transform(X)
        
        return X_scaled
    
    def generate_confidence_intervals(self, predictions: np.ndarray, 
                                    confidence_levels: List[float] = [0.8, 0.95]) -> Dict[str, List[float]]:
        # Simple confidence intervals based on prediction variance
        # In production, use proper prediction intervals from the models
        std_dev = np.std(predictions) * 0.1  # Simplified approach
        
        intervals = {}
        for level in confidence_levels:
            z_score = 1.96 if level == 0.95 else 1.28  # Approximate z-scores
            margin = z_score * std_dev
            
            intervals[f"ci_{int(level*100)}"] = [
                float(pred - margin) for pred in predictions
            ], [
                float(pred + margin) for pred in predictions
            ]
        
        return intervals
    
    def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        # Create dataframe from request
        data = {
            'store_id': [request.store_id],
            'product_id': [request.product_id],
            'date': [pd.to_datetime(request.date)],
            'sales': [0]  # Dummy value for feature engineering
        }
        
        # Add additional features if provided
        for key, value in request.additional_features.items():
            data[key] = [value]
        
        df = pd.DataFrame(data)
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Get predictions from each model
        xgb_pred = self.models['xgboost'].predict(X)
        lgb_pred = self.models['lightgbm'].predict(X)
        
        # Prophet requires different input format
        prophet_df = df[['date']].rename(columns={'date': 'ds'})
        for col in self.models['prophet'].extra_regressors.keys():
            if col in df.columns:
                prophet_df[col] = df[col]
        
        prophet_pred = self.models['prophet'].predict(prophet_df)['yhat'].values
        
        # Ensemble prediction
        ensemble_pred = (xgb_pred + lgb_pred + prophet_pred) / 3
        
        # Generate confidence intervals
        all_preds = np.array([xgb_pred[0], lgb_pred[0], prophet_pred[0]])
        intervals = self.generate_confidence_intervals(all_preds)
        
        return PredictionResponse(
            store_id=request.store_id,
            product_id=request.product_id,
            date=request.date,
            prediction=float(ensemble_pred[0]),
            confidence_interval_80=[
                float(ensemble_pred[0] - intervals['ci_80'][1][0] + intervals['ci_80'][0][0]),
                float(intervals['ci_80'][1][0])
            ],
            confidence_interval_95=[
                float(ensemble_pred[0] - intervals['ci_95'][1][0] + intervals['ci_95'][0][0]),
                float(intervals['ci_95'][1][0])
            ],
            model_version=self.model_version,
            prediction_timestamp=datetime.now().isoformat()
        )
    
    async def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        # Process predictions concurrently
        tasks = [self.predict_single_async(req) for req in requests]
        results = await asyncio.gather(*tasks)
        return results
    
    async def predict_single_async(self, request: PredictionRequest) -> PredictionResponse:
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict_single, request)


# Global instance
inference_service = ModelInferenceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    inference_service.load_models()
    yield
    # Shutdown
    logger.info("Shutting down inference service")


# Create FastAPI app
app = FastAPI(
    title="Sales Forecast Inference API",
    description="API for sales forecasting model inference",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_version": inference_service.model_version,
        "models_loaded": list(inference_service.models.keys())
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        return inference_service.predict_single(request)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    try:
        return await inference_service.predict_batch(request.predictions)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-models")
async def reload_models(background_tasks: BackgroundTasks, stage: str = "Production"):
    background_tasks.add_task(inference_service.load_models, stage)
    return {"message": "Model reload initiated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)