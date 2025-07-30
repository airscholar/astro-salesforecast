import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import optuna
import mlflow

from utils.mlflow_utils import MLflowManager
from feature_engineering.feature_pipeline import FeatureEngineer
from data_validation.validators import DataValidator
from ml_models.advanced_ensemble import AdvancedEnsemble
from ml_models.diagnostics import diagnose_model_performance
from ml_models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.data_validator = DataValidator(config_path)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'sales',
                    date_col: str = 'date', group_cols: Optional[List[str]] = None,
                    categorical_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        logger.info("Preparing data for training")
        
        # Skip validation for aggregated data that doesn't include product_id
        # The validation expects product-level data but we're training on store-level aggregates
        # Validate only essential columns
        required_cols = ['date', target_col]
        if group_cols:
            required_cols.extend(group_cols)
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")
        
        # Feature engineering
        df_features = self.feature_engineer.create_all_features(
            df, target_col, date_col, group_cols, categorical_cols
        )
        
        # Skip target encoding to avoid data leakage
        # Target encoding can cause overfitting in time series
        
        # Split data chronologically for time series
        df_sorted = df_features.sort_values(date_col)
        
        # Use more recent data for validation and testing for better performance
        # This ensures the model learns from patterns closer to what it will predict
        train_size = int(len(df_sorted) * (1 - self.training_config['test_size'] - self.training_config['validation_size']))
        val_size = int(len(df_sorted) * self.training_config['validation_size'])
        
        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size:train_size + val_size]
        test_df = df_sorted[train_size + val_size:]
        
        # Remove any rows with NaN in target column
        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])
        
        # Skip feature selection for now - let models handle all features
        # This allows models to learn which features are important
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def preprocess_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, target_col: str,
                           exclude_cols: List[str] = ['date']) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col not in exclude_cols + [target_col]]
        
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        # Encode categorical variables
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X_train.loc[:, col] = self.encoders[col].fit_transform(X_train[col].astype(str))
            else:
                X_train.loc[:, col] = self.encoders[col].transform(X_train[col].astype(str))
            
            X_val.loc[:, col] = self.encoders[col].transform(X_val[col].astype(str))
            X_test.loc[:, col] = self.encoders[col].transform(X_test[col].astype(str))
        
        # Scale numerical features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame to preserve feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        self.scalers['standard'] = scaler
        self.feature_cols = feature_cols
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred)
        }
        return metrics
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     use_optuna: bool = True) -> xgb.XGBRegressor:
        
        logger.info("Training XGBoost model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42
                }
                
                params['early_stopping_rounds'] = 50
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
        else:
            best_params = self.model_config['xgboost']['params']
        
        best_params['early_stopping_rounds'] = 50
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      use_optuna: bool = True) -> lgb.LGBMRegressor:
        
        logger.info("Training LightGBM model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'verbosity': -1,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt'
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['verbosity'] = -1
        else:
            best_params = self.model_config['lightgbm']['params']
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 callbacks=[lgb.early_stopping(50)])
        
        self.models['lightgbm'] = model
        return model
    
    def train_prophet(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                     date_col: str = 'date', target_col: str = 'sales') -> Prophet:
        
        logger.info("Training Prophet model")
        
        # Prepare data for Prophet
        prophet_train = train_df[[date_col, target_col]].rename(
            columns={date_col: 'ds', target_col: 'y'}
        )
        
        # Remove any NaN values
        prophet_train = prophet_train.dropna()
        
        # Ensure dates are sorted
        prophet_train = prophet_train.sort_values('ds')
        
        # Initialize Prophet with simplified parameters to avoid memory issues
        prophet_params = self.model_config['prophet']['params'].copy()
        
        # Override some parameters for stability
        prophet_params.update({
            'stan_backend': 'CMDSTANPY',  # Use cmdstanpy backend
            'mcmc_samples': 0,  # Disable MCMC for speed and stability
            'uncertainty_samples': 100,  # Reduce uncertainty samples
        })
        
        try:
            model = Prophet(**prophet_params)
            
            # Add only essential regressors to reduce complexity
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            regressor_cols = [col for col in numeric_cols if col not in [target_col, 'year', 'month', 'day', 'week', 'quarter']]
            
            # Limit to top 5 most important regressors based on variance
            if len(regressor_cols) > 5:
                variances = {col: train_df[col].var() for col in regressor_cols}
                regressor_cols = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)[:5]
            
            for col in regressor_cols:
                if train_df[col].std() > 0:  # Only add regressors with variance
                    model.add_regressor(col)
                    prophet_train[col] = train_df[col]
            
            # Fit the model with error handling
            model.fit(prophet_train)
            
            self.models['prophet'] = model
            return model
            
        except Exception as e:
            logger.error(f"Prophet training failed with parameters: {e}")
            # Try with minimal configuration
            logger.info("Retrying Prophet with minimal configuration...")
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                uncertainty_samples=50,
                mcmc_samples=0
            )
            
            # Train without any additional regressors
            model.fit(prophet_train[['ds', 'y']])
            
            self.models['prophet'] = model
            return model
    
    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, target_col: str = 'sales',
                        use_optuna: bool = True) -> Dict[str, Dict[str, Any]]:
        
        results = {}
        
        # Start MLflow run
        run_id = self.mlflow_manager.start_run(
            run_name=f"sales_forecast_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={"model_type": "ensemble", "use_optuna": str(use_optuna)}
        )
        
        try:
            # Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_features(
                train_df, val_df, test_df, target_col
            )
            
            # Log data stats
            self.mlflow_manager.log_params({
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "n_features": X_train.shape[1]
            })
            
            # Train XGBoost
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val, use_optuna)
            xgb_pred = xgb_model.predict(X_test)
            xgb_metrics = self.calculate_metrics(y_test, xgb_pred)
            
            self.mlflow_manager.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
            self.mlflow_manager.log_model(xgb_model, "xgboost", 
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top XGBoost features:\n{feature_importance.to_string()}")
            self.mlflow_manager.log_params({f"xgb_top_feature_{i}": f"{row['feature']} ({row['importance']:.4f})" 
                                           for i, (_, row) in enumerate(feature_importance.iterrows())})
            
            results['xgboost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'predictions': xgb_pred
            }
            
            # Train LightGBM
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val, use_optuna)
            lgb_pred = lgb_model.predict(X_test)
            lgb_metrics = self.calculate_metrics(y_test, lgb_pred)
            
            self.mlflow_manager.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
            self.mlflow_manager.log_model(lgb_model, "lightgbm",
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance for LightGBM
            lgb_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top LightGBM features:\n{lgb_importance.to_string()}")
            
            results['lightgbm'] = {
                'model': lgb_model,
                'metrics': lgb_metrics,
                'predictions': lgb_pred
            }
            
            # Train Prophet if enabled
            prophet_enabled = self.model_config.get('prophet', {}).get('enabled', True)
            
            if prophet_enabled:
                try:
                    prophet_model = self.train_prophet(train_df, val_df)
                    
                    # Create future dataframe for Prophet predictions
                    future = test_df[['date']].rename(columns={'date': 'ds'})
                    
                    # Add regressors if they exist
                    if hasattr(prophet_model, 'extra_regressors') and prophet_model.extra_regressors:
                        regressor_cols = [col for col in prophet_model.extra_regressors.keys()]
                        for col in regressor_cols:
                            if col in test_df.columns:
                                future[col] = test_df[col]
                    
                    prophet_pred = prophet_model.predict(future)['yhat'].values
                    prophet_metrics = self.calculate_metrics(y_test, prophet_pred)
                    
                    self.mlflow_manager.log_metrics({f"prophet_{k}": v for k, v in prophet_metrics.items()})
                    
                    results['prophet'] = {
                        'model': prophet_model,
                        'metrics': prophet_metrics,
                        'predictions': prophet_pred
                    }
                    
                    # Ensemble predictions with all three models
                    ensemble_pred = (xgb_pred + lgb_pred + prophet_pred) / 3
                except Exception as e:
                    logger.warning(f"Prophet training failed: {e}. Using weighted ensemble of XGBoost and LightGBM.")
                    prophet_enabled = False
            
            if not prophet_enabled:
                # Weighted ensemble based on individual model performance (using validation R2)
                xgb_val_pred = xgb_model.predict(X_val)
                lgb_val_pred = lgb_model.predict(X_val)
                
                xgb_val_r2 = r2_score(y_val, xgb_val_pred)
                lgb_val_r2 = r2_score(y_val, lgb_val_pred)
                
                # Calculate weights based on R2 scores with minimum weight constraint
                # This prevents a poorly performing model from being completely ignored
                min_weight = 0.2
                xgb_weight = max(min_weight, xgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
                lgb_weight = max(min_weight, lgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
                
                # Normalize weights
                total_weight = xgb_weight + lgb_weight
                xgb_weight = xgb_weight / total_weight
                lgb_weight = lgb_weight / total_weight
                
                logger.info(f"Ensemble weights - XGBoost: {xgb_weight:.3f}, LightGBM: {lgb_weight:.3f}")
                
                # Create ensemble model
                ensemble_weights = {
                    'xgboost': xgb_weight,
                    'lightgbm': lgb_weight
                }
                
                # Use simple weighted ensemble based on validation performance
                ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred
            
            # Create the ensemble model object
            ensemble_models = {
                'xgboost': xgb_model,
                'lightgbm': lgb_model
            }
            
            if 'prophet' in results:
                ensemble_models['prophet'] = results['prophet']['model']
                ensemble_weights = {
                    'xgboost': 1/3,
                    'lightgbm': 1/3,
                    'prophet': 1/3
                }
            
            ensemble_model = EnsembleModel(ensemble_models, ensemble_weights)
            
            # Save ensemble model
            self.models['ensemble'] = ensemble_model
            
            ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)
            
            self.mlflow_manager.log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})
            self.mlflow_manager.log_model(ensemble_model, "ensemble", 
                                         input_example=X_train.iloc[:5])
            
            results['ensemble'] = {
                'model': ensemble_model,
                'metrics': ensemble_metrics,
                'predictions': ensemble_pred
            }
            
            # Run diagnostics
            logger.info("Running model diagnostics...")
            test_predictions = {
                'xgboost': xgb_pred if 'xgboost' in results else None,
                'lightgbm': lgb_pred if 'lightgbm' in results else None,
                'ensemble': ensemble_pred
            }
            
            diagnosis = diagnose_model_performance(
                train_df, val_df, test_df, test_predictions, target_col
            )
            
            logger.info("Diagnostic recommendations:")
            for rec in diagnosis['recommendations']:
                logger.warning(f"- {rec}")
            
            # Generate visualizations
            logger.info("Generating model comparison visualizations...")
            try:
                self._generate_and_log_visualizations(results, test_df, target_col)
            except Exception as viz_error:
                logger.error(f"Visualization generation failed: {viz_error}", exc_info=True)
            
            # Save artifacts
            self.save_artifacts()
            
            # Get current run ID for verification
            current_run_id = mlflow.active_run().info.run_id
            
            self.mlflow_manager.end_run()
            
            # Sync artifacts to S3
            from utils.mlflow_s3_utils import MLflowS3Manager
            
            logger.info("Syncing artifacts to S3...")
            try:
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(current_run_id)
                logger.info("✓ Successfully synced artifacts to S3")
                
                # Verify S3 artifacts after sync
                from utils.s3_verification import verify_s3_artifacts, log_s3_verification_results
                
                logger.info("Verifying S3 artifact storage...")
                verification_results = verify_s3_artifacts(
                    run_id=current_run_id,
                    expected_artifacts=[
                        'models/', 
                        'scalers.pkl', 
                        'encoders.pkl', 
                        'feature_cols.pkl',
                        'visualizations/',
                        'reports/'
                    ]
                )
                log_s3_verification_results(verification_results)
                
                if not verification_results["success"]:
                    logger.warning("S3 artifact verification failed after sync")
            except Exception as e:
                logger.error(f"Failed to sync artifacts to S3: {e}")
            
        except Exception as e:
            self.mlflow_manager.end_run(status="FAILED")
            raise e
        
        return results
    
    def _generate_and_log_visualizations(self, results: Dict[str, Any], 
                                       test_df: pd.DataFrame, 
                                       target_col: str = 'sales') -> None:
        """Generate and log model comparison visualizations to MLflow"""
        try:
            from ml_models.model_visualization import ModelVisualizer
            import tempfile
            import os
            
            logger.info("Starting visualization generation...")
            visualizer = ModelVisualizer()
            
            # Extract metrics
            metrics_dict = {}
            for model_name, model_results in results.items():
                if 'metrics' in model_results:
                    metrics_dict[model_name] = model_results['metrics']
            
            # Prepare predictions data
            predictions_dict = {}
            for model_name, model_results in results.items():
                if 'predictions' in model_results and model_results['predictions'] is not None:
                    pred_df = test_df[['date']].copy()
                    pred_df['prediction'] = model_results['predictions']
                    predictions_dict[model_name] = pred_df
            
            # Extract feature importance if available
            feature_importance_dict = {}
            for model_name, model_results in results.items():
                if model_name in ['xgboost', 'lightgbm'] and 'model' in model_results:
                    model = model_results['model']
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': self.feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        feature_importance_dict[model_name] = importance_df
            
            # Create temporary directory for visualizations
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Creating visualizations in temporary directory: {temp_dir}")
                
                # Generate all visualizations
                saved_files = visualizer.create_comprehensive_report(
                    metrics_dict=metrics_dict,
                    predictions_dict=predictions_dict,
                    actual_data=test_df,
                    feature_importance_dict=feature_importance_dict if feature_importance_dict else None,
                    save_dir=temp_dir
                )
                
                logger.info(f"Generated {len(saved_files)} visualization files: {list(saved_files.keys())}")
                
                # Log each visualization to MLflow
                for viz_name, file_path in saved_files.items():
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, "visualizations")
                        logger.info(f"Logged visualization: {viz_name} from {file_path}")
                    else:
                        logger.warning(f"Visualization file not found: {file_path}")
                
                # Also create a combined HTML report
                self._create_combined_html_report(saved_files, temp_dir)
                
                # Log the combined report
                combined_report = os.path.join(temp_dir, 'model_comparison_report.html')
                if os.path.exists(combined_report):
                    mlflow.log_artifact(combined_report, "reports")
                    logger.info("Logged combined HTML report")
                    
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            # Don't fail the entire training if visualization fails
    
    def _create_combined_html_report(self, saved_files: Dict[str, str], save_dir: str) -> None:
        """Create a combined HTML report with all visualizations"""
        import os
        from datetime import datetime
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1, h2 {
                    color: #333;
                }
                .section {
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .timestamp {
                    color: #666;
                    font-size: 14px;
                }
                iframe {
                    width: 100%;
                    height: 800px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin-top: 10px;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Sales Forecast Model Comparison Report</h1>
            <p class="timestamp">Generated on: {timestamp}</p>
        """
        
        html_content = html_content.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add each visualization section
        sections = [
            ('metrics_comparison', 'Model Performance Metrics'),
            ('predictions_comparison', 'Predictions Comparison'),
            ('residuals_analysis', 'Residuals Analysis'),
            ('error_distribution', 'Error Distribution'),
            ('feature_importance', 'Feature Importance'),
            ('summary', 'Summary Statistics')
        ]
        
        for key, title in sections:
            if key in saved_files:
                html_content += f'<div class="section"><h2>{title}</h2>'
                
                # All files are now PNG - base64 encode them
                import base64
                with open(saved_files[key], 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                html_content += f'<img src="data:image/png;base64,{img_data}" alt="{title}">'
                
                html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save the combined report
        with open(os.path.join(save_dir, 'model_comparison_report.html'), 'w') as f:
            f.write(html_content)
    
    def save_artifacts(self):
        # Save scalers and encoders
        joblib.dump(self.scalers, '/tmp/scalers.pkl')
        joblib.dump(self.encoders, '/tmp/encoders.pkl')
        joblib.dump(self.feature_cols, '/tmp/feature_cols.pkl')
        
        # Save individual models in the expected format
        import os
        os.makedirs('/tmp/models/xgboost', exist_ok=True)
        os.makedirs('/tmp/models/lightgbm', exist_ok=True)
        os.makedirs('/tmp/models/ensemble', exist_ok=True)
        
        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost'], '/tmp/models/xgboost/xgboost_model.pkl')
        
        if 'lightgbm' in self.models:
            joblib.dump(self.models['lightgbm'], '/tmp/models/lightgbm/lightgbm_model.pkl')
            
        if 'ensemble' in self.models:
            joblib.dump(self.models['ensemble'], '/tmp/models/ensemble/ensemble_model.pkl')
        
        self.mlflow_manager.log_artifacts('/tmp/')
        
        logger.info("Artifacts saved successfully")