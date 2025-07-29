import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class AdvancedEnsemble:
    """Advanced ensemble techniques for better model performance"""
    
    def __init__(self):
        self.meta_model = None
        self.base_models = {}
        self.model_weights = {}
        
    def create_stacking_ensemble(self, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                base_predictions: Dict[str, np.ndarray],
                                meta_model_type: str = 'ridge') -> np.ndarray:
        """
        Create a stacking ensemble using base model predictions
        """
        # Create training data for meta-model
        train_meta_features = []
        val_meta_features = []
        
        # Get base model predictions on training data using cross-validation
        # This prevents overfitting of the meta-model
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name in base_predictions.keys():
            # Store out-of-fold predictions
            oof_predictions = np.zeros(len(X_train))
            
            # We'll need to retrain base models on folds
            # For now, use validation predictions as approximation
            train_meta_features.append(base_predictions[model_name]['train'])
            val_meta_features.append(base_predictions[model_name]['val'])
        
        # Stack predictions
        X_meta_train = np.column_stack(train_meta_features)
        X_meta_val = np.column_stack(val_meta_features)
        
        # Add diversity features (differences between predictions)
        if len(base_predictions) > 1:
            model_names = list(base_predictions.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    diff_train = train_meta_features[i] - train_meta_features[j]
                    diff_val = val_meta_features[i] - val_meta_features[j]
                    X_meta_train = np.column_stack([X_meta_train, diff_train])
                    X_meta_val = np.column_stack([X_meta_val, diff_val])
        
        # Train meta-model
        if meta_model_type == 'ridge':
            self.meta_model = Ridge(alpha=1.0)
        elif meta_model_type == 'lasso':
            self.meta_model = Lasso(alpha=0.01)
        elif meta_model_type == 'elastic':
            self.meta_model = ElasticNet(alpha=0.01, l1_ratio=0.5)
        elif meta_model_type == 'rf':
            self.meta_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        else:
            self.meta_model = Ridge(alpha=1.0)
        
        self.meta_model.fit(X_meta_train, y_train)
        
        # Get meta-model predictions
        meta_predictions = self.meta_model.predict(X_meta_val)
        
        logger.info(f"Stacking ensemble created with {meta_model_type} meta-model")
        
        return meta_predictions
    
    def create_blended_ensemble(self,
                               predictions: Dict[str, np.ndarray],
                               y_true: np.ndarray,
                               optimization_metric: str = 'rmse') -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Create optimally weighted blend of predictions
        """
        from scipy.optimize import minimize
        
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            
            # Calculate weighted prediction
            blended = np.zeros_like(y_true)
            for i, (model_name, pred) in enumerate(predictions.items()):
                blended += weights[i] * pred
            
            # Calculate metric
            if optimization_metric == 'rmse':
                return np.sqrt(np.mean((blended - y_true) ** 2))
            elif optimization_metric == 'mae':
                return np.mean(np.abs(blended - y_true))
            else:
                return -1 * self._r2_score(y_true, blended)
        
        # Initialize equal weights
        n_models = len(predictions)
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize weights
        bounds = [(0, 1) for _ in range(n_models)]
        result = optimize.minimize(objective, initial_weights, bounds=bounds)
        
        # Normalize weights
        optimal_weights = result.x / result.x.sum()
        
        # Create final blend
        blended_pred = np.zeros_like(y_true)
        weight_dict = {}
        for i, (model_name, pred) in enumerate(predictions.items()):
            blended_pred += optimal_weights[i] * pred
            weight_dict[model_name] = optimal_weights[i]
        
        logger.info(f"Optimal blend weights: {weight_dict}")
        
        return blended_pred, weight_dict
    
    def _r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def create_dynamic_ensemble(self,
                               base_predictions: Dict[str, Dict[str, np.ndarray]],
                               y_val: np.ndarray,
                               window_size: int = 30) -> np.ndarray:
        """
        Create ensemble with dynamic weights that change over time
        """
        val_preds = {name: preds['val'] for name, preds in base_predictions.items()}
        n_samples = len(y_val)
        dynamic_predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Use a sliding window to calculate local performance
            start_idx = max(0, i - window_size)
            end_idx = min(n_samples, i + window_size)
            
            if end_idx - start_idx < 10:  # Not enough data
                # Use equal weights
                for pred in val_preds.values():
                    dynamic_predictions[i] += pred[i] / len(val_preds)
            else:
                # Calculate local weights based on recent performance
                local_weights = {}
                total_weight = 0
                
                for name, pred in val_preds.items():
                    local_error = np.mean(np.abs(pred[start_idx:end_idx] - y_val[start_idx:end_idx]))
                    # Convert error to weight (lower error = higher weight)
                    weight = 1.0 / (local_error + 1e-6)
                    local_weights[name] = weight
                    total_weight += weight
                
                # Normalize and apply weights
                for name, pred in val_preds.items():
                    dynamic_predictions[i] += (local_weights[name] / total_weight) * pred[i]
        
        return dynamic_predictions