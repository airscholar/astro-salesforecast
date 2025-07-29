"""
Ensemble Model for combining predictions from multiple models
"""
import numpy as np
from typing import Dict, Any, List
import joblib


class EnsembleModel:
    """Ensemble model that combines predictions from multiple models"""
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        """
        Initialize ensemble model
        
        Args:
            models: Dictionary of model_name -> model object
            weights: Dictionary of model_name -> weight (if None, uses equal weights)
        """
        self.models = models
        
        if weights is None:
            # Equal weights for all models
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models.keys()}
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            self.weights = {name: w / total_weight for name, w in weights.items()}
            
        # Ensure we have weights for all models
        for model_name in self.models.keys():
            if model_name not in self.weights:
                self.weights[model_name] = 0.0
                
    def predict(self, X):
        """Make ensemble predictions"""
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            if self.weights[model_name] > 0:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.weights[model_name])
        
        if not predictions:
            raise ValueError("No models available for prediction")
            
        # Weighted average of predictions
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        # Calculate weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {
            'models': self.models,
            'weights': self.weights
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def save(self, filepath: str):
        """Save ensemble model to file"""
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        """Load ensemble model from file"""
        return joblib.load(filepath)
    
    def __repr__(self):
        model_info = []
        for name, weight in self.weights.items():
            model_info.append(f"{name}: {weight:.3f}")
        return f"EnsembleModel({', '.join(model_info)})"