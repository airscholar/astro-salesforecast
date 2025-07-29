import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def diagnose_model_performance(train_df: pd.DataFrame, 
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame,
                              predictions: Dict[str, np.ndarray],
                              target_col: str = 'sales') -> Dict[str, Any]:
    """Diagnose why models are underperforming"""
    
    diagnosis = {
        'data_quality': {},
        'distribution_shift': {},
        'prediction_analysis': {},
        'recommendations': []
    }
    
    # 1. Check data quality
    logger.info("Checking data quality...")
    
    # Check for outliers in target
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]
    
    train_outliers = detect_outliers(y_train)
    val_outliers = detect_outliers(y_val)
    test_outliers = detect_outliers(y_test)
    
    diagnosis['data_quality']['train_outliers'] = train_outliers
    diagnosis['data_quality']['val_outliers'] = val_outliers
    diagnosis['data_quality']['test_outliers'] = test_outliers
    
    # 2. Check for distribution shift
    logger.info("Checking for distribution shift...")
    
    train_mean, train_std = y_train.mean(), y_train.std()
    val_mean, val_std = y_val.mean(), y_val.std()
    test_mean, test_std = y_test.mean(), y_test.std()
    
    diagnosis['distribution_shift']['train_stats'] = {'mean': train_mean, 'std': train_std}
    diagnosis['distribution_shift']['val_stats'] = {'mean': val_mean, 'std': val_std}
    diagnosis['distribution_shift']['test_stats'] = {'mean': test_mean, 'std': test_std}
    
    # Check if there's significant shift
    mean_shift_val = abs(val_mean - train_mean) / train_mean
    mean_shift_test = abs(test_mean - train_mean) / train_mean
    
    if mean_shift_val > 0.2:
        diagnosis['recommendations'].append(
            f"Significant distribution shift in validation set (mean shift: {mean_shift_val:.1%})"
        )
    if mean_shift_test > 0.2:
        diagnosis['recommendations'].append(
            f"Significant distribution shift in test set (mean shift: {mean_shift_test:.1%})"
        )
    
    # 3. Analyze predictions
    logger.info("Analyzing predictions...")
    
    for model_name, pred in predictions.items():
        if pred is not None:
            # Check prediction distribution
            pred_mean = pred.mean()
            pred_std = pred.std()
            
            # Check for extreme predictions
            extreme_low = (pred < y_test.min() * 0.5).sum()
            extreme_high = (pred > y_test.max() * 1.5).sum()
            
            # Calculate residuals
            residuals = y_test - pred
            
            diagnosis['prediction_analysis'][model_name] = {
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'extreme_low_count': extreme_low,
                'extreme_high_count': extreme_high,
                'residual_mean': residuals.mean(),
                'residual_std': residuals.std(),
                'mape': np.mean(np.abs(residuals / y_test)) * 100
            }
    
    # 4. Feature importance check
    feature_cols = [col for col in train_df.columns if col != target_col and col != 'date']
    diagnosis['data_quality']['n_features'] = len(feature_cols)
    
    if len(feature_cols) > 50:
        diagnosis['recommendations'].append(
            f"High number of features ({len(feature_cols)}). Consider more aggressive feature selection."
        )
    
    # 5. Check for data leakage
    # Look for perfect correlations
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) > 0:
        correlations = train_df[numeric_cols].corrwith(train_df[target_col])
        high_corr = correlations[abs(correlations) > 0.95]
        
        if len(high_corr) > 0:
            diagnosis['recommendations'].append(
                f"Potential data leakage: {len(high_corr)} features have >95% correlation with target"
            )
            diagnosis['data_quality']['high_correlation_features'] = high_corr.to_dict()
    
    # 6. Sample size check
    if len(train_df) < 1000:
        diagnosis['recommendations'].append(
            f"Small training set ({len(train_df)} samples). Consider generating more data."
        )
    
    # 7. Target variable analysis
    target_zeros = (y_train == 0).sum()
    if target_zeros > len(y_train) * 0.1:
        diagnosis['recommendations'].append(
            f"Many zero sales ({target_zeros} in training). Consider log transformation or zero-inflated models."
        )
    
    return diagnosis


def detect_outliers(data: pd.Series, method: str = 'iqr') -> Dict[str, Any]:
    """Detect outliers in data"""
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': len(outliers) / len(data) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min_outlier': outliers.min() if len(outliers) > 0 else None,
            'max_outlier': outliers.max() if len(outliers) > 0 else None
        }
    
    return {}


def plot_diagnostic_charts(train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          predictions: Dict[str, np.ndarray],
                          target_col: str = 'sales',
                          save_path: str = None):
    """Create diagnostic visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Target distribution across splits
    ax = axes[0, 0]
    ax.hist(train_df[target_col], bins=50, alpha=0.5, label='Train', density=True)
    ax.hist(val_df[target_col], bins=50, alpha=0.5, label='Val', density=True)
    ax.hist(test_df[target_col], bins=50, alpha=0.5, label='Test', density=True)
    ax.set_title('Target Distribution Across Splits')
    ax.set_xlabel(target_col)
    ax.legend()
    
    # 2. Predictions vs Actual
    ax = axes[0, 1]
    y_test = test_df[target_col]
    for model_name, pred in predictions.items():
        if pred is not None:
            ax.scatter(y_test, pred, alpha=0.5, label=model_name)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect')
    ax.set_title('Predictions vs Actual')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend()
    
    # 3. Residual distribution
    ax = axes[1, 0]
    for model_name, pred in predictions.items():
        if pred is not None:
            residuals = y_test - pred
            ax.hist(residuals, bins=50, alpha=0.5, label=model_name, density=True)
    ax.set_title('Residual Distribution')
    ax.set_xlabel('Residuals')
    ax.legend()
    
    # 4. Time series of actual vs predicted
    ax = axes[1, 1]
    if 'date' in test_df.columns:
        dates = test_df['date']
        ax.plot(dates, y_test, 'k-', label='Actual', linewidth=2)
        for model_name, pred in predictions.items():
            if pred is not None:
                ax.plot(dates, pred, '--', label=model_name, alpha=0.7)
        ax.set_title('Time Series: Actual vs Predicted')
        ax.set_xlabel('Date')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig