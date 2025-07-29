"""
Simple predictor for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SimplePredictor:
    """Simple predictor that works with SimpleModelLoader"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        # Ensure date column is datetime
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract time features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['weekofyear'] = df['date'].dt.isocalendar().week
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            
            # Add cyclical features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            
        # Add lag features if we have sales data
        if 'sales' in df.columns:
            # Multiple lag features
            for lag in [1, 2, 3, 7, 14, 21, 30]:
                df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
            
            # Rolling statistics for different windows
            for window in [3, 7, 14, 21, 30]:
                df[f'sales_rolling_{window}_mean'] = df['sales'].rolling(window).mean()
                df[f'sales_rolling_{window}_std'] = df['sales'].rolling(window).std()
                df[f'sales_rolling_{window}_min'] = df['sales'].rolling(window).min()
                df[f'sales_rolling_{window}_max'] = df['sales'].rolling(window).max()
                df[f'sales_rolling_{window}_median'] = df['sales'].rolling(window).median()
            
            # Fill NaN values with appropriate defaults
            sales_mean = df['sales'].mean()
            for col in df.columns:
                if 'sales_lag' in col or 'sales_rolling' in col:
                    if 'std' in col:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(sales_mean)
        
        # Add default values for features that might be missing
        if 'quantity_sold' not in df.columns:
            df['quantity_sold'] = 100  # Default quantity
        if 'profit' not in df.columns:
            df['profit'] = 1000  # Default profit
        if 'has_promotion' not in df.columns:
            df['has_promotion'] = 0  # No promotion by default
        if 'customer_traffic' not in df.columns:
            df['customer_traffic'] = 500  # Default traffic
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = 0  # Not holiday by default
            
        return df
    
    def predict(self, input_data: pd.DataFrame, model_type: str = 'ensemble', 
                forecast_days: int = 30) -> Dict[str, Any]:
        """Make predictions"""
        try:
            if not self.model_loader.loaded:
                return {
                    'success': False,
                    'error': 'Models not loaded'
                }
            
            # Prepare historical data
            historical_df = self.prepare_features(input_data)
            
            # Create future dates
            last_date = pd.to_datetime(input_data['date']).max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Create future dataframe
            future_df = pd.DataFrame({
                'date': future_dates,
                'store_id': input_data['store_id'].iloc[-1] if 'store_id' in input_data.columns else 'store_001'
            })
            
            # Prepare features for future dates
            future_df = self.prepare_features(future_df)
            
            # Use last known values for lag features
            if len(historical_df) > 0 and 'sales' in historical_df.columns:
                # Get recent sales values for lag features
                recent_sales = historical_df['sales'].tail(30).values
                sales_mean = historical_df['sales'].mean()
                
                # Set lag features based on historical data
                for lag in [1, 2, 3, 7, 14, 21, 30]:
                    if len(recent_sales) >= lag:
                        future_df[f'sales_lag_{lag}'] = recent_sales[-lag]
                    else:
                        future_df[f'sales_lag_{lag}'] = sales_mean
                
                # Set rolling statistics based on historical data
                for window in [3, 7, 14, 21, 30]:
                    if len(recent_sales) >= window:
                        window_data = recent_sales[-window:]
                        future_df[f'sales_rolling_{window}_mean'] = np.mean(window_data)
                        future_df[f'sales_rolling_{window}_std'] = np.std(window_data)
                        future_df[f'sales_rolling_{window}_min'] = np.min(window_data)
                        future_df[f'sales_rolling_{window}_max'] = np.max(window_data)
                        future_df[f'sales_rolling_{window}_median'] = np.median(window_data)
                    else:
                        future_df[f'sales_rolling_{window}_mean'] = sales_mean
                        future_df[f'sales_rolling_{window}_std'] = 0
                        future_df[f'sales_rolling_{window}_min'] = sales_mean
                        future_df[f'sales_rolling_{window}_max'] = sales_mean
                        future_df[f'sales_rolling_{window}_median'] = sales_mean
            
            # Handle categorical features (store_id)
            if 'store_id' in future_df.columns and future_df['store_id'].dtype == 'object':
                # If we have encoders, use them
                if self.model_loader.encoders and 'store_id' in self.model_loader.encoders:
                    try:
                        # Transform store_id
                        encoder = self.model_loader.encoders['store_id']
                        # Handle unknown categories
                        known_stores = list(encoder.classes_)
                        future_df['store_id'] = future_df['store_id'].apply(
                            lambda x: x if x in known_stores else known_stores[0]
                        )
                        encoded_stores = encoder.transform(future_df['store_id'])
                        future_df['store_id'] = encoded_stores
                    except Exception as e:
                        logger.warning(f"Error encoding store_id: {e}")
                        # Default to numeric encoding
                        future_df['store_id'] = 1
                else:
                    # No encoder, convert to numeric
                    # Extract numeric part if format is "store_XXX"
                    if future_df['store_id'].str.contains('store_').any():
                        future_df['store_id'] = future_df['store_id'].str.extract('(\d+)').astype(int)
                    else:
                        future_df['store_id'] = 1
            
            # Select features based on what the model expects
            if self.model_loader.feature_cols:
                # Use only features that exist in both the data and expected features
                available_features = [col for col in self.model_loader.feature_cols 
                                    if col in future_df.columns]
                if len(available_features) < len(self.model_loader.feature_cols):
                    # Add missing features with default values
                    for col in self.model_loader.feature_cols:
                        if col not in future_df.columns:
                            # Special handling for categorical encoded features
                            if col.startswith('store_'):
                                future_df[col] = 0
                            else:
                                future_df[col] = 0
                X = future_df[self.model_loader.feature_cols].values
            else:
                # Fallback to basic features (exclude string columns)
                feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter', 
                               'is_weekend', 'sales_lag_1', 'sales_lag_7',
                               'sales_rolling_mean_7', 'sales_rolling_std_7']
                # Add any store_id encoded columns
                store_cols = [col for col in future_df.columns if col.startswith('store_id_') and col != 'store_id']
                feature_cols.extend(store_cols)
                available_features = [col for col in feature_cols if col in future_df.columns]
                X = future_df[available_features].values
            
            # Scale features if scaler is available
            if self.model_loader.scalers and 'features' in self.model_loader.scalers:
                try:
                    X = self.model_loader.scalers['features'].transform(X)
                except:
                    logger.warning("Could not apply feature scaling")
            
            # Make predictions
            predictions = self.model_loader.predict(X, model_type=model_type)
            
            # Scale predictions back if scaler is available
            if self.model_loader.scalers and 'target' in self.model_loader.scalers:
                try:
                    predictions = self.model_loader.scalers['target'].inverse_transform(
                        predictions.reshape(-1, 1)
                    ).flatten()
                except:
                    logger.warning("Could not inverse transform predictions")
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'date': future_dates,
                'predicted_sales': predictions,
                'lower_bound': predictions * 0.9,  # Simple 10% confidence interval
                'upper_bound': predictions * 1.1
            })
            
            # Calculate summary statistics
            summary = {
                'total_predicted_sales': predictions.sum(),
                'average_daily_sales': predictions.mean(),
                'max_daily_sales': predictions.max(),
                'min_daily_sales': predictions.min()
            }
            
            return {
                'success': True,
                'predictions': results_df,
                'summary': summary,
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }