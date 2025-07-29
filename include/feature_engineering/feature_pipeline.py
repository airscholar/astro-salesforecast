import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import holidays
import yaml
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
        self.validation_config = self.config['validation']
        
    def create_date_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        df = df.copy()
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        date_features = self.feature_config['date_features']
        
        if 'year' in date_features:
            df['year'] = df[date_col].dt.year
        if 'month' in date_features:
            df['month'] = df[date_col].dt.month
        if 'day' in date_features:
            df['day'] = df[date_col].dt.day
        if 'dayofweek' in date_features:
            df['dayofweek'] = df[date_col].dt.dayofweek
        if 'quarter' in date_features:
            df['quarter'] = df[date_col].dt.quarter
        if 'weekofyear' in date_features:
            df['weekofyear'] = df[date_col].dt.isocalendar().week
        if 'is_weekend' in date_features:
            df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        if 'is_holiday' in date_features:
            us_holidays = holidays.US()
            df['is_holiday'] = df[date_col].apply(lambda x: x in us_holidays).astype(int)
        
        logger.info(f"Created {len(date_features)} date features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = df.copy()
        lag_values = self.feature_config['lag_features']
        
        if group_cols:
            for lag in lag_values:
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        else:
            for lag in lag_values:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Created {len(lag_values)} lag features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str,
                               group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = df.copy()
        windows = self.feature_config['rolling_features']['windows']
        functions = self.feature_config['rolling_features']['functions']
        
        if group_cols:
            for window in windows:
                for func in functions:
                    col_name = f'{target_col}_rolling_{window}_{func}'
                    df[col_name] = df.groupby(group_cols)[target_col].transform(
                        lambda x: x.rolling(window, min_periods=1).agg(func)
                    )
        else:
            for window in windows:
                for func in functions:
                    col_name = f'{target_col}_rolling_{window}_{func}'
                    df[col_name] = df[target_col].rolling(window, min_periods=1).agg(func)
        
        logger.info(f"Created {len(windows) * len(functions)} rolling features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                df[f'{col1}_{col2}_interaction'] = df[col1].astype(str) + "_" + df[col2].astype(str)
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        df = df.copy()
        
        df['month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
        
        df['day_sin'] = np.sin(2 * np.pi * df[date_col].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df[date_col].dt.day / 31)
        
        df['dayofweek_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)
        
        logger.info("Created cyclical features")
        return df
    
    def create_all_features(self, df: pd.DataFrame, target_col: str = 'sales',
                           date_col: str = 'date', 
                           group_cols: Optional[List[str]] = None,
                           categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        
        logger.info("Starting feature engineering pipeline")
        
        # Sort by date for proper lag and rolling calculations
        if group_cols:
            df = df.sort_values(group_cols + [date_col])
        else:
            df = df.sort_values(date_col)
        
        # Create date features
        df = self.create_date_features(df, date_col)
        
        # Create lag features
        df = self.create_lag_features(df, target_col, group_cols)
        
        # Create rolling features
        df = self.create_rolling_features(df, target_col, group_cols)
        
        # Create cyclical features
        df = self.create_cyclical_features(df, date_col)
        
        # Create interaction features if categorical columns provided
        if categorical_cols:
            df = self.create_interaction_features(df, categorical_cols)
        
        # Skip advanced features for now to reduce complexity
        # df = self.create_advanced_features(df, target_col, date_col, group_cols)
        
        # Handle missing values created by lag and rolling features
        df = self.handle_missing_values(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # For lag and rolling features, forward fill or use mean
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().any():
                if 'lag' in col or 'rolling' in col:
                    # For time-based features, forward fill then backward fill
                    df[col] = df[col].ffill().bfill()
                else:
                    # For other features, use mean
                    df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str,
                       importance_threshold: float = 0.001) -> List[str]:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data for feature selection
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Train random forest for feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select features above threshold
        selected_features = feature_importance[
            feature_importance['importance'] >= importance_threshold
        ]['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
        return selected_features
    
    def create_advanced_features(self, df: pd.DataFrame, target_col: str,
                                date_col: str, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Create advanced features for better model performance"""
        df = df.copy()
        
        # Exponentially weighted moving averages (more weight on recent data)
        ewm_spans = [7, 14]  # Reduced spans to avoid overfitting
        for span in ewm_spans:
            if group_cols:
                df[f'{target_col}_ewm_{span}'] = df.groupby(group_cols)[target_col].transform(
                    lambda x: x.ewm(span=span, adjust=False).mean()
                )
            else:
                df[f'{target_col}_ewm_{span}'] = df[target_col].ewm(span=span, adjust=False).mean()
        
        # Trend features
        if group_cols:
            # Linear trend within groups
            df['trend'] = df.groupby(group_cols).cumcount()
            df['trend_squared'] = df['trend'] ** 2
        else:
            df['trend'] = np.arange(len(df))
            df['trend_squared'] = df['trend'] ** 2
        
        # Sales velocity (rate of change)
        df[f'{target_col}_velocity'] = df[target_col].diff()
        df[f'{target_col}_acceleration'] = df[f'{target_col}_velocity'].diff()
        
        # Ratio features
        for window in [7, 30]:
            rolling_mean = df[target_col].rolling(window, min_periods=1).mean()
            df[f'{target_col}_ratio_to_{window}d_avg'] = df[target_col] / (rolling_mean + 1)
        
        # Day of month features
        df['day_of_month'] = df[date_col].dt.day
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        # Week of month
        df['week_of_month'] = (df['day_of_month'] - 1) // 7 + 1
        
        # Business quarter features
        df['quarter_progress'] = (df[date_col].dt.month - 1) % 3 + 1
        df['is_quarter_end'] = (df['quarter_progress'] == 3).astype(int)
        
        # Add carefully selected features that improve time series prediction
        if 'has_promotion' in df.columns and 'is_weekend' in df.columns:
            # Simple interaction between promotion and weekend
            df['promotion_weekend'] = df['has_promotion'] * df['is_weekend']
        
        # Ratio features that capture relative performance
        for window in [7, 30]:
            rolling_mean = df[target_col].rolling(window, min_periods=1).mean()
            df[f'{target_col}_ratio_to_{window}d'] = df[target_col] / (rolling_mean + 1)
        
        # Days since month start (useful for monthly patterns)
        df['days_since_month_start'] = df['day_of_month']
        
        logger.info("Created advanced features")
        return df
    
    def create_target_encoding(self, df: pd.DataFrame, target_col: str, 
                              categorical_cols: List[str], smoothing: float = 1.0) -> pd.DataFrame:
        """Create target encoding for categorical variables with smoothing"""
        df = df.copy()
        
        for col in categorical_cols:
            # Calculate mean target for each category
            mean_target = df.groupby(col)[target_col].mean()
            global_mean = df[target_col].mean()
            
            # Calculate counts for smoothing
            counts = df[col].value_counts()
            
            # Apply smoothing to prevent overfitting
            smooth_mean = {}
            for cat in counts.index:
                n = counts[cat]
                smooth_mean[cat] = (n * mean_target[cat] + smoothing * global_mean) / (n + smoothing)
            
            # Create new feature
            df[f'{col}_target_encoded'] = df[col].map(smooth_mean)
            
            # Handle unknown categories
            df[f'{col}_target_encoded'].fillna(global_mean, inplace=True)
        
        logger.info(f"Created target encoding for {len(categorical_cols)} categorical features")
        return df