import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from include.feature_engineering.feature_pipeline import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample sales data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
    
    data = []
    for date in dates:
        for store in ['store_001', 'store_002']:
            for product in ['prod_A', 'prod_B']:
                sales = np.random.uniform(500, 1500)
                data.append({
                    'date': date,
                    'store_id': store,
                    'product_id': product,
                    'sales': sales,
                    'price': np.random.uniform(10, 100)
                })
    
    return pd.DataFrame(data)


@pytest.fixture
def feature_engineer():
    """Create FeatureEngineer instance"""
    # Use test config
    return FeatureEngineer()


class TestFeatureEngineer:
    
    def test_create_date_features(self, feature_engineer, sample_data):
        """Test date feature creation"""
        df = feature_engineer.create_date_features(sample_data, 'date')
        
        # Check if date features are created
        expected_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'weekofyear', 'is_weekend', 'is_holiday']
        for feature in expected_features:
            assert feature in df.columns, f"Missing date feature: {feature}"
        
        # Validate feature values
        assert df['year'].iloc[0] == 2023
        assert df['month'].min() >= 1 and df['month'].max() <= 12
        assert df['is_weekend'].isin([0, 1]).all()
    
    def test_create_lag_features(self, feature_engineer, sample_data):
        """Test lag feature creation"""
        df = feature_engineer.create_lag_features(
            sample_data, 
            target_col='sales',
            group_cols=['store_id', 'product_id']
        )
        
        # Check if lag features are created
        expected_lags = [1, 7, 14, 30]
        for lag in expected_lags:
            assert f'sales_lag_{lag}' in df.columns, f"Missing lag feature: sales_lag_{lag}"
        
        # Verify lag calculation
        # Group by store and product, then check first non-null lag value
        for (store, product), group in df.groupby(['store_id', 'product_id']):
            lag_1_values = group['sales_lag_1'].dropna()
            if len(lag_1_values) > 0:
                original_idx = group.index[1]  # Second row should have lag_1
                lagged_idx = group.index[0]    # First row value
                assert group.loc[original_idx, 'sales_lag_1'] == group.loc[lagged_idx, 'sales']
    
    def test_create_rolling_features(self, feature_engineer, sample_data):
        """Test rolling feature creation"""
        df = feature_engineer.create_rolling_features(
            sample_data,
            target_col='sales',
            group_cols=['store_id', 'product_id']
        )
        
        # Check if rolling features are created
        windows = [7, 14, 30]
        functions = ['mean', 'std', 'min', 'max']
        
        for window in windows:
            for func in functions:
                feature_name = f'sales_rolling_{window}_{func}'
                assert feature_name in df.columns, f"Missing rolling feature: {feature_name}"
        
        # Verify rolling calculations are not all null
        rolling_features = [col for col in df.columns if 'rolling' in col]
        for feature in rolling_features:
            assert df[feature].notna().sum() > 0, f"Rolling feature {feature} is all null"
    
    def test_create_cyclical_features(self, feature_engineer, sample_data):
        """Test cyclical feature creation"""
        df = feature_engineer.create_cyclical_features(sample_data, 'date')
        
        # Check if cyclical features are created
        cyclical_features = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos']
        for feature in cyclical_features:
            assert feature in df.columns, f"Missing cyclical feature: {feature}"
        
        # Verify cyclical values are in correct range [-1, 1]
        for feature in cyclical_features:
            assert df[feature].min() >= -1 and df[feature].max() <= 1, \
                f"Cyclical feature {feature} has values outside [-1, 1]"
    
    def test_create_all_features(self, feature_engineer, sample_data):
        """Test complete feature engineering pipeline"""
        df = feature_engineer.create_all_features(
            sample_data,
            target_col='sales',
            date_col='date',
            group_cols=['store_id', 'product_id'],
            categorical_cols=['store_id', 'product_id']
        )
        
        # Check that we have more features than original
        assert len(df.columns) > len(sample_data.columns), "No new features created"
        
        # Check for no missing values in original columns
        original_cols = sample_data.columns
        for col in original_cols:
            assert df[col].notna().all(), f"Missing values introduced in {col}"
        
        # Verify data integrity
        assert len(df) == len(sample_data), "Row count changed during feature engineering"
    
    def test_handle_missing_values(self, feature_engineer, sample_data):
        """Test missing value handling"""
        # Create data with missing values
        df = sample_data.copy()
        df.loc[0:5, 'sales'] = np.nan
        df['test_feature'] = np.nan
        
        df_filled = feature_engineer.handle_missing_values(df)
        
        # Check no missing values remain in numeric columns
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert df_filled[col].notna().all(), f"Missing values remain in {col}"
    
    def test_feature_selection(self, feature_engineer, sample_data):
        """Test feature selection"""
        # Create features first
        df = feature_engineer.create_all_features(
            sample_data,
            target_col='sales',
            date_col='date'
        )
        
        # Select features
        selected_features = feature_engineer.select_features(
            df,
            target_col='sales',
            importance_threshold=0.01
        )
        
        # Verify output
        assert isinstance(selected_features, list), "Feature selection should return a list"
        assert len(selected_features) > 0, "No features selected"
        assert 'sales' not in selected_features, "Target column should not be in selected features"